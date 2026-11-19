// plan.rs records the text encoder forward pass for the activation plan.
mod plan;
// embed.rs computes the input of layer 0 on the CPU.
mod embed;
// bind.rs binds every op to the slots of the plan.
mod bind;

// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The code that shows a part to the caller.
use crate::block::{BlockOutput, check_tensor};
// Buffers that count their bytes.
use crate::memory::GpuBuffer;
// The buffers of the activation plan.
use crate::plan::ActivationPool;
// One submit, timed and checked against the 0.25 s limit.
use crate::submit::submit_within_limit;
// One submit of a stage.
use crate::step::Step;
// The weight reader, the weights on the GPU, and the weight buffers of the kernels.
use crate::weights::{GpuWeights, Uploading, WeightBuffer, WeightBuffers, WeightSource};

// The planned graph, and the names of the input and the output.
use plan::{EMBEDDINGS, OUTPUT, planned_text};
// The token rows.
use embed::{TOKEN_TABLE, copy_token_rows};
// The binder.
use bind::bind_text;

// Stage 1 of a generation: the text encoder (CLIPTextModel) with all its buffers and bind groups, made once.
// `prepare` makes the slot buffers of the activation plan, gets the weights and binds every kernel.
// `run` copies the token rows of the prompt, submits the steps, and returns last_hidden_state [1, 77, 1024].
// Dropping the stage frees all of it, before the UNet stage starts.
pub struct TextStage {
    // One buffer per slot of the activation plan.
    pool: ActivationPool,
    // The bytes of all slots: the planned peak of the activations.
    planned_bytes: u64,
    // Every submit, in run order.
    steps: Vec<Step>,
    // The rows of the token table.
    table_rows: usize,
    // The token table, when the stage made its own copy of it (from a CPU reader).
    token_table: Option<GpuBuffer>,
    // The weight file on the GPU that the kernels read, when the stage was prepared from one. Dropping the stage frees it.
    // It is the last field, so Rust drops it after the steps that bind its buffers.
    _weights: Option<GpuWeights>,
}

impl TextStage {
    // Plan the activations, make their buffers, read the float32 weights from a CPU reader, and bind the kernels.
    // Each kernel gets its own GPU copy of its weights, and the stage gets a copy of the token table.
    pub fn prepare(gpu: &Gpu, weights: &impl WeightSource) -> Result<Self, EngineError> {
        // The uploader over the reader.
        let uploading = Uploading(weights);
        // Plan and make the slots.
        let (pool, planned_bytes) = planned_pool(gpu);
        // Bind every op.
        let steps = bind_text(gpu, &uploading, &pool)?;
        // The token table on the GPU, 202 MB.
        let table = uploading.float(gpu, TOKEN_TABLE)?;
        // It was made for this stage, so the stage keeps it.
        let WeightBuffer::Owned(token_table) = table.buffer else { return Err(EngineError::MissingWeight(TOKEN_TABLE.to_string())) };
        // The stage, ready to run.
        Ok(Self { planned_bytes, pool, steps, table_rows: table.shape[0], token_table: Some(token_table), _weights: None })
    }

    // The same, with the text encoder file on the GPU already, from the streaming loader. The browser uses this.
    // The kernels and the token row copies read its buffers. The stage keeps the weights, so dropping the stage frees them.
    pub fn prepare_loaded(gpu: &Gpu, weights: GpuWeights) -> Result<Self, EngineError> {
        // Plan and make the slots.
        let (pool, planned_bytes) = planned_pool(gpu);
        // Bind every op to the loaded buffers.
        let steps = bind_text(gpu, &weights, &pool)?;
        // The token table must be float32 rows, because the run copies its bytes.
        let table = weights.tensor(TOKEN_TABLE)?;
        if table.dtype != "float32" {
            return Err(EngineError::WeightDtype { name: TOKEN_TABLE.to_string(), dtype: table.dtype.clone() });
        }
        // The rows of the table.
        let table_rows = table.shape[0];
        // The stage, ready to run.
        Ok(Self { planned_bytes, pool, steps, table_rows, token_table: None, _weights: Some(weights) })
    }

    // The bytes of all slot buffers. They live for the whole stage, so this is the planned peak of the activations.
    pub fn planned_bytes(&self) -> u64 {
        self.planned_bytes
    }

    // The bytes of the output that `run` copies back. That copy is the only buffer a run without checks makes.
    pub fn output_bytes(&self) -> Result<u64, EngineError> {
        // The output size in the plan, 4 bytes per f32 value.
        Ok(self.pool.tensor(OUTPUT)?.dims.iter().map(|&size| u64::from(size)).product::<u64>() * 4)
    }

    // The token table: the stage's own copy, or the buffer of the loaded weights.
    fn token_table(&self) -> Result<&wgpu::Buffer, EngineError> {
        // One of the two exists, from prepare or prepare_loaded.
        match (&self.token_table, &self._weights) {
            (Some(table), _) => Ok(table),
            (None, Some(weights)) => Ok(&weights.tensor(TOKEN_TABLE)?.buffer),
            (None, None) => Err(EngineError::MissingWeight(TOKEN_TABLE.to_string())),
        }
    }

    // Run the text encoder on 77 token ids and return last_hidden_state, 77 x 1024 values.
    // After a step that ends a part named in `wanted`, `check` sees that part.
    pub async fn run<E: From<EngineError>>(&self, gpu: &Gpu, ids: &[u32], wanted: &[&str], mut check: impl FnMut(&BlockOutput) -> Result<(), E>) -> Result<Vec<f32>, E> {
        // The token rows of the prompt, into the embeddings slot. The first step adds the positions.
        copy_token_rows(gpu, self.token_table()?, self.table_rows, ids, self.pool.tensor(EMBEDDINGS)?.buffer)?;
        // The seconds of the submits since the last check that ran.
        let mut seconds = Vec::new();
        // Every step, in order.
        for step in &self.steps {
            // One submit, with the bind groups from `prepare`.
            seconds.push(submit_within_limit(gpu, &step.label, &step.dispatches()).await?);
            // The checks of this step that the caller wants.
            for expected in step.checks.iter().filter(|expected| wanted.contains(&expected.golden.as_str())) {
                // The slot of the checked tensor.
                let tensor = self.pool.tensor(&expected.tensor)?;
                // Compare with the golden tensor.
                check_tensor(gpu, &expected.golden, tensor.buffer, &expected.layout.torch_shape(tensor.dims), &seconds, &mut check).await?;
                // The next check counts its own submits.
                seconds.clear();
            }
        }
        // The output slot.
        let output = self.pool.tensor(OUTPUT)?;
        // Copy only the output values back. The UNet stage needs them after this stage is freed.
        Ok(gpu.read_f32_prefix(output.buffer, output.dims.iter().map(|&size| size as usize).product()).await?)
    }
}

// The activation plan of the text encoder, its slot buffers, and their bytes.
fn planned_pool(gpu: &Gpu) -> (ActivationPool, u64) {
    // The forward pass and the slot of each of its tensors.
    let (graph, plan) = planned_text();
    // Make the slot buffers once.
    (ActivationPool::new(gpu, &graph, &plan), plan.total_bytes())
}
