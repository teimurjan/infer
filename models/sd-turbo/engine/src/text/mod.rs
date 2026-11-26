// plan.rs records the text encoder forward pass for the activation plan.
mod plan;
// embed.rs computes the input of layer 0 on the CPU.
mod embed;
// bind.rs binds every op to the slots of the plan.
mod bind;
// token_rows.rs unpacks the prompt's rows of a packed token table.
mod token_rows;

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
// The weight reader, the weights on the GPU, the weight buffers of the kernels, and the two kinds of matrix.
use crate::weights::{GpuWeights, Matrix, Uploading, WeightBuffer, WeightBuffers, WeightSource};

// The planned graph, and the names of the input and the output.
use plan::{EMBEDDINGS, OUTPUT, planned_text};
// The token rows.
use embed::{TOKEN_TABLE, copy_token_rows};
// The binder.
use bind::bind_text;
// The kernel of a packed token table.
use token_rows::PackedTokenRows;

// How a run puts the prompt's token rows into the embeddings slot.
enum TokenRows {
    // Buffer copies from a float32 table: the stage's own copy (from a CPU reader), or the buffer of the loaded weights.
    Copy { table_rows: usize, own_table: Option<GpuBuffer> },
    // The unpacking kernel, bound to a packed table of the loaded weights.
    Packed(PackedTokenRows),
}

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
    // How the run writes the token rows.
    token_rows: TokenRows,
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
        Ok(Self { planned_bytes, pool, steps, token_rows: TokenRows::Copy { table_rows: table.shape[0], own_table: Some(token_table) }, _weights: None })
    }

    // The same, with the text encoder file on the GPU already, from the streaming loader. The browser uses this.
    // The kernels and the token row copies read its buffers. The stage keeps the weights, so dropping the stage frees them.
    pub fn prepare_loaded(gpu: &Gpu, weights: GpuWeights) -> Result<Self, EngineError> {
        // Plan and make the slots.
        let (pool, planned_bytes) = planned_pool(gpu);
        // Bind every op to the loaded buffers.
        let steps = bind_text(gpu, &weights, &pool)?;
        // A float32 table is copied row by row. A packed one needs the unpacking kernel.
        let token_rows = loaded_token_rows(gpu, &weights, &pool)?;
        // The stage, ready to run.
        Ok(Self { planned_bytes, pool, steps, token_rows, _weights: Some(weights) })
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

    // Put the token rows of `ids` into the embeddings slot, with buffer copies or with the unpacking kernel.
    fn write_token_rows(&self, gpu: &Gpu, ids: &[u32]) -> Result<(), EngineError> {
        // The slot that layer 0 reads.
        let embeddings = self.pool.tensor(EMBEDDINGS)?.buffer;
        match &self.token_rows {
            // The stage's own float32 table, from prepare.
            TokenRows::Copy { table_rows, own_table: Some(table) } => copy_token_rows(gpu, table, *table_rows, ids, embeddings),
            // The float32 table of the loaded weights, from prepare_loaded.
            TokenRows::Copy { table_rows, own_table: None } => {
                let weights = self._weights.as_ref().ok_or_else(|| EngineError::MissingWeight(TOKEN_TABLE.to_string()))?;
                copy_token_rows(gpu, &weights.tensor(TOKEN_TABLE)?.buffer, *table_rows, ids, embeddings)
            }
            // The packed table of the loaded weights.
            TokenRows::Packed(kernel) => kernel.run(gpu, ids),
        }
    }

    // Run the text encoder on 77 token ids and return last_hidden_state, 77 x 1024 values.
    // After a step that ends a part named in `wanted`, `check` sees that part.
    pub async fn run<E: From<EngineError>>(&self, gpu: &Gpu, ids: &[u32], wanted: &[&str], mut check: impl FnMut(&BlockOutput) -> Result<(), E>) -> Result<Vec<f32>, E> {
        // The token rows of the prompt, into the embeddings slot. The first step adds the positions.
        self.write_token_rows(gpu, ids)?;
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

// How a stage with loaded weights writes its token rows: copies of a float32 table, or the kernel of a packed one.
fn loaded_token_rows(gpu: &Gpu, weights: &GpuWeights, pool: &ActivationPool) -> Result<TokenRows, EngineError> {
    // The table, float or packed, with its shape.
    let table = weights.matrix(gpu, TOKEN_TABLE)?;
    let table_rows = table.shape[0];
    match table.matrix {
        // The run copies the bytes of a float32 table. A float16 one was widened into a new buffer, which this path does not keep.
        Matrix::Float(WeightBuffer::Borrowed(_)) => Ok(TokenRows::Copy { table_rows, own_table: None }),
        Matrix::Float(WeightBuffer::Owned(_)) => Err(EngineError::WeightDtype { name: TOKEN_TABLE.to_string(), dtype: weights.tensor(TOKEN_TABLE)?.dtype.clone() }),
        // The kernel unpacks the rows into the embeddings slot.
        Matrix::Packed { words, scale, bits, groups: 1 } => Ok(TokenRows::Packed(PackedTokenRows::bind(gpu, words, scale, bits, table_rows, pool.tensor(EMBEDDINGS)?.buffer)?)),
        // The token rows kernel reads one scale per row.
        Matrix::Packed { groups, .. } => Err(EngineError::ScaleGroups { groups, rule: "but the token rows kernel reads one scale per row" }),
    }
}

// The activation plan of the text encoder, its slot buffers, and their bytes.
fn planned_pool(gpu: &Gpu) -> (ActivationPool, u64) {
    // The forward pass and the slot of each of its tensors.
    let (graph, plan) = planned_text();
    // Make the slot buffers once.
    (ActivationPool::new(gpu, &graph, &plan), plan.total_bytes())
}
