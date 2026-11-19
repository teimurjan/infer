// plan/ records the UNet forward pass and plans its activation buffers.
mod plan;
// time.rs computes the sinusoidal timestep features on the CPU.
mod time;
// resnet.rs binds one ResNet block to its slots.
mod resnet;
// transformer.rs binds one transformer to its slots.
mod transformer;
// down.rs binds one down level: its ResNet blocks, transformers and downsample.
mod down;
// middle.rs binds the middle block.
mod middle;
// up.rs binds one up level: its concats, ResNet blocks, transformers and upsample.
mod up;
// bind.rs binds the whole forward pass, level by level.
mod bind;

// The activation plan, for the report.
pub use plan::{ActivationPlan, plan_unet_activations};

// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The code that shows a block output to the caller.
use crate::block::{BlockOutput, check_tensor};
// The shared tensor length check.
use crate::checks::check_length;
// The buffers of the activation plan.
use crate::plan::ActivationPool;
// One submit, timed and checked against the 0.25 s limit.
use crate::submit::submit_within_limit;
// The weight reader, the weights on the GPU, and the uploader for a CPU reader.
use crate::weights::{GpuWeights, Uploading, WeightSource};

// The planned graph and its slots.
use plan::planned_unet;
// One submit of the stage.
use crate::step::Step;
// The timestep features.
use time::time_features;
// The forward pass binder and the names of the inputs and the output.
use bind::{OUTPUT, SAMPLE, TEXT, TIME_FEATURES, bind_unet};

// What the UNet reads in one run.
pub struct UnetInputs<'a> {
    // The scaled latent, [1, 4, 64, 64].
    pub sample: &'a [f32],
    // The timestep, 999 for SD-Turbo's one step.
    pub timestep: f32,
    // The text encoder output, [1, 77, 1024].
    pub text: &'a [f32],
}

// Stage 2 of a generation: the UNet with all its buffers and bind groups, made once.
// `prepare` makes the slot buffers of the activation plan, reads the weights, and binds every kernel into a list of steps.
// `run` only uploads its inputs, submits the steps and reads results back. It makes no buffer for activations and no bind group.
// Dropping the stage frees all of it, before the decoder stage starts.
pub struct UnetStage {
    // One buffer per slot of the activation plan.
    pool: ActivationPool,
    // The bytes of all slots: the planned peak of the activations.
    planned_bytes: u64,
    // Every submit, in run order.
    steps: Vec<Step>,
    // The weight file on the GPU that the kernels read, if the stage was prepared from one. Dropping the stage frees it.
    // It is the last field, so Rust drops it after the steps that bind its buffers.
    _weights: Option<GpuWeights>,
}

impl UnetStage {
    // Plan the activations, make their buffers, read the float32 weights from a CPU reader, and bind the kernels.
    // Each kernel gets its own GPU copy of its weights.
    pub fn prepare(gpu: &Gpu, weights: &impl WeightSource) -> Result<Self, EngineError> {
        // Plan and make the slots.
        let (pool, planned_bytes) = planned_pool(gpu);
        // Bind every op of the forward pass.
        let steps = bind_unet(gpu, &Uploading(weights), &pool)?;
        // The stage, ready to run.
        Ok(Self { planned_bytes, pool, steps, _weights: None })
    }

    // The same, with a weight file that is on the GPU already, such as the packed file from the streaming loader.
    // The kernels read its buffers directly: the packed words, the float16 row scales, and f32 copies of its small float16 tensors.
    // The stage keeps the weights, so dropping the stage frees them.
    pub fn prepare_loaded(gpu: &Gpu, weights: GpuWeights) -> Result<Self, EngineError> {
        // Plan and make the slots.
        let (pool, planned_bytes) = planned_pool(gpu);
        // Bind every op of the forward pass to the loaded buffers.
        let steps = bind_unet(gpu, &weights, &pool)?;
        // The stage, ready to run.
        Ok(Self { planned_bytes, pool, steps, _weights: Some(weights) })
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

    // Run the UNet on `inputs` and return its output, the predicted noise [1, 4, 64, 64].
    // After a step that ends a part named in `wanted`, the part's tensor goes back to the CPU and `check` sees it.
    // With an empty `wanted`, the only read-back is the output. An error from `check` stops the run there.
    pub async fn run<E: From<EngineError>>(&self, gpu: &Gpu, inputs: &UnetInputs<'_>, wanted: &[&str], mut check: impl FnMut(&BlockOutput) -> Result<(), E>) -> Result<Vec<f32>, E> {
        // Copy the three inputs into their slots. The first submit waits for these uploads.
        self.write_input(gpu, SAMPLE, inputs.sample)?;
        self.write_input(gpu, TEXT, inputs.text)?;
        self.write_input(gpu, TIME_FEATURES, &time_features(inputs.timestep))?;
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
        // Copy only the output values back. The scheduler step and the decoder stage need them.
        Ok(gpu.read_f32_prefix(output.buffer, output.dims.iter().map(|&size| size as usize).product()).await?)
    }

    // Check the length of one input and copy it into its slot.
    fn write_input(&self, gpu: &Gpu, name: &'static str, values: &[f32]) -> Result<(), EngineError> {
        // The slot of the input.
        let tensor = self.pool.tensor(name)?;
        // The input must have the planned number of values.
        check_length(name, values, tensor.dims.iter().map(|&size| size as usize).product())?;
        // Copy it.
        gpu.write_f32(tensor.buffer, values);
        // Done.
        Ok(())
    }
}

// The activation plan of the UNet, its slot buffers, and their bytes.
fn planned_pool(gpu: &Gpu) -> (ActivationPool, u64) {
    // The forward pass and the slot of each of its tensors.
    let (graph, plan) = planned_unet();
    // Make the slot buffers once.
    (ActivationPool::new(gpu, &graph, &plan), plan.total_bytes())
}
