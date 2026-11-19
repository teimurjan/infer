// schedule.rs holds the one Euler step on the CPU.
mod schedule;
// event.rs holds what the pipeline reports about each stage.
mod event;
// sources.rs says where each planned stage gets its weights.
mod sources;

// A clock that works natively and in the browser.
use web_time::Instant;

// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The decoder.
use crate::taesd::run_taesd;
// The UNet inputs.
use crate::unet::UnetInputs;
// The weight reader.
use crate::weights::WeightSource;

// The reports of the stages.
pub use event::{Stage, StageEvent};
// The scheduler step.
pub use schedule::{euler_step, scale_input};
// The sources of the weights.
pub use sources::{FromReader, Loaded, TextWeights, UnetWeights};
#[cfg(not(target_arch = "wasm32"))]
pub use sources::StreamedFile;

// What one generation reads.
pub struct GenerateInputs<'a> {
    // The 77 token ids of the prompt, from the CLIP tokenizer.
    pub token_ids: &'a [u32],
    // The starting noise, [1, 4, 64, 64], unit variance.
    pub noise: &'a [f32],
    // The noise level of the one step (14.614646911621094 for SD-Turbo).
    pub sigma: f32,
    // The timestep of the one step (999 for SD-Turbo).
    pub timestep: f32,
}

// The weights of the three stages.
pub struct StageWeights<'a, T: TextWeights, U: UnetWeights, D: WeightSource> {
    // The text encoder.
    pub text: T,
    // The UNet: float32 from a CPU reader, or the packed file.
    pub unet: U,
    // The TAESD decoder.
    pub decoder: &'a D,
}

// What one generation gives back. The tensors between the stages are kept, so a test can compare each one.
pub struct Generated {
    // The text encoder output, [1, 77, 1024].
    pub text: Vec<f32>,
    // The UNet output, the predicted noise, [1, 4, 64, 64].
    pub noise_prediction: Vec<f32>,
    // The latent after the Euler step, [1, 4, 64, 64].
    pub latent: Vec<f32>,
    // The image, [1, 3, 512, 512], in [-1, 1].
    pub image: Vec<f32>,
}

// Run the three stages one after the other: text encoder, UNet, TAESD decoder. Each stage gets its weights, runs, and is
// dropped, which frees every GPU buffer of it, before the next stage starts. Only the small outputs go from stage to stage, on the CPU.
// `report` sees each stage when it is ready, when it ran, and when it was freed.
pub async fn generate<T: TextWeights, U: UnetWeights, D: WeightSource>(gpu: &Gpu, weights: StageWeights<'_, T, U, D>, inputs: &GenerateInputs<'_>, mut report: impl FnMut(&StageEvent)) -> Result<Generated, EngineError> {
    // Stage 1: the text encoder.
    let text = text_stage(gpu, weights.text, inputs.token_ids, &mut report).await?;
    // Stage 2: the UNet on the scaled noise.
    let sample = scale_input(inputs.noise, inputs.sigma);
    let noise_prediction = unet_stage(gpu, weights.unet, &UnetInputs { sample: &sample, timestep: inputs.timestep, text: &text }, &mut report).await?;
    // The scheduler step, on the CPU.
    let latent = euler_step(inputs.noise, &noise_prediction, inputs.sigma);
    // Stage 3: the decoder.
    let image = decoder_stage(gpu, weights.decoder, &latent, &mut report).await?;
    // Everything.
    Ok(Generated { text, noise_prediction, latent, image })
}

// Stage 1: prepare, run and free the text encoder. Returns its output, 77 x 1024 values.
pub async fn text_stage(gpu: &Gpu, weights: impl TextWeights, ids: &[u32], report: &mut impl FnMut(&StageEvent)) -> Result<Vec<f32>, EngineError> {
    // The peak of this stage starts from what is live now. Weights that were loaded for it are live already.
    gpu.reset_peak_bytes();
    // Get the weights and make the slots and bind groups.
    let stage = weights.prepare(gpu).await?;
    // The planned peak: everything live now, plus the output copy of the run.
    report(&prepared(gpu, Stage::TextEncoder, stage.planned_bytes(), stage.output_bytes()?));
    // Run it without checks, as a real generation does.
    let start = Instant::now();
    let output = stage.run(gpu, ids, &[], |_| Ok::<(), EngineError>(())).await?;
    report(&StageEvent::Ran { stage: Stage::TextEncoder, measured_peak: gpu.peak_bytes(), seconds: start.elapsed().as_secs_f64() });
    // Free every buffer of the stage.
    drop(stage);
    report(&StageEvent::Freed { stage: Stage::TextEncoder, live: gpu.memory_report().total() });
    // The output goes to the UNet stage.
    Ok(output)
}

// Stage 2: prepare, run and free the UNet. Returns its output, the predicted noise.
pub async fn unet_stage(gpu: &Gpu, weights: impl UnetWeights, inputs: &UnetInputs<'_>, report: &mut impl FnMut(&StageEvent)) -> Result<Vec<f32>, EngineError> {
    // The peak of this stage starts from what is live now.
    gpu.reset_peak_bytes();
    // Get the weights and make the slots and bind groups.
    let stage = weights.prepare(gpu).await?;
    // The planned peak: everything live now, plus the output copy of the run.
    report(&prepared(gpu, Stage::Unet, stage.planned_bytes(), stage.output_bytes()?));
    // Run it without checks.
    let start = Instant::now();
    let output = stage.run(gpu, inputs, &[], |_| Ok::<(), EngineError>(())).await?;
    report(&StageEvent::Ran { stage: Stage::Unet, measured_peak: gpu.peak_bytes(), seconds: start.elapsed().as_secs_f64() });
    // Free every buffer of the stage.
    drop(stage);
    report(&StageEvent::Freed { stage: Stage::Unet, live: gpu.memory_report().total() });
    // The output goes to the scheduler step.
    Ok(output)
}

// Stage 3: run the TAESD decoder. It reads its weights block by block during the run and frees each block after it, so it has no plan.
pub async fn decoder_stage(gpu: &Gpu, weights: &impl WeightSource, latent: &[f32], report: &mut impl FnMut(&StageEvent)) -> Result<Vec<f32>, EngineError> {
    // The peak of this stage starts from what is live now.
    gpu.reset_peak_bytes();
    // Nothing is loaded before the run.
    report(&StageEvent::Prepared { stage: Stage::Decoder, planned_activations: None, planned_peak: None, live: gpu.memory_report().total() });
    // Decode without checks: the only copy back is the image.
    let start = Instant::now();
    let image = run_taesd(gpu, weights, latent, &[], |_| Ok::<(), EngineError>(())).await?;
    report(&StageEvent::Ran { stage: Stage::Decoder, measured_peak: gpu.peak_bytes(), seconds: start.elapsed().as_secs_f64() });
    // run_taesd drops its buffers before it returns.
    report(&StageEvent::Freed { stage: Stage::Decoder, live: gpu.memory_report().total() });
    // The image.
    Ok(image)
}

// The event of a stage that is ready to run: its plan, and the peak it will reach.
fn prepared(gpu: &Gpu, stage: Stage, planned_activations: u64, output_bytes: u64) -> StageEvent {
    // The weights, slots and uniforms that are live now.
    let live = gpu.memory_report().total();
    // A run without checks adds only its output copy.
    StageEvent::Prepared { stage, planned_activations: Some(planned_activations), planned_peak: Some(live + output_bytes), live }
}
