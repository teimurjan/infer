// activation.rs holds the tensor type that flows between layers on the GPU.
// The shared GPU base, under the same names as before, so `crate::gpu::Gpu` and the other paths keep working.
use engine_core::{bench, bytes, checks, dispatch, error, gpu, layers, max_error, memory, plan, profile, submit, weights};
mod activation;
// attention/ holds the attention kernel and its test cases.
mod attention;
// bench/ holds the copy and math benchmarks.
// block.rs holds a part of a network that ran, and shows its output to the caller.
mod block;
// bytes.rs turns raw bytes into f32 values.
// cases.rs lists the real SD-Turbo layers we test and checks them.
mod cases;
// checks.rs holds the size checks that several kernels share.
// conv.rs connects a 3x3 convolution whose weights come from the weight file.
mod conv;
// dispatch.rs compiles a kernel, connects its buffers and records dispatches.
// error.rs lists every way the engine can fail.
// gpu.rs opens the GPU and moves data to it and back.
// image.rs turns the decoded image into RGBA bytes for a canvas.
mod image;
// linear.rs connects linear layers whose weights come from the weight file to the slots of the plan.
mod linear;
// kernels/ holds the norms, adds, GEGLU, GELU, concat and upsample on the slots of a plan, for the UNet and the text encoder.
mod kernels;
// step.rs holds one submit of a stage and the tensors it checks.
mod step;
// layers/ holds the linear and convolution kernels.
// memory/ counts the bytes of our GPU buffers per category, and the peak.
// max_error.rs measures how far the GPU result is from the expected result.
// profile/ times each kernel and prints the table of where the time goes.
// plan/ works out the lifetime of each tensor of a forward pass and puts the tensors into reusable buffers.
// noise/ makes the starting noise of a seed, the same as torch.randn on the CPU.
mod noise;
// packed_cases.rs lists the real SD-Turbo layers that we test with packed 4-bit and 8-bit weights.
mod packed_cases;
// silu/ holds the SiLU kernel and the Rust code that runs it.
mod silu;
// submit.rs times one submit and checks it against the 0.25 s limit.
// taesd/ runs the TAESD decoder, which turns a latent into an image.
mod taesd;
// pipeline/ runs the three stages one after the other, with the scheduler step between the UNet and the decoder.
mod pipeline;
// text/ runs the text encoder stage.
mod text;
// tokenizer/ turns a prompt into the 77 token ids of the text encoder.
mod tokenizer;
// unet/ runs the SD-Turbo UNet block by block.
mod unet;
// weights/ reads the flat weight files that convert_weights.py writes.
// web.rs is the entry point for the browser. It is only built for WASM.
#[cfg(target_arch = "wasm32")]
mod web;

// The parts that the native tests and the browser entry use.
pub use attention::{ATTENTION_CASES, AttentionCase, AttentionReport, AttentionShape, AttentionTensors, check_attention, find_attention_case};
pub use bench::{BenchReport, copy_benchmark, run_benchmarks};
pub use profile::{Profile, ProfileMode, ProfileRow, ProfileTable, best_mode, start_profile};
pub use bytes::f32_from_le_bytes;
pub use cases::{CASES, Case, CaseReport, check_case, find_case};
pub use error::{EngineError, SubmitPhase};
pub use gpu::Gpu;
pub use layers::{Conv1x1Shape, Conv3x3Shape, LayerShape, LinearShape, PackedTensors, Tensors, WeightBits};
pub use packed_cases::{PACKED_CASES, PackedCase, check_packed_case};
pub use max_error::max_abs_error;
pub use silu::{run_silu, silu_max_error};
pub use block::BlockOutput;
pub use image::image_to_rgba;
pub use taesd::{TAESD_PARTS, TaesdReport, decode_and_report, run_taesd};
pub use unet::{ActivationPlan, UnetInputs, UnetStage, plan_unet_activations};
pub use text::TextStage;
pub use tokenizer::{CONTEXT_IDS, Tokenizer};
pub use noise::{NOISE_VALUES, starting_noise};
pub use pipeline::{FromReader, GenerateInputs, Generated, Loaded, Stage, StageEvent, StageWeights, TextWeights, UnetWeights, decoder_stage, euler_step, generate, scale_input, text_stage, unet_stage};
#[cfg(not(target_arch = "wasm32"))]
pub use pipeline::StreamedFile;
pub use memory::{MemoryCategory, MemoryReport};
pub use weights::{GpuTensor, GpuWeights, WeightBytes, WeightSource, WeightTensor, WeightUpload, weight_bytes_on_gpu};
#[cfg(not(target_arch = "wasm32"))]
pub use weights::{WeightFile, load_to_gpu};
