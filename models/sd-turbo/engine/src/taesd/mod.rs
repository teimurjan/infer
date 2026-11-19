// kernels.rs connects the small TAESD kernels: clamp, ReLU, add + ReLU, upsample and image.
mod kernels;
// steps.rs runs the parts of the decoder, each in one or more submits.
mod steps;
// report.rs runs the decoder for the web page and turns the result into lines and RGBA bytes.
mod report;

// The report for the web page.
pub use report::{TaesdReport, decode_and_report};
// The nearest 2x upsample shader, which the UNet reuses.
pub(crate) use kernels::UPSAMPLE as UPSAMPLE_SHADER;

// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The tensor type that flows between layers.
use crate::activation::Activation;
// A part of the network that ran, and the code that shows it to the caller.
use crate::block::{Block, BlockOutput, check_block};
// The weight reader.
use crate::weights::WeightSource;

// The decoder input has 4 latent channels (latent_channels in the TAESD config).
const LATENT_CHANNELS: u32 = 4;
// A 512 x 512 image comes from a 64 x 64 latent.
const LATENT_SIDE: u32 = 64;
// Every part that `run_taesd` can show, in run order. They are the names of the golden files.
pub const TAESD_PARTS: [&str; 9] = ["taesd_conv_in", "taesd_blocks_0", "taesd_upsample_0", "taesd_blocks_1", "taesd_upsample_1", "taesd_blocks_2", "taesd_upsample_2", "taesd_blocks_3", "taesd_image"];

// Decode `latent` [1, 4, 64, 64] with the TAESD decoder (madebyollin/taesd) into an image [1, 3, 512, 512] in [-1, 1], and return the image values.
// The latent is the SD-Turbo latent as the pipeline gives it to the VAE. TAESD's scaling_factor is 1.0, so there is no division.
// After each part with a golden tensor, the output goes back to the CPU and `check` sees it. An error from `check` stops the run there.
// The numbers are the positions in diffusers' DecoderTiny.layers, which is also the order of the weight names.
// Only the parts named in `wanted` are copied back and shown. A generation wants none: then the only copy is the image itself.
// Each copy of a block at 512 x 512 is 67 MB in WASM memory, and WASM memory never shrinks, so unwanted copies cost the page for its whole life.
pub async fn run_taesd<E: From<EngineError>>(gpu: &Gpu, weights: &impl WeightSource, latent: &[f32], wanted: &[&str], mut check: impl FnMut(&BlockOutput) -> Result<(), E>) -> Result<Vec<f32>, E> {
    // Copy the latent to the GPU.
    let latent = Activation::upload(gpu, "taesd latent", latent, [LATENT_CHANNELS, LATENT_SIDE, LATENT_SIDE])?;
    // Clamp, layer 0 (conv 4 to 64), layer 1 (ReLU). 64 x 64.
    let hidden = checked(gpu, "taesd_conv_in", steps::conv_in(gpu, weights, latent).await?, wanted, &mut check).await?;
    // Layers 2 to 4: three blocks at 64 x 64.
    let hidden = checked(gpu, "taesd_blocks_0", steps::tiny_blocks(gpu, weights, 2..=4, hidden).await?, wanted, &mut check).await?;
    // Layer 5 (upsample) and layer 6 (conv). 128 x 128.
    let hidden = checked(gpu, "taesd_upsample_0", steps::upsample_conv(gpu, weights, 6, hidden).await?, wanted, &mut check).await?;
    // Layers 7 to 9: three blocks at 128 x 128.
    let hidden = checked(gpu, "taesd_blocks_1", steps::tiny_blocks(gpu, weights, 7..=9, hidden).await?, wanted, &mut check).await?;
    // Layer 10 (upsample) and layer 11 (conv). 256 x 256.
    let hidden = checked(gpu, "taesd_upsample_1", steps::upsample_conv(gpu, weights, 11, hidden).await?, wanted, &mut check).await?;
    // Layers 12 to 14: three blocks at 256 x 256.
    let hidden = checked(gpu, "taesd_blocks_2", steps::tiny_blocks(gpu, weights, 12..=14, hidden).await?, wanted, &mut check).await?;
    // Layer 15 (upsample) and layer 16 (conv). 512 x 512.
    let hidden = checked(gpu, "taesd_upsample_2", steps::upsample_conv(gpu, weights, 16, hidden).await?, wanted, &mut check).await?;
    // Layer 17: one block at 512 x 512.
    let hidden = checked(gpu, "taesd_blocks_3", steps::tiny_blocks(gpu, weights, 17..=17, hidden).await?, wanted, &mut check).await?;
    // Layer 18 (conv 64 to 3), then x * 2 - 1.
    let image = steps::conv_out(gpu, weights, 18, hidden).await?;
    // Show the image to `check` with the golden name if it is wanted, and return its values.
    if wanted.contains(&"taesd_image") {
        return check_block(gpu, "taesd_image", &image, &mut check).await;
    }
    // Otherwise only copy it back.
    Ok(gpu.read_f32(&image.output.buffer).await?)
}

// Show a block to `check` if it is wanted, then give its output back for the next part.
async fn checked<E: From<EngineError>>(gpu: &Gpu, name: &'static str, block: Block, wanted: &[&str], check: &mut impl FnMut(&BlockOutput) -> Result<(), E>) -> Result<Activation, E> {
    // Copy the output back and compare, if the caller wants this part.
    if wanted.contains(&name) {
        check_block(gpu, name, &block, check).await?;
    }
    // The output goes on to the next part.
    Ok(block.output)
}
