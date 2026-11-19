// A clock that works natively and in the browser.
use web_time::Instant;

// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// What the decoder shows after each part.
use crate::block::BlockOutput;
// The function that compares two lists.
use crate::max_error::max_abs_error;
// [-1, 1] planes to RGBA bytes.
use crate::image::image_to_rgba;
// The weight reader.
use crate::weights::WeightSource;

// The decoder itself.
use super::{TAESD_PARTS, run_taesd};

// The image is 512 x 512 pixels.
const IMAGE_SIDE: usize = 512;

// What the TAESD page shows.
pub struct TaesdReport {
    // One line per part, then the error of the image, then the total time.
    pub lines: Vec<String>,
    // The image as RGBA bytes, 512 x 512 x 4, ready for a canvas.
    pub rgba: Vec<u8>,
}

// Decode `latent`, record one line per part, compare the image with `expected` (the diffusers image), and make the RGBA bytes.
// The page and the native example both call this, so they print the same lines.
pub async fn decode_and_report(gpu: &Gpu, weights: &impl WeightSource, latent: &[f32], expected: &[f32]) -> Result<TaesdReport, EngineError> {
    // Start the clock before the latent goes to the GPU.
    let start = Instant::now();
    // One line per part, filled by the closure below.
    let mut lines = Vec::new();
    // Run the decoder. After each part, the closure writes one line. It never fails, so the run stops only on an engine error.
    let image = run_taesd(gpu, weights, latent, &TAESD_PARTS, |block| {
        // Keep the line.
        lines.push(part_line(block));
        // Go on to the next part.
        Ok::<(), EngineError>(())
    })
    .await?;
    // The time from the latent upload to the image on the CPU. It includes shader compiles, weight uploads and 9 read-backs.
    let total_seconds = start.elapsed().as_secs_f64();
    // How far the engine image is from the diffusers image.
    let error = max_abs_error(&image, expected)?;
    // The error line.
    lines.push(format!("taesd_image: largest error {error:e} against diffusers"));
    // The time line.
    lines.push(format!("total {:.1} ms from latent upload to image on the CPU, with 9 read-backs", total_seconds * 1000.0));
    // The bytes for the canvas.
    let rgba = image_to_rgba(&image, IMAGE_SIDE, IMAGE_SIDE)?;
    // Everything the page shows.
    Ok(TaesdReport { lines, rgba })
}

// One line for one part: its submits, its longest submit and the peak of GPU buffer bytes so far.
fn part_line(block: &BlockOutput) -> String {
    // The same words as the native test, without the golden comparison.
    format!("{}: {} submits, longest {:.1} ms, peak {} bytes", block.name, block.submits, block.longest_submit_seconds * 1000.0, block.peak_bytes)
}
