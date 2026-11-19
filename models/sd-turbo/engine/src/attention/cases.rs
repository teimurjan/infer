// Display lets us print a report as one line.
use std::fmt;

// The shared timing loop and its report.
use crate::bench::{BenchReport, Kernel, measure};
// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;
// The function that compares two lists.
use crate::max_error::max_abs_error;

// The attention sizes, tensors and the code that prepares them.
use super::{AttentionShape, AttentionTensors, prepare_attention};

// Bytes in one megabyte, for the printed sizes.
const MEGABYTE: f64 = 1e6;

// One real SD-Turbo attention call that we test. testdata/make_attention.py writes its files to testdata/<name>/.
pub struct AttentionCase {
    // The folder name, for example "attention_self".
    pub name: &'static str,
    // The layer in the SD-Turbo UNet that has this shape.
    pub layer: &'static str,
    // The sizes. They must match testdata/make_attention.py.
    pub shape: AttentionShape,
}

// The two attention kinds of the first UNet block at a 512 x 512 image: 64 x 64 = 4096 tokens, 320 channels = 5 heads of 64.
pub const ATTENTION_CASES: [AttentionCase; 2] = [
    // Self attention: the image tokens look at each other.
    AttentionCase {
        name: "attention_self",
        layer: "down_blocks.0.attentions.0.transformer_blocks.0.attn1",
        shape: AttentionShape { query_tokens: 4096, key_tokens: 4096, heads: 5, head_size: 64 },
    },
    // Cross attention: the image tokens look at the 77 tokens of the text prompt.
    AttentionCase {
        name: "attention_cross",
        layer: "down_blocks.0.attentions.0.transformer_blocks.0.attn2",
        shape: AttentionShape { query_tokens: 4096, key_tokens: 77, heads: 5, head_size: 64 },
    },
];

// Find a case by its name.
pub fn find_attention_case(name: &str) -> Result<&'static AttentionCase, EngineError> {
    // Look through the list.
    ATTENTION_CASES.iter().find(|case| case.name == name).ok_or_else(|| EngineError::UnknownCase(name.to_string()))
}

// The result of one case: how wrong, how fast, and how much GPU memory.
pub struct AttentionReport {
    // Query rows per chunk.
    chunk_rows: u32,
    // The largest difference from the PyTorch output.
    max_error: f32,
    // The most bytes that our GPU buffers held at one time during the case.
    peak_bytes: u64,
    // Bytes that the full table of scores would need. We print it to compare.
    full_scores_bytes: u64,
    // The speed in G ops/s.
    speed: BenchReport,
}

impl AttentionReport {
    // The largest difference from the PyTorch output.
    pub fn max_error(&self) -> f32 {
        self.max_error
    }

    // The most bytes that our GPU buffers held at one time.
    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    // Bytes that the full table of scores would need.
    pub fn full_scores_bytes(&self) -> u64 {
        self.full_scores_bytes
    }
}

// One line: the speed, then the chunk, the error and the peak memory.
impl fmt::Display for AttentionReport {
    // Write the line.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "{}, chunk {} rows, largest error {:e}, peak {} bytes allocated ({:.1} MB, the full scores table alone would be {:.1} MB)",
            self.speed,
            self.chunk_rows,
            self.max_error,
            self.peak_bytes,
            self.peak_bytes as f64 / MEGABYTE,
            self.full_scores_bytes as f64 / MEGABYTE
        )
    }
}

// Run one case with chunks of `chunk_rows` query rows: compare with PyTorch once, then time it.
pub async fn check_attention(gpu: &Gpu, case: &AttentionCase, chunk_rows: u32, tensors: &AttentionTensors<'_>, expected: &[f32]) -> Result<AttentionReport, EngineError> {
    // Measure the peak from here. Buffers of earlier cases are already freed.
    gpu.reset_peak_bytes();
    // Check the sizes, copy the data to the GPU and connect one dispatch per chunk.
    let attention = prepare_attention(gpu, &case.shape, chunk_rows, tensors)?;
    // Run once and copy the output back. The read-back buffer counts toward the peak too.
    let output = attention.run(gpu).await?;
    // Compare with the PyTorch output.
    let max_error = max_abs_error(&output, expected)?;
    // Time it: one skipped submit, calibration, then the timed submits. One run is every chunk.
    let timing = measure(gpu, &Kernel { name: case.name, dispatches: attention.dispatches() }).await?;
    // Operations per run divided by the time, in G ops/s.
    let speed = BenchReport::new(case.name, "G ops/s", case.shape.operations(), &timing);
    // All results. The chunk is the one we used, which is never larger than the queries.
    Ok(AttentionReport {
        chunk_rows: chunk_rows.min(case.shape.query_tokens),
        max_error,
        peak_bytes: gpu.peak_bytes(),
        full_scores_bytes: case.shape.full_scores_bytes(),
        speed,
    })
}
