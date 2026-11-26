// The end of the language model: the final RMSNorm (embedding_norm) and the output layer, which shares its weights
// with the token table (tie_word_embeddings). The weights go to the GPU once, and every run binds to them.

// Our GPU connection, kernels, our error type, buffers, and the weights.
use engine_core::dispatch::BoundKernel;
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::memory::{GpuBuffer, MemoryCategory};
use engine_core::weights::{Matrix, WeightBuffer, WeightSource};

// The norm kernel and the linear layer.
use crate::kernels::{NormMode, bind_rms_norm};
use crate::linear::bind_linear_matrix;

// The token table, which is also the output layer.
pub const TOKEN_TABLE: &str = "embed_tokens.weight";
// The final RMSNorm.
const FINAL_NORM: &str = "embedding_norm.weight";
// Tokens per chunk of the output layer. One float32 chunk is 16000 x 2048 x 4 = 131,072,000 bytes, under the 128 MiB
// (134,217,728 bytes) that one storage binding may hold with the default WebGPU limits.
const CHUNK_TOKENS: usize = 16_000;

// The final norm weight and the output layer in chunks, on the GPU.
pub struct OutputLayer {
    width: u32,
    norm: GpuBuffer,
    // Each chunk of the token table, and its number of tokens.
    chunks: Vec<(GpuBuffer, u32)>,
}

// The output layer bound to one run: its kernels and the score buffer of each chunk.
pub struct BoundOutput {
    pub kernels: Vec<BoundKernel>,
    // The scores of each chunk, [rows][chunk tokens], and the number of tokens of the chunk.
    pub scores: Vec<(GpuBuffer, u32)>,
}

impl OutputLayer {
    // Read the final norm and the token table from `table` (weights/embed.bin) and copy them to the GPU.
    pub fn new(gpu: &Gpu, table: &impl WeightSource) -> Result<Self, EngineError> {
        let norm = table.read(FINAL_NORM)?;
        let width = norm.shape[0] as u32;
        let full = table.read(TOKEN_TABLE)?;
        let chunks = full.bytes.chunks(CHUNK_TOKENS * width as usize * 4).enumerate().map(|(index, chunk)| {
            let tokens = (chunk.len() / (width as usize * 4)) as u32;
            (gpu.storage_buffer_from_bytes(&format!("output layer chunk {index}"), chunk, MemoryCategory::Weights), tokens)
        }).collect();
        Ok(Self { width, norm: gpu.storage_buffer_from_bytes(FINAL_NORM, &norm.bytes, MemoryCategory::Weights), chunks })
    }

    // Values in one row.
    pub fn width(&self) -> u32 {
        self.width
    }

    // The final norm of `rows` rows of `hidden` into `normed`.
    pub fn bind_norm(&self, gpu: &Gpu, rows: u32, hidden: &wgpu::Buffer, normed: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
        bind_rms_norm(gpu, rows, self.width, hidden, &self.norm, normed, NormMode::Write)
    }

    // The scores of every token for `rows` normed rows of `input`.
    pub fn bind_scores(&self, gpu: &Gpu, rows: u32, input: &wgpu::Buffer) -> Result<BoundOutput, EngineError> {
        let mut kernels = Vec::new();
        let mut scores = Vec::new();
        for (index, (weight, tokens)) in self.chunks.iter().enumerate() {
            let output = gpu.empty_storage_buffer(&format!("scores chunk {index}"), (rows * tokens) as usize);
            kernels.push(bind_linear_matrix(gpu, "output layer", Matrix::Float(WeightBuffer::Borrowed(weight)), [*tokens, self.width], rows, input, &output)?);
            scores.push((output, *tokens));
        }
        Ok(BoundOutput { kernels, scores })
    }
}

// The scores of row `row` of a bound output layer, read back from its chunks.
pub async fn read_scores(gpu: &Gpu, scores: &[(GpuBuffer, u32)], row: usize) -> Result<Vec<f32>, EngineError> {
    let mut all = Vec::new();
    for (buffer, tokens) in scores {
        let values = gpu.read_f32(buffer).await?;
        let tokens = *tokens as usize;
        all.extend_from_slice(&values[row * tokens..(row + 1) * tokens]);
    }
    Ok(all)
}
