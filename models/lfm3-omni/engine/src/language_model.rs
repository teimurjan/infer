// The language model of LFM3-Omni-2B over a prompt: token rows, 24 layers, the final norm (embedding_norm), and the
// scores of the next token through the token table (the output layer shares its weights with it).

// Our GPU connection, kernels, our error type, buffers, and the weights.
use engine_core::bytes::f32_from_le_bytes;
use engine_core::dispatch::{BoundKernel, Dispatch};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::memory::{GpuBuffer, MemoryCategory};
use engine_core::weights::{GpuWeights, Matrix, WeightBuffer, WeightSource};

// The layers, their kinds, the norm kernel and the linear layer.
use crate::attention_layer::{AttentionKind, AttentionLayer};
use crate::config::{LAYER_KINDS, LayerKind};
use crate::conv_layer::ConvLayer;
use crate::kernels::{NormMode, bind_rms_norm};
use crate::linear::bind_linear_matrix;

// The token table, which is also the output layer.
const TOKEN_TABLE: &str = "embed_tokens.weight";
// The final RMSNorm.
const FINAL_NORM: &str = "embedding_norm.weight";
// Tokens per chunk of the output layer. One float32 chunk is 16000 x 2048 x 4 = 131,072,000 bytes, under the 128 MiB
// (134,217,728 bytes) that one storage binding may hold with the default WebGPU limits.
const CHUNK_TOKENS: usize = 16_000;

// One layer of either kind.
enum Layer {
    Conv(ConvLayer),
    Attention(AttentionLayer),
}

// The language model, bound for a prompt of `rows` tokens.
pub struct LanguageModel {
    // Every dispatch in run order: the layers, the final norm, and the chunks of the output layer.
    dispatches: Vec<Dispatch>,
    // The layers and kernels that own the bind groups and buffers of the dispatches.
    _layers: Vec<Layer>,
    _kernels: Vec<BoundKernel>,
    // The final hidden rows [rows][width], and the scores of each chunk of the output layer [rows][chunk tokens].
    last_hidden: GpuBuffer,
    scores: Vec<(GpuBuffer, usize)>,
    // The output layer chunks and the final norm weight, kept as long as the kernels.
    _weights: Vec<GpuBuffer>,
}

impl LanguageModel {
    // Bind the 24 layers from `layers` (weights/layers.bin) and the table and final norm from `table`
    // (weights/embed.bin, read on the CPU). `hidden` holds the token rows and is changed in place.
    pub fn bind(gpu: &Gpu, layers: &GpuWeights, table: &impl WeightSource, rows: u32, hidden: &wgpu::Buffer) -> Result<Self, EngineError> {
        // The 24 layers, by kind.
        let bound = LAYER_KINDS.iter().enumerate().map(|(index, kind)| {
            let prefix = format!("layers.{index}.");
            Ok(match kind {
                LayerKind::Conv => Layer::Conv(ConvLayer::bind(gpu, layers, &prefix, rows, hidden)?),
                LayerKind::Full => Layer::Attention(AttentionLayer::bind(gpu, layers, &prefix, AttentionKind::Full, rows, hidden)?),
                LayerKind::Sliding => Layer::Attention(AttentionLayer::bind(gpu, layers, &prefix, AttentionKind::Sliding, rows, hidden)?),
            })
        }).collect::<Result<Vec<_>, EngineError>>()?;
        // The final norm, into its own buffer, so the test can read last_hidden_state.
        let norm = table.read(FINAL_NORM)?;
        let width = norm.shape[0] as u32;
        let norm_weight = gpu.storage_buffer_from_bytes(FINAL_NORM, &norm.bytes, MemoryCategory::Weights);
        let last_hidden = gpu.empty_storage_buffer("last hidden", (rows * width) as usize);
        let mut kernels = vec![bind_rms_norm(gpu, rows, width, hidden, &norm_weight, &last_hidden, NormMode::Write)?];
        // The output layer in chunks of the token table.
        let full = table.read(TOKEN_TABLE)?;
        let row_bytes = width as usize * 4;
        let mut weights = vec![norm_weight];
        let mut scores = Vec::new();
        for (index, chunk) in full.bytes.chunks(CHUNK_TOKENS * row_bytes).enumerate() {
            let tokens = chunk.len() / row_bytes;
            let weight = gpu.storage_buffer_from_bytes(&format!("output layer chunk {index}"), chunk, MemoryCategory::Weights);
            let output = gpu.empty_storage_buffer(&format!("scores chunk {index}"), rows as usize * tokens);
            kernels.push(bind_linear_matrix(gpu, "output layer", Matrix::Float(WeightBuffer::Borrowed(&weight)), [tokens as u32, width], rows, &last_hidden, &output)?);
            weights.push(weight);
            scores.push((output, tokens));
        }
        // Every dispatch in run order.
        let layer_dispatches = bound.iter().flat_map(|layer| match layer {
            Layer::Conv(layer) => layer.dispatches(),
            Layer::Attention(layer) => layer.dispatches(),
        });
        let dispatches = layer_dispatches.chain(kernels.iter().flat_map(|kernel| kernel.dispatches().iter().cloned())).collect();
        Ok(Self { dispatches, _layers: bound, _kernels: kernels, last_hidden, scores, _weights: weights })
    }

    // Every dispatch, in run order, for one submit.
    pub fn dispatches(&self) -> &[Dispatch] {
        &self.dispatches
    }

    // The final hidden rows [rows][width], after the final norm.
    pub fn last_hidden(&self) -> &wgpu::Buffer {
        &self.last_hidden
    }

    // The scores of every token of the vocabulary for row `row`, read back from the chunks.
    pub async fn scores_of_row(&self, gpu: &Gpu, row: usize) -> Result<Vec<f32>, EngineError> {
        let mut all = Vec::new();
        for (buffer, tokens) in &self.scores {
            let values = gpu.read_f32(buffer).await?;
            all.extend_from_slice(&values[row * tokens..(row + 1) * tokens]);
        }
        Ok(all)
    }
}

// The token rows of `ids` from the token table, on the CPU: [ids.len()][width].
pub fn token_rows(table: &impl WeightSource, ids: &[u32]) -> Result<Vec<f32>, EngineError> {
    let rows = ids.iter().map(|&id| table.read_row(TOKEN_TABLE, id as usize).map(|bytes| f32_from_le_bytes(&bytes)));
    Ok(rows.collect::<Result<Vec<_>, EngineError>>()?.concat())
}
