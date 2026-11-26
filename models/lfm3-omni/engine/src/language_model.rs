// The language model of LFM3-Omni-2B for one run of `rows` new tokens: the 24 layers, the final norm, and the
// scores of the token after the last new row. The memory of earlier tokens holds what the layers need from earlier runs.

// Our GPU connection, kernels, our error type, buffers, and the weights.
use engine_core::bytes::f32_from_le_bytes;
use engine_core::dispatch::{BoundKernel, Dispatch};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::memory::GpuBuffer;
use engine_core::weights::{GpuWeights, WeightSource};

// The layers, their kinds, the copy kernel, the memory and the output layer.
use crate::attention_layer::AttentionLayer;
use crate::config::{LAYER_KINDS, LayerKind};
use crate::conv_layer::ConvLayer;
use crate::kernels::{Target, bind_copy_rows};
use crate::memory::Memory;
use crate::output_layer::{OutputLayer, TOKEN_TABLE, read_scores};

// One layer of either kind.
enum Layer {
    Conv(ConvLayer),
    Attention(AttentionLayer),
}

// The language model, bound for runs of `rows` new tokens.
pub struct LanguageModel {
    // Every dispatch in run order.
    dispatches: Vec<Dispatch>,
    // The layers and kernels that own the bind groups and buffers of the dispatches.
    _layers: Vec<Layer>,
    _kernels: Vec<BoundKernel>,
    // The final hidden rows [rows][width], the last of them alone, and the scores of each chunk of the output layer.
    last_hidden: GpuBuffer,
    _last_row: GpuBuffer,
    scores: Vec<(GpuBuffer, u32)>,
}

impl LanguageModel {
    // Bind the 24 layers from `layers`, the memory and the output layer for runs of `rows` new tokens.
    // `hidden` holds the token rows of each run and is changed in place.
    pub fn bind(gpu: &Gpu, layers: &GpuWeights, output: &OutputLayer, memory: &Memory, rows: u32, hidden: &wgpu::Buffer) -> Result<Self, EngineError> {
        let bound = LAYER_KINDS.iter().enumerate().map(|(index, kind)| Ok(match kind {
            LayerKind::Conv => Layer::Conv(ConvLayer::bind(gpu, layers, index, rows, hidden, memory)?),
            LayerKind::Full | LayerKind::Sliding => Layer::Attention(AttentionLayer::bind(gpu, layers, index, rows, hidden, memory)?),
        })).collect::<Result<Vec<_>, EngineError>>()?;
        // The final norm of every row, then the last row alone, then the scores of the next token from it.
        let width = output.width();
        let last_hidden = gpu.empty_storage_buffer("last hidden", (rows * width) as usize);
        let last_row = gpu.empty_storage_buffer("last row", width as usize);
        let mut kernels = vec![
            output.bind_norm(gpu, rows, hidden, &last_hidden)?,
            bind_copy_rows(gpu, [1, width, rows - 1], Target::Row(0), memory.step(), &last_hidden, &last_row)?,
        ];
        let scores = output.bind_scores(gpu, 1, &last_row)?;
        kernels.extend(scores.kernels);
        // Every dispatch in run order.
        let layer_dispatches = bound.iter().flat_map(|layer| match layer {
            Layer::Conv(layer) => layer.dispatches(),
            Layer::Attention(layer) => layer.dispatches(),
        });
        let dispatches = layer_dispatches.chain(kernels.iter().flat_map(|kernel| kernel.dispatches().iter().cloned())).collect();
        Ok(Self { dispatches, _layers: bound, _kernels: kernels, last_hidden, _last_row: last_row, scores: scores.scores })
    }

    // Every dispatch, in run order, for one submit.
    pub fn dispatches(&self) -> &[Dispatch] {
        &self.dispatches
    }

    // The final hidden rows [rows][width], after the final norm.
    pub fn last_hidden(&self) -> &wgpu::Buffer {
        &self.last_hidden
    }

    // The scores of every token of the vocabulary for the token after the last row of the run.
    pub async fn scores(&self, gpu: &Gpu) -> Result<Vec<f32>, EngineError> {
        read_scores(gpu, &self.scores, 0).await
    }
}

// The token rows of `ids` from the token table, on the CPU: [ids.len()][width].
pub fn token_rows(table: &impl WeightSource, ids: &[u32]) -> Result<Vec<f32>, EngineError> {
    let rows = ids.iter().map(|&id| table.read_row(TOKEN_TABLE, id as usize).map(|bytes| f32_from_le_bytes(&bytes)));
    Ok(rows.collect::<Result<Vec<_>, EngineError>>()?.concat())
}
