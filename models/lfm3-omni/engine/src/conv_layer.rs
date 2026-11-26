// One short convolution layer of the LFM3 language model (Lfm3DecoderLayer with Lfm3ShortConv), with sandwich norms:
// hidden += post_operator_norm(out_proj(conv(in_proj(operator_norm(hidden))))), then the feed-forward half.

// Our GPU connection, kernels with their buffers, our error type, buffers that count their bytes, and the weights.
use engine_core::dispatch::{BoundKernel, Dispatch};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::memory::GpuBuffer;
use engine_core::weights::{GpuWeights, WeightBuffers};

// The kernels of this crate, the feed-forward half that every layer shares, and the memory of earlier tokens.
use crate::feed_forward::bind_feed_forward;
use crate::kernels::{ConvBuffers, ConvShape, NormMode, bind_rms_norm, bind_short_conv};
use crate::linear::bind_linear;
use crate::memory::{LayerMemory, Memory};

// A short convolution layer, bound to its weights, its memory, and the hidden rows that it changes in place.
pub struct ConvLayer {
    // The kernels, in run order.
    kernels: Vec<BoundKernel>,
    // The buffers between the kernels. We keep them as long as the kernels.
    _buffers: Vec<GpuBuffer>,
}

impl ConvLayer {
    // Bind layer `index` for runs of `rows` new tokens. `hidden` holds [rows][width] and is changed in place.
    pub fn bind(gpu: &Gpu, weights: &GpuWeights, index: usize, rows: u32, hidden: &wgpu::Buffer, memory: &Memory) -> Result<Self, EngineError> {
        // The weight name of one part of this layer.
        let prefix = format!("layers.{index}.");
        let name = |part: &str| format!("{prefix}{part}.weight");
        let LayerMemory::Conv { history } = memory.layer(index) else {
            return Err(EngineError::WeightLayout(format!("layer {index} has no convolution memory")));
        };
        // The width from the first norm, the convolution width from the taps [width, 1, kernel].
        let width = weights.float(gpu, &name("operator_norm"))?.shape[0] as u32;
        let kernel = weights.float(gpu, &name("conv.conv"))?.shape[2] as u32;
        // The buffers between the kernels, each [rows][columns].
        let buffer = |label: &str, columns: u32| gpu.empty_storage_buffer(label, (rows * columns) as usize);
        let (normed, bcx, mixed, branch) = (buffer("normed", width), buffer("bcx", 3 * width), buffer("mixed", width), buffer("branch", width));
        // A weight of this layer, as the buffer that the kernel binds.
        let float = |part: &str| weights.float(gpu, &name(part));
        let taps = float("conv.conv")?;
        let conv = ConvBuffers { step: memory.step(), bcx: &bcx, history, taps: taps.buffer.buffer(), output: &mixed };
        // The convolution half.
        let mut kernels = vec![
            bind_rms_norm(gpu, rows, width, hidden, float("operator_norm")?.buffer.buffer(), &normed, NormMode::Write)?,
            bind_linear(gpu, weights, &name("conv.in_proj"), rows, &normed, &bcx)?,
            bind_short_conv(gpu, &ConvShape { rows, width, kernel }, &conv)?,
            bind_linear(gpu, weights, &name("conv.out_proj"), rows, &mixed, &branch)?,
            bind_rms_norm(gpu, rows, width, &branch, float("post_operator_norm")?.buffer.buffer(), hidden, NormMode::Add)?,
        ];
        // The feed-forward half, which reuses `normed` and `branch`.
        let (feed_forward, wide) = bind_feed_forward(gpu, weights, &prefix, rows, hidden, &normed, &branch)?;
        kernels.extend(feed_forward);
        Ok(Self { kernels, _buffers: [vec![normed, bcx, mixed, branch], wide].into_iter().flatten().collect() })
    }

    // Every dispatch of the layer, in run order.
    pub fn dispatches(&self) -> Vec<Dispatch> {
        self.kernels.iter().flat_map(|kernel| kernel.dispatches().iter().cloned()).collect()
    }
}
