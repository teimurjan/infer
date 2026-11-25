// One short convolution layer of the LFM3 language model (Lfm3DecoderLayer with Lfm3ShortConv), with sandwich norms:
// hidden += post_operator_norm(out_proj(conv(in_proj(operator_norm(hidden)))))
// hidden += post_ffn_norm(w2(silu(w1(n)) * w3(n))), with n = ffn_norm(hidden).

// Our GPU connection, kernels with their buffers, our error type, buffers that count their bytes, and the weights.
use engine_core::dispatch::{BoundKernel, Dispatch};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::memory::GpuBuffer;
use engine_core::weights::{GpuWeights, WeightBuffers};

// The kernels of this crate.
use crate::kernels::{NormMode, bind_rms_norm, bind_short_conv, bind_swiglu};
use crate::linear::bind_linear;

// A short convolution layer, bound to its weights and to the hidden rows that it changes in place.
pub struct ConvLayer {
    // The 11 kernels, in run order.
    kernels: Vec<BoundKernel>,
    // The buffers between the kernels. We keep them as long as the kernels.
    _buffers: Vec<GpuBuffer>,
}

impl ConvLayer {
    // Bind layer `prefix` (for example "layers.0.") for `rows` tokens. `hidden` holds [rows][width] and is changed in place.
    pub fn bind(gpu: &Gpu, weights: &GpuWeights, prefix: &str, rows: u32, hidden: &wgpu::Buffer) -> Result<Self, EngineError> {
        // The weight name of one part of this layer.
        let name = |part: &str| format!("{prefix}{part}.weight");
        // The sizes come from the weights: width from the first norm, the feed-forward width from w1, the taps from conv.
        let width = weights.float(gpu, &name("operator_norm"))?.shape[0] as u32;
        let feed_forward = weights.matrix(gpu, &name("feed_forward.w1"))?.shape[0] as u32;
        let kernel = weights.float(gpu, &name("conv.conv"))?.shape[2] as u32;
        // The buffers between the kernels, each [rows][columns].
        let buffer = |label: &str, columns: u32| gpu.empty_storage_buffer(label, (rows * columns) as usize);
        let (normed, bcx, mixed, branch) = (buffer("normed", width), buffer("bcx", 3 * width), buffer("mixed", width), buffer("branch", width));
        let (gate, up) = (buffer("gate", feed_forward), buffer("up", feed_forward));
        // A norm weight of this layer, as the buffer that the kernel binds.
        let norm = |part: &str| weights.float(gpu, &name(part));
        let kernels = vec![
            // The convolution half.
            bind_rms_norm(gpu, rows, width, hidden, norm("operator_norm")?.buffer.buffer(), &normed, NormMode::Write)?,
            bind_linear(gpu, weights, &name("conv.in_proj"), rows, &normed, &bcx)?,
            bind_short_conv(gpu, rows, width, kernel, &bcx, norm("conv.conv")?.buffer.buffer(), &mixed)?,
            bind_linear(gpu, weights, &name("conv.out_proj"), rows, &mixed, &branch)?,
            bind_rms_norm(gpu, rows, width, &branch, norm("post_operator_norm")?.buffer.buffer(), hidden, NormMode::Add)?,
            // The feed-forward half.
            bind_rms_norm(gpu, rows, width, hidden, norm("ffn_norm")?.buffer.buffer(), &normed, NormMode::Write)?,
            bind_linear(gpu, weights, &name("feed_forward.w1"), rows, &normed, &gate)?,
            bind_linear(gpu, weights, &name("feed_forward.w3"), rows, &normed, &up)?,
            bind_swiglu(gpu, rows * feed_forward, &gate, &up)?,
            bind_linear(gpu, weights, &name("feed_forward.w2"), rows, &gate, &branch)?,
            bind_rms_norm(gpu, rows, width, &branch, norm("post_ffn_norm")?.buffer.buffer(), hidden, NormMode::Add)?,
        ];
        // Ready to run.
        Ok(Self { kernels, _buffers: vec![normed, bcx, mixed, branch, gate, up] })
    }

    // Every dispatch of the layer, in run order, for one submit.
    pub fn dispatches(&self) -> Vec<Dispatch> {
        self.kernels.iter().flat_map(|kernel| kernel.dispatches().iter().cloned()).collect()
    }
}
