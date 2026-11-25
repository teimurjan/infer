// The second half of every LFM3 layer, with its sandwich norm:
// hidden += post_ffn_norm(w2(silu(w1(n)) * w3(n))), with n = ffn_norm(hidden).

// Our GPU connection, kernels with their buffers, our error type, buffers that count their bytes, and the weights.
use engine_core::dispatch::BoundKernel;
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::memory::GpuBuffer;
use engine_core::weights::{GpuWeights, WeightBuffers};

// The kernels of this crate.
use crate::kernels::{NormMode, bind_rms_norm, bind_swiglu};
use crate::linear::bind_linear;

// The feed-forward half of layer `prefix`, bound to `hidden` [rows][width]. `normed` and `branch` are [rows][width]
// buffers that the layer lends: the first half of the layer is done with them when this half runs.
pub fn bind_feed_forward(gpu: &Gpu, weights: &GpuWeights, prefix: &str, rows: u32, hidden: &wgpu::Buffer, normed: &wgpu::Buffer, branch: &wgpu::Buffer) -> Result<(Vec<BoundKernel>, Vec<GpuBuffer>), EngineError> {
    // The weight name of one part of this layer.
    let name = |part: &str| format!("{prefix}{part}.weight");
    // The width from the norm, the feed-forward width from w1.
    let width = weights.float(gpu, &name("ffn_norm"))?.shape[0] as u32;
    let wide = weights.matrix(gpu, &name("feed_forward.w1"))?.shape[0] as u32;
    // The two wide buffers, [rows][wide]. SwiGLU writes over the first one.
    let gate = gpu.empty_storage_buffer("gate", (rows * wide) as usize);
    let up = gpu.empty_storage_buffer("up", (rows * wide) as usize);
    // A norm weight of this layer, as the buffer that the kernel binds.
    let norm = |part: &str| weights.float(gpu, &name(part));
    let kernels = vec![
        bind_rms_norm(gpu, rows, width, hidden, norm("ffn_norm")?.buffer.buffer(), normed, NormMode::Write)?,
        bind_linear(gpu, weights, &name("feed_forward.w1"), rows, normed, &gate)?,
        bind_linear(gpu, weights, &name("feed_forward.w3"), rows, normed, &up)?,
        bind_swiglu(gpu, rows * wide, &gate, &up)?,
        bind_linear(gpu, weights, &name("feed_forward.w2"), rows, &gate, branch)?,
        bind_rms_norm(gpu, rows, width, branch, norm("post_ffn_norm")?.buffer.buffer(), hidden, NormMode::Add)?,
    ];
    Ok((kernels, vec![gate, up]))
}
