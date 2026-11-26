// Our GPU connection, a kernel with its buffers, and our error type.
use engine_core::dispatch::{BoundKernel, Dispatch, Work};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;

// The shared workgroup size.
use super::WORKGROUP_SIZE;

// The WGSL sources: h = B * x into the history, then the convolution over the history.
const GATE_SHADER: &str = include_str!("conv_gate.wgsl");
const CONV_SHADER: &str = include_str!("short_conv.wgsl");

// The sizes of the gated convolution: `rows` new tokens, `width` channels, `kernel` taps.
pub struct ConvShape {
    pub rows: u32,
    pub width: u32,
    pub kernel: u32,
}

// The buffers of the gated convolution.
pub struct ConvBuffers<'a> {
    // Where this run starts.
    pub step: &'a wgpu::Buffer,
    // The output of in_proj for the new tokens: [rows][B | C | x].
    pub bcx: &'a wgpu::Buffer,
    // h of every position so far: [capacity][width].
    pub history: &'a wgpu::Buffer,
    // The convolution weights: [width][kernel].
    pub taps: &'a wgpu::Buffer,
    // The output: [rows][width].
    pub output: &'a wgpu::Buffer,
}

// Bind the two kernels of the gated convolution, in run order.
pub fn bind_short_conv(gpu: &Gpu, shape: &ConvShape, buffers: &ConvBuffers) -> Result<BoundKernel, EngineError> {
    // The field order of the Shape structs in conv_gate.wgsl and short_conv.wgsl.
    let gate_sizes = gpu.uniform_buffer("conv_gate shape", &[shape.rows, shape.width]);
    let conv_sizes = gpu.uniform_buffer("short_conv shape", &[shape.rows, shape.width, shape.kernel]);
    // One invocation per new token and channel.
    let values = shape.rows * shape.width;
    let workgroups = [values.div_ceil(WORKGROUP_SIZE), 1, 1];
    // The gate reads B and x and writes h. The convolution reads 7 positions of h and C, and writes y.
    let gate_work = Work::f32_values("conv_gate", 3 * u64::from(values));
    let conv_work = Work::f32_values("short_conv", u64::from(values) * (u64::from(shape.kernel) + 2) + u64::from(shape.width * shape.kernel));
    let gate = Dispatch::new(gpu, "conv_gate", GATE_SHADER, &[&gate_sizes, buffers.step, buffers.bcx, buffers.history], workgroups, gate_work)?;
    let conv = Dispatch::new(gpu, "short_conv", CONV_SHADER, &[&conv_sizes, buffers.step, buffers.bcx, buffers.history, buffers.taps, buffers.output], workgroups, conv_work)?;
    Ok(BoundKernel::new(vec![gate, conv], vec![gate_sizes, conv_sizes]))
}
