// Our GPU connection, a kernel with its buffers, and our error type.
use engine_core::dispatch::{BoundKernel, Dispatch, Work};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;

// The shared workgroup size.
use super::WORKGROUP_SIZE;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("short_conv.wgsl");

// Bind the gated convolution: `bcx` [rows][3 x width] and `taps` [width][kernel] to `output` [rows][width].
pub fn bind_short_conv(gpu: &Gpu, rows: u32, width: u32, kernel: u32, bcx: &wgpu::Buffer, taps: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // The field order of the Shape struct in short_conv.wgsl.
    let shape = gpu.uniform_buffer("short_conv shape", &[rows, width, kernel]);
    // It reads B, C and x and writes one value per token and channel.
    let work = Work::f32_values("short_conv", 4 * u64::from(rows) * u64::from(width) + u64::from(width) * u64::from(kernel));
    // One invocation per output value.
    let workgroups = [(rows * width).div_ceil(WORKGROUP_SIZE), 1, 1];
    let dispatch = Dispatch::new(gpu, "short_conv", SHADER, &[&shape, bcx, taps, output], workgroups, work)?;
    // Keep the sizes buffer with the dispatch.
    Ok(BoundKernel::new(vec![dispatch], vec![shape]))
}
