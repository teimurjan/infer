// Our GPU connection, a kernel with its buffers, the kernel compiler, and our error type.
use engine_core::dispatch::{BoundKernel, Dispatch, Work};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("rms_norm.wgsl");
// The eps of every RMSNorm of LFM3 (norm_eps in config.json).
const EPS: f32 = 1e-5;

// Whether the norm writes a new tensor or adds to the residual stream.
#[derive(Clone, Copy)]
pub enum NormMode {
    // output = norm(input).
    Write,
    // output += norm(input).
    Add,
}

// Bind RMSNorm of `rows` rows of `width` values: `input` and `weight` to `output`.
pub fn bind_rms_norm(gpu: &Gpu, rows: u32, width: u32, input: &wgpu::Buffer, weight: &wgpu::Buffer, output: &wgpu::Buffer, mode: NormMode) -> Result<BoundKernel, EngineError> {
    // The field order of the Shape struct in rms_norm.wgsl.
    let accumulate = match mode {
        NormMode::Write => 0,
        NormMode::Add => 1,
    };
    let shape = gpu.uniform_buffer("rms_norm shape", &[rows, width, accumulate, EPS.to_bits()]);
    // It reads the input and the weight and writes the output. An add also reads the output, but the count stays simple.
    let work = Work::f32_values("rms_norm", 2 * u64::from(rows) * u64::from(width) + u64::from(width));
    // One workgroup per row.
    let dispatch = Dispatch::new(gpu, "rms_norm", SHADER, &[&shape, input, weight, output], [rows, 1, 1], work)?;
    // Keep the sizes buffer with the dispatch.
    Ok(BoundKernel::new(vec![dispatch], vec![shape]))
}
