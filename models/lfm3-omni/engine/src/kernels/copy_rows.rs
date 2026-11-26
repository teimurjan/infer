// Our GPU connection, a kernel with its buffers, and our error type.
use engine_core::dispatch::{BoundKernel, Dispatch, Work};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;

// The shared workgroup size.
use super::WORKGROUP_SIZE;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("copy_rows.wgsl");

// Where the copied rows go: a fixed row, or the rows of the new tokens in the memory of earlier tokens.
#[derive(Clone, Copy)]
pub enum Target {
    // Target row `first` on.
    Row(u32),
    // Target row `start` on, from the step buffer.
    AtStart,
}

// Bind a copy of `rows` rows of `width` values, from source row `source_first` on.
pub fn bind_copy_rows(gpu: &Gpu, [rows, width, source_first]: [u32; 3], target: Target, step: &wgpu::Buffer, source: &wgpu::Buffer, destination: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    let (target_first, after_start) = match target {
        Target::Row(first) => (first, 0),
        Target::AtStart => (0, 1),
    };
    // The field order of the Shape struct in copy_rows.wgsl.
    let sizes = gpu.uniform_buffer("copy_rows shape", &[rows, width, source_first, target_first, after_start]);
    let values = rows * width;
    let dispatch = Dispatch::new(gpu, "copy_rows", SHADER, &[&sizes, step, source, destination], [values.div_ceil(WORKGROUP_SIZE), 1, 1], Work::f32_values("copy_rows", 2 * u64::from(values)))?;
    Ok(BoundKernel::new(vec![dispatch], vec![sizes]))
}
