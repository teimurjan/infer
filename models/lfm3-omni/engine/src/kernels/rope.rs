// Our GPU connection, a kernel with its buffers, and our error type.
use engine_core::dispatch::{BoundKernel, Dispatch, Work};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;

// The shared workgroup size.
use super::WORKGROUP_SIZE;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("rope.wgsl");

// The sizes of one rotation: `rows` tokens of `heads` heads of `head_dim` values, the first `angles` pairs rotated.
pub struct RopeShape {
    pub rows: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub angles: u32,
}

// The tables of one kind of rotation: cos and sin, each [positions][angles].
pub struct RopeTables<'a> {
    pub cos: &'a wgpu::Buffer,
    pub sin: &'a wgpu::Buffer,
}

// Bind the rotation of `values` in place. Row r is at position start + r, from the step buffer.
pub fn bind_rope(gpu: &Gpu, shape: &RopeShape, step: &wgpu::Buffer, values: &wgpu::Buffer, tables: &RopeTables) -> Result<BoundKernel, EngineError> {
    // The field order of the Shape struct in rope.wgsl.
    let sizes = gpu.uniform_buffer("rope shape", &[shape.rows, shape.heads, shape.head_dim, shape.angles]);
    // Each pair reads and writes two values, and reads one cos and one sin.
    let pairs = shape.rows * shape.heads * shape.angles;
    let work = Work::f32_values("rope", 6 * u64::from(pairs));
    // One invocation per pair.
    let dispatch = Dispatch::new(gpu, "rope", SHADER, &[&sizes, step, values, tables.cos, tables.sin], [pairs.div_ceil(WORKGROUP_SIZE), 1, 1], work)?;
    Ok(BoundKernel::new(vec![dispatch], vec![sizes]))
}
