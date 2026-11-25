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
    // The position of row 0.
    pub first_position: u32,
}

// Bind the rotation of `values` in place, with the cos and sin tables [positions][angles].
pub fn bind_rope(gpu: &Gpu, shape: &RopeShape, values: &wgpu::Buffer, cos: &wgpu::Buffer, sin: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // The field order of the Shape struct in rope.wgsl.
    let sizes = gpu.uniform_buffer("rope shape", &[shape.rows, shape.heads, shape.head_dim, shape.angles, shape.first_position]);
    // Each pair reads and writes two values, and reads one cos and one sin.
    let pairs = shape.rows * shape.heads * shape.angles;
    let work = Work::f32_values("rope", 6 * u64::from(pairs));
    // One invocation per pair.
    let dispatch = Dispatch::new(gpu, "rope", SHADER, &[&sizes, values, cos, sin], [pairs.div_ceil(WORKGROUP_SIZE), 1, 1], work)?;
    Ok(BoundKernel::new(vec![dispatch], vec![sizes]))
}
