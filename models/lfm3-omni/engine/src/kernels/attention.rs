// Our GPU connection, a kernel with its buffers, and our error type.
use engine_core::dispatch::{BoundKernel, Dispatch, Work};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("attention.wgsl");
// Values per head. attention.wgsl has the same number in HEAD_DIM.
pub const HEAD_DIM: u32 = 128;

// The sizes of one attention call. The position of the first row and the key count come from the step buffer.
pub struct AttentionShape {
    // Query tokens in this run.
    pub rows: u32,
    // The most positions that the memory holds. It sets the work count for the profiler.
    pub capacity: u32,
    // Query heads, and key and value heads.
    pub heads: u32,
    pub kv_heads: u32,
    // 0: no window. Otherwise the keys that a query sees, its own included.
    pub window: u32,
}

// The buffers of one attention call: query, the memory of keys and values, gate, and the output.
pub struct AttentionBuffers<'a> {
    pub step: &'a wgpu::Buffer,
    pub query: &'a wgpu::Buffer,
    pub keys: &'a wgpu::Buffer,
    pub values: &'a wgpu::Buffer,
    pub gate: &'a wgpu::Buffer,
    pub output: &'a wgpu::Buffer,
}

// Bind causal attention with grouped keys and a gate per head.
pub fn bind_attention(gpu: &Gpu, shape: &AttentionShape, buffers: &AttentionBuffers) -> Result<BoundKernel, EngineError> {
    // Each key head serves the same number of query heads.
    if shape.kv_heads == 0 || shape.heads % shape.kv_heads != 0 {
        return Err(EngineError::BadDimension { dimension: "kv_heads", value: shape.kv_heads });
    }
    // The scale of the scores: 1 / sqrt(head_dim), as self.scaling.
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    // The field order of the Shape struct in attention.wgsl.
    let sizes = gpu.uniform_buffer("attention shape", &[shape.rows, shape.heads, shape.kv_heads, shape.window, scale.to_bits()]);
    // At most: the queries and the output once, every key and value of the memory once.
    let work = Work::f32_values("attention", u64::from(HEAD_DIM) * (2 * u64::from(shape.rows * shape.heads) + 2 * u64::from(shape.capacity * shape.kv_heads)));
    let bound = [&*sizes, buffers.step, buffers.query, buffers.keys, buffers.values, buffers.gate, buffers.output];
    // One workgroup per query row and head.
    let dispatch = Dispatch::new(gpu, "attention", SHADER, &bound, [shape.rows, shape.heads, 1], work)?;
    Ok(BoundKernel::new(vec![dispatch], vec![sizes]))
}
