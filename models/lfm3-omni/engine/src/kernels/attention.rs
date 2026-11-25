// Our GPU connection, a kernel with its buffers, and our error type.
use engine_core::dispatch::{BoundKernel, Dispatch, Work};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("attention.wgsl");
// Values per head. attention.wgsl has the same number in HEAD_DIM.
pub const HEAD_DIM: u32 = 128;

// The sizes of one attention call.
pub struct AttentionShape {
    // Query tokens, and keys that exist.
    pub rows: u32,
    pub keys: u32,
    // Query heads, and key and value heads.
    pub heads: u32,
    pub kv_heads: u32,
    // The position of query row 0.
    pub first_position: u32,
    // 0: no window. Otherwise the keys that a query sees, its own included.
    pub window: u32,
}

// The buffers of one attention call: query, key, value, gate, and the output.
pub struct AttentionBuffers<'a> {
    pub query: &'a wgpu::Buffer,
    pub key: &'a wgpu::Buffer,
    pub value: &'a wgpu::Buffer,
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
    let sizes = gpu.uniform_buffer("attention shape", &[shape.rows, shape.keys, shape.heads, shape.kv_heads, shape.first_position, shape.window, scale.to_bits()]);
    // It reads the queries, keys and values once and writes the output.
    let work = Work::f32_values("attention", u64::from(HEAD_DIM) * (2 * u64::from(shape.rows * shape.heads) + 2 * u64::from(shape.keys * shape.kv_heads)));
    // One workgroup per query row and head.
    let dispatch = Dispatch::new(gpu, "attention", SHADER, &[&sizes, buffers.query, buffers.key, buffers.value, buffers.gate, buffers.output], [shape.rows, shape.heads, 1], work)?;
    Ok(BoundKernel::new(vec![dispatch], vec![sizes]))
}
