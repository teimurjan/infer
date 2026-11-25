// One attention layer of the LFM3 language model (Lfm3DecoderLayer with Lfm3Attention), with sandwich norms:
// hidden += post_operator_norm(o_proj(attention(q, k, v) * sigmoid(gate_proj(n)))), with n = operator_norm(hidden),
// then the feed-forward half. q and k get an RMSNorm per head and rotary positions.

// Our GPU connection, kernels with their buffers, our error type, buffers that count their bytes, and the weights.
use engine_core::dispatch::{BoundKernel, Dispatch};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::memory::{GpuBuffer, MemoryCategory};
use engine_core::weights::{GpuWeights, WeightBuffers};

// The kernels of this crate, the feed-forward half, and the rotary table.
use crate::feed_forward::bind_feed_forward;
use crate::kernels::{AttentionBuffers, AttentionShape, HEAD_DIM, NormMode, RopeShape, bind_attention, bind_rms_norm, bind_rope};
use crate::linear::bind_linear;
use crate::rope_table::rope_table;

// The two kinds of attention layer, with the rotary settings of config.json (rope_parameters, sliding_window).
#[derive(Clone, Copy)]
pub enum AttentionKind {
    // Every earlier key. Theta 1,000,000, and only the first half of the pairs rotate (partial_rotary_factor 0.5).
    Full,
    // The last 256 keys. Theta 10,000, and every pair rotates.
    Sliding,
}

impl AttentionKind {
    // (theta, rotated pairs out of head_dim / 2, window). A window of 0 means no window.
    fn settings(self) -> (f32, u32, u32) {
        match self {
            Self::Full => (1_000_000.0, HEAD_DIM / 4, 0),
            Self::Sliding => (10_000.0, HEAD_DIM / 2, 256),
        }
    }
}

// An attention layer, bound to its weights and to the hidden rows that it changes in place.
pub struct AttentionLayer {
    // The kernels, in run order.
    kernels: Vec<BoundKernel>,
    // The buffers between the kernels and the rotary tables. We keep them as long as the kernels.
    _buffers: Vec<GpuBuffer>,
}

impl AttentionLayer {
    // Bind layer `prefix` (for example "layers.2.") for a prompt of `rows` tokens at positions 0 to rows - 1.
    pub fn bind(gpu: &Gpu, weights: &GpuWeights, prefix: &str, kind: AttentionKind, rows: u32, hidden: &wgpu::Buffer) -> Result<Self, EngineError> {
        // The weight name of one part of this layer.
        let name = |part: &str| format!("{prefix}{part}.weight");
        // The width from the first norm. The head counts from the projections: [heads x 128, width].
        let width = weights.float(gpu, &name("operator_norm"))?.shape[0] as u32;
        let heads = weights.matrix(gpu, &name("self_attn.q_proj"))?.shape[0] as u32 / HEAD_DIM;
        let kv_heads = weights.matrix(gpu, &name("self_attn.k_proj"))?.shape[0] as u32 / HEAD_DIM;
        // The buffers between the kernels, each [rows][columns].
        let buffer = |label: &str, columns: u32| gpu.empty_storage_buffer(label, (rows * columns) as usize);
        let (normed, branch, mixed) = (buffer("normed", width), buffer("branch", width), buffer("mixed", heads * HEAD_DIM));
        let (query_raw, query) = (buffer("query raw", heads * HEAD_DIM), buffer("query", heads * HEAD_DIM));
        let (key_raw, key, value) = (buffer("key raw", kv_heads * HEAD_DIM), buffer("key", kv_heads * HEAD_DIM), buffer("value", kv_heads * HEAD_DIM));
        let gate = buffer("gate", heads);
        // The rotary tables of the prompt positions.
        let (theta, angles, window) = kind.settings();
        let table = rope_table(rows, angles, HEAD_DIM, theta);
        let cos = gpu.storage_buffer_from("rope cos", &table.cos, MemoryCategory::Weights);
        let sin = gpu.storage_buffer_from("rope sin", &table.sin, MemoryCategory::Weights);
        // A weight of this layer, as the buffer that the kernel binds.
        let float = |part: &str| weights.float(gpu, &name(part));
        // Rotate `heads` heads of each row, from position 0.
        let rope = |heads: u32| RopeShape { rows, heads, head_dim: HEAD_DIM, angles, first_position: 0 };
        let attention = AttentionShape { rows, keys: rows, heads, kv_heads, first_position: 0, window };
        let mut kernels = vec![
            bind_rms_norm(gpu, rows, width, hidden, float("operator_norm")?.buffer.buffer(), &normed, NormMode::Write)?,
            bind_linear(gpu, weights, &name("self_attn.q_proj"), rows, &normed, &query_raw)?,
            bind_linear(gpu, weights, &name("self_attn.k_proj"), rows, &normed, &key_raw)?,
            bind_linear(gpu, weights, &name("self_attn.v_proj"), rows, &normed, &value)?,
            bind_linear(gpu, weights, &name("self_attn.gate_proj"), rows, &normed, &gate)?,
            // The RMSNorm of each head of 128 values: one row per token and head.
            bind_rms_norm(gpu, rows * heads, HEAD_DIM, &query_raw, float("self_attn.q_norm")?.buffer.buffer(), &query, NormMode::Write)?,
            bind_rms_norm(gpu, rows * kv_heads, HEAD_DIM, &key_raw, float("self_attn.k_norm")?.buffer.buffer(), &key, NormMode::Write)?,
            bind_rope(gpu, &rope(heads), &query, &cos, &sin)?,
            bind_rope(gpu, &rope(kv_heads), &key, &cos, &sin)?,
            bind_attention(gpu, &attention, &AttentionBuffers { query: &query, key: &key, value: &value, gate: &gate, output: &mixed })?,
            bind_linear(gpu, weights, &name("self_attn.o_proj"), rows, &mixed, &branch)?,
            bind_rms_norm(gpu, rows, width, &branch, float("post_operator_norm")?.buffer.buffer(), hidden, NormMode::Add)?,
        ];
        // The feed-forward half, which reuses `normed` and `branch`.
        let (feed_forward, wide) = bind_feed_forward(gpu, weights, prefix, rows, hidden, &normed, &branch)?;
        kernels.extend(feed_forward);
        let buffers = vec![normed, branch, mixed, query_raw, query, key_raw, key, value, gate, cos, sin];
        Ok(Self { kernels, _buffers: [buffers, wide].into_iter().flatten().collect() })
    }

    // Every dispatch of the layer, in run order, for one submit.
    pub fn dispatches(&self) -> Vec<Dispatch> {
        self.kernels.iter().flat_map(|kernel| kernel.dispatches().iter().cloned()).collect()
    }
}
