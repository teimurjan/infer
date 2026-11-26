// One attention layer of the LFM3 language model (Lfm3DecoderLayer with Lfm3Attention), with sandwich norms:
// hidden += post_operator_norm(o_proj(attention(q, k, v) * sigmoid(gate_proj(n)))), with n = operator_norm(hidden),
// then the feed-forward half. q and k get an RMSNorm per head and rotary positions. The keys and values of the new
// tokens go into the memory, and the queries attend over every position in it.

// Our GPU connection, kernels with their buffers, our error type, buffers that count their bytes, and the weights.
use engine_core::dispatch::{BoundKernel, Dispatch};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::memory::GpuBuffer;
use engine_core::weights::{GpuWeights, WeightBuffers};

// The kernels of this crate, the feed-forward half, the layer kinds and the memory of earlier tokens.
use crate::config::{LAYER_KINDS, rope_settings};
use crate::feed_forward::bind_feed_forward;
use crate::kernels::{AttentionBuffers, AttentionShape, HEAD_DIM, NormMode, RopeShape, Target, bind_attention, bind_copy_rows, bind_rms_norm, bind_rope};
use crate::linear::bind_linear;
use crate::memory::{LayerMemory, Memory};

// An attention layer, bound to its weights, its memory, and the hidden rows that it changes in place.
pub struct AttentionLayer {
    // The kernels, in run order.
    kernels: Vec<BoundKernel>,
    // The buffers between the kernels. We keep them as long as the kernels.
    _buffers: Vec<GpuBuffer>,
}

impl AttentionLayer {
    // Bind layer `index` for runs of `rows` new tokens. `hidden` holds [rows][width] and is changed in place.
    pub fn bind(gpu: &Gpu, weights: &GpuWeights, index: usize, rows: u32, hidden: &wgpu::Buffer, memory: &Memory) -> Result<Self, EngineError> {
        // The weight name of one part of this layer.
        let prefix = format!("layers.{index}.");
        let name = |part: &str| format!("{prefix}{part}.weight");
        let kind = LAYER_KINDS[index];
        let (Some(settings), LayerMemory::Attention { keys, values }) = (rope_settings(kind), memory.layer(index)) else {
            return Err(EngineError::WeightLayout(format!("layer {index} is not an attention layer")));
        };
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
        // A weight of this layer, as the buffer that the kernel binds.
        let float = |part: &str| weights.float(gpu, &name(part));
        let (step, tables) = (memory.step(), memory.rope(kind));
        // Rotate `heads` heads of each new row.
        let rope = |heads: u32| RopeShape { rows, heads, head_dim: HEAD_DIM, angles: settings.angles };
        // Copy the new rows of keys or values into the memory, at their positions.
        let store = |source: &wgpu::Buffer, target: &wgpu::Buffer| bind_copy_rows(gpu, [rows, kv_heads * HEAD_DIM, 0], Target::AtStart, step, source, target);
        let attention = AttentionShape { rows, capacity: memory.capacity(), heads, kv_heads, window: settings.window };
        let mut kernels = vec![
            bind_rms_norm(gpu, rows, width, hidden, float("operator_norm")?.buffer.buffer(), &normed, NormMode::Write)?,
            bind_linear(gpu, weights, &name("self_attn.q_proj"), rows, &normed, &query_raw)?,
            bind_linear(gpu, weights, &name("self_attn.k_proj"), rows, &normed, &key_raw)?,
            bind_linear(gpu, weights, &name("self_attn.v_proj"), rows, &normed, &value)?,
            bind_linear(gpu, weights, &name("self_attn.gate_proj"), rows, &normed, &gate)?,
            // The RMSNorm of each head of 128 values: one row per token and head.
            bind_rms_norm(gpu, rows * heads, HEAD_DIM, &query_raw, float("self_attn.q_norm")?.buffer.buffer(), &query, NormMode::Write)?,
            bind_rms_norm(gpu, rows * kv_heads, HEAD_DIM, &key_raw, float("self_attn.k_norm")?.buffer.buffer(), &key, NormMode::Write)?,
            bind_rope(gpu, &rope(heads), step, &query, &tables)?,
            bind_rope(gpu, &rope(kv_heads), step, &key, &tables)?,
            store(&key, keys)?,
            store(&value, values)?,
            bind_attention(gpu, &attention, &AttentionBuffers { step, query: &query, keys, values, gate: &gate, output: &mixed })?,
            bind_linear(gpu, weights, &name("self_attn.o_proj"), rows, &mixed, &branch)?,
            bind_rms_norm(gpu, rows, width, &branch, float("post_operator_norm")?.buffer.buffer(), hidden, NormMode::Add)?,
        ];
        // The feed-forward half, which reuses `normed` and `branch`.
        let (feed_forward, wide) = bind_feed_forward(gpu, weights, &prefix, rows, hidden, &normed, &branch)?;
        kernels.extend(feed_forward);
        let buffers = vec![normed, branch, mixed, query_raw, query, key_raw, key, value, gate];
        Ok(Self { kernels, _buffers: [buffers, wide].into_iter().flatten().collect() })
    }

    // Every dispatch of the layer, in run order.
    pub fn dispatches(&self) -> Vec<Dispatch> {
        self.kernels.iter().flat_map(|kernel| kernel.dispatches().iter().cloned()).collect()
    }
}
