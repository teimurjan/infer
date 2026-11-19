// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// A kernel ready to record, a kernel with its buffers, and the shader compiler.
use crate::dispatch::{BoundKernel, Dispatch, Work, compute_pipeline};
// The slot size check.
use crate::checks::check_buffer_holds;
// A tensor in a slot of the activation plan.
use crate::plan::SlotTensor;

// The shader, its fixed head size, its rows per workgroup, and the chunk split of attention/mod.rs.
use super::{HEAD_SIZE, ROWS_PER_WORKGROUP, SHADER, chunks};

// The four tensors that attention reads or writes, each in its slot.
pub(crate) struct AttentionSlots<'a> {
    // [query tokens, heads * 64, 1].
    pub(crate) query: SlotTensor<'a>,
    // [key tokens, heads * 64, 1].
    pub(crate) key: SlotTensor<'a>,
    // [key tokens, heads * 64, 1].
    pub(crate) value: SlotTensor<'a>,
    // [heads, key tokens, chunk rows]. Its size in the plan sets the head count and the chunk.
    pub(crate) scores: SlotTensor<'a>,
    // [query tokens, heads * 64, 1].
    pub(crate) output: SlotTensor<'a>,
    // Query row i sees only keys 0 to i. The text encoder uses this, the UNet does not.
    pub(crate) causal: bool,
}

// Connect the attention kernel to its slots: one dispatch and one uniform per chunk of query rows, all made once.
// All chunks go into one kernel, so they run in one submit.
pub(crate) fn bind_attention_into(gpu: &Gpu, slots: &AttentionSlots) -> Result<BoundKernel, EngineError> {
    // One group that holds every query row.
    let mut groups = bind_attention_groups(gpu, slots, u32::MAX)?;
    // There is always exactly one group, because the query rows are at least 1 and at most u32::MAX.
    groups.pop().ok_or(EngineError::ZeroDimension { dimension: "attention query rows" })
}

// The same chunks as bind_attention_into, but split into groups of at most `rows_per_group` query rows, in row order.
// Each group is its own kernel, so each can be its own submit. A group always has at least one whole chunk.
pub(crate) fn bind_attention_groups(gpu: &Gpu, slots: &AttentionSlots, rows_per_group: u32) -> Result<Vec<BoundKernel>, EngineError> {
    // The plan gives the head count, the key count and the chunk rows of the scratch.
    let [heads, key_tokens, chunk_rows] = slots.scores.dims;
    // The query rows and their width.
    let [query_tokens, channels, _] = slots.query.dims;
    // Every head has 64 values, so the width must be heads x 64.
    if channels != heads * HEAD_SIZE {
        return Err(EngineError::HeadSize { value: channels / heads.max(1), supported: HEAD_SIZE });
    }
    // The key and the value must have the planned key count and the same width. The output has the size of the query.
    for (tensor, expected) in [(&slots.key, [key_tokens, channels, 1]), (&slots.value, [key_tokens, channels, 1]), (&slots.output, slots.query.dims)] {
        if tensor.dims != expected {
            return Err(EngineError::PlanMismatch { tensor: tensor.name.to_string(), planned: tensor.dims, actual: expected });
        }
    }
    // The scores of one chunk must fit one binding.
    let scores_bytes = u64::from(heads) * u64::from(key_tokens) * u64::from(chunk_rows) * 4;
    if scores_bytes > gpu.max_storage_binding_bytes() {
        return Err(EngineError::ScoresTooLarge { bytes: scores_bytes, max: gpu.max_storage_binding_bytes() });
    }
    // The slots must hold their tensors.
    check_buffer_holds("attention query", slots.query.buffer, query_tokens as usize * channels as usize)?;
    check_buffer_holds("attention key", slots.key.buffer, key_tokens as usize * channels as usize)?;
    check_buffer_holds("attention value", slots.value.buffer, key_tokens as usize * channels as usize)?;
    check_buffer_holds("attention scores", slots.scores.buffer, (scores_bytes / 4) as usize)?;
    check_buffer_holds("attention output", slots.output.buffer, query_tokens as usize * channels as usize)?;
    // Compile once. Every chunk uses the same pipeline with its own sizes.
    let pipeline = compute_pipeline(gpu, "attention", SHADER);
    // The buffers in binding order 1 to 5. Binding 0 is the sizes of each chunk.
    let buffers = [slots.query.buffer, slots.key.buffer, slots.value.buffer, slots.scores.buffer, slots.output.buffer];
    // Whole chunks per group. A group smaller than one chunk still holds one chunk.
    let chunks_per_group = (rows_per_group / chunk_rows).max(1) as usize;
    // The finished groups, and the dispatches and sizes buffers of the group that is filling now.
    let mut groups = Vec::new();
    let mut dispatches = Vec::new();
    let mut uniforms = Vec::new();
    for (start, rows) in chunks(query_tokens, chunk_rows) {
        // The field order of the Shape struct in attention.wgsl.
        let sizes = gpu.uniform_buffer("attention chunk", &[key_tokens, heads, chunk_rows, start, rows, u32::from(slots.causal)]);
        // The sizes first, then the five tensors.
        let bound: Vec<&wgpu::Buffer> = std::iter::once(&*sizes).chain(buffers).collect();
        // The query and output rows of the chunk, and every key and value. The scores are scratch, so they do not count.
        let work = Work::f32_values("attention", 2 * u64::from(rows) * u64::from(channels) + 2 * u64::from(key_tokens) * u64::from(channels));
        // x covers the rows of the chunk, 16 per workgroup. y picks the head.
        dispatches.push(Dispatch::with_pipeline(gpu, "attention", &pipeline, &bound, [rows.div_ceil(ROWS_PER_WORKGROUP), heads, 1], work)?);
        // Keep the sizes with the dispatch.
        uniforms.push(sizes);
        // A full group becomes one kernel, and the next group starts empty.
        if dispatches.len() == chunks_per_group {
            groups.push(BoundKernel::new(std::mem::take(&mut dispatches), std::mem::take(&mut uniforms)));
        }
    }
    // The last group, if it has fewer chunks than a full one.
    if !dispatches.is_empty() {
        groups.push(BoundKernel::new(dispatches, uniforms));
    }
    // Every group, in row order.
    Ok(groups)
}
