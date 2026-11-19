// cases.rs lists the SD-Turbo attention calls that we test, and checks them.
mod cases;
// slots.rs connects the attention kernel to the slots of the activation plan.
mod slots;

// The kinds of GPU buffer that the memory counter keeps apart.
use crate::memory::MemoryCategory;
// The pipeline compiler, a kernel ready to run, and the function that records runs of it.
use crate::dispatch::{Dispatch, Work, compute_pipeline, encode_runs};
// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;
// Buffers that count their bytes.
use crate::memory::GpuBuffer;
// The shared size checks.
use crate::checks::{check_length, positive_dimension};

// The test cases, for the native test and the web page.
pub use cases::{ATTENTION_CASES, AttentionCase, AttentionReport, check_attention, find_attention_case};
// The attention of the UNet stage.
pub(crate) use slots::{AttentionSlots, bind_attention_groups, bind_attention_into};

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("attention.wgsl");
// Values per head. attention.wgsl keeps a query row and a sum of one head in registers, so the size is fixed.
// Every attention layer of SD-Turbo uses heads of 64.
const HEAD_SIZE: u32 = 64;
// Query rows per workgroup. It must match ROWS in attention.wgsl. 16 invocations share each row, so a workgroup has 256.
const ROWS_PER_WORKGROUP: u32 = 16;

// The sizes of one attention call.
#[derive(Clone, Copy)]
pub struct AttentionShape {
    // Query rows, for example 4096 for a 64 x 64 latent.
    pub query_tokens: u32,
    // Key and value rows. Self attention: the same as query_tokens. Cross attention: 77 text tokens.
    pub key_tokens: u32,
    // Heads, for example 5.
    pub heads: u32,
    // Values per head. It must be 64.
    pub head_size: u32,
}

impl AttentionShape {
    // Operations in one call, counting a multiply-add as 2. The scores and the weighted sum each need
    // query_tokens x key_tokens x head_size multiply-adds per head. The softmax adds much less, so we leave it out.
    pub fn operations(&self) -> f64 {
        4.0 * f64::from(self.query_tokens) * f64::from(self.key_tokens) * f64::from(self.head_size) * f64::from(self.heads)
    }

    // Bytes that the full query_tokens x key_tokens table of scores would need for all heads. We never make it.
    pub fn full_scores_bytes(&self) -> u64 {
        u64::from(self.query_tokens) * u64::from(self.key_tokens) * u64::from(self.heads) * F32_BYTES
    }

    // Values in one query, key or value row: every head side by side.
    fn channels(&self) -> usize {
        self.heads as usize * self.head_size as usize
    }
}

// The three tensors of one attention call, each [tokens][heads * head_size].
pub struct AttentionTensors<'a> {
    // The query rows.
    pub query: &'a [f32],
    // The key rows.
    pub key: &'a [f32],
    // The value rows.
    pub value: &'a [f32],
}

// Bytes in one f32.
const F32_BYTES: u64 = 4;

// Attention with its data on the GPU, ready to run many times. One run is one dispatch per chunk.
pub(crate) struct PreparedAttention {
    // One dispatch per chunk, in order.
    dispatches: Vec<Dispatch>,
    // The buffer the kernel writes.
    output: GpuBuffer,
    // The query, key, value, scores and chunk buffers. We keep them, because the browser frees a buffer when Rust drops it.
    _buffers: Vec<GpuBuffer>,
}

impl PreparedAttention {
    // The dispatches of one run, for timing.
    pub(crate) fn dispatches(&self) -> &[Dispatch] {
        &self.dispatches
    }

    // Run every chunk once and copy the output back to the CPU.
    pub(crate) async fn run(&self, gpu: &Gpu) -> Result<Vec<f32>, EngineError> {
        // Record one run and send it to the GPU.
        gpu.queue.submit([encode_runs(gpu, &self.dispatches, 1)]);
        // Copy the output back.
        gpu.read_f32(&self.output).await
    }
}

// Check the sizes, copy the tensors to the GPU, and connect one dispatch per chunk of `chunk_rows` query rows.
pub(crate) fn prepare_attention(gpu: &Gpu, shape: &AttentionShape, chunk_rows: u32, tensors: &AttentionTensors) -> Result<PreparedAttention, EngineError> {
    // Zero sizes would make empty buffers or a division by zero in the softmax.
    let query_tokens = positive_dimension("query_tokens", shape.query_tokens)?;
    let key_tokens = positive_dimension("key_tokens", shape.key_tokens)?;
    let heads = positive_dimension("heads", shape.heads)?;
    let chunk_rows = positive_dimension("chunk_rows", chunk_rows)?;
    // The kernel supports only one head size.
    if shape.head_size != HEAD_SIZE {
        return Err(EngineError::HeadSize { value: shape.head_size, supported: HEAD_SIZE });
    }
    // A chunk larger than the queries would only waste scratch memory.
    let chunk_rows = chunk_rows.min(query_tokens);
    // Each tensor must have one row per token.
    check_length("query", tensors.query, query_tokens as usize * shape.channels())?;
    check_length("key", tensors.key, key_tokens as usize * shape.channels())?;
    check_length("value", tensors.value, key_tokens as usize * shape.channels())?;
    // The scratch holds the scores of one chunk for all heads.
    let scores_values = chunk_rows as usize * key_tokens as usize * heads as usize;
    // One binding must cover all of it.
    let scores_bytes = scores_values as u64 * F32_BYTES;
    if scores_bytes > gpu.max_storage_binding_bytes() {
        return Err(EngineError::ScoresTooLarge { bytes: scores_bytes, max: gpu.max_storage_binding_bytes() });
    }
    // The tensors go to @binding(1) to @binding(3), the scratch to @binding(4), the output to @binding(5).
    let query = gpu.storage_buffer_from("attention query", tensors.query, MemoryCategory::Activations);
    let key = gpu.storage_buffer_from("attention key", tensors.key, MemoryCategory::Activations);
    let value = gpu.storage_buffer_from("attention value", tensors.value, MemoryCategory::Activations);
    let scores = gpu.empty_storage_buffer("attention scores", scores_values);
    let output = gpu.empty_storage_buffer("attention output", tensors.query.len());
    // Compile once. Every chunk uses the same pipeline with its own sizes.
    let pipeline = compute_pipeline(gpu, "attention", SHADER);
    // One sizes buffer and one dispatch per chunk.
    let (dispatches, chunk_buffers): (Vec<Dispatch>, Vec<GpuBuffer>) = chunks(query_tokens, chunk_rows)
        .map(|(start, rows)| {
            // The field order of the Shape struct in attention.wgsl.
            // The last 0 turns the causal mask off: these test cases are the UNet attentions.
            let sizes = gpu.uniform_buffer("attention chunk", &[key_tokens, heads, chunk_rows, start, rows, 0]);
            // x covers the rows of the chunk, 16 per workgroup. y picks the head.
            let workgroups = [rows.div_ceil(ROWS_PER_WORKGROUP), heads, 1];
            // The query and output rows of the chunk, and every key and value, as in slots.rs.
            let channels = shape.channels() as u64;
            let work = Work::f32_values("attention", 2 * u64::from(rows) * channels + 2 * u64::from(key_tokens) * channels);
            // Connect the buffers in binding order.
            let dispatch = Dispatch::with_pipeline(gpu, "attention", &pipeline, &[&sizes, &query, &key, &value, &scores, &output], workgroups, work)?;
            // Keep the sizes buffer with its dispatch.
            Ok((dispatch, sizes))
        })
        // Stop at the first error.
        .collect::<Result<Vec<_>, EngineError>>()?
        // Split the pairs into two lists.
        .into_iter()
        .unzip();
    // Keep every buffer alive as long as the dispatches.
    let buffers = [query, key, value, scores].into_iter().chain(chunk_buffers).collect();
    // Everything together.
    Ok(PreparedAttention { dispatches, output, _buffers: buffers })
}

// Split `total` query rows into chunks of `size` rows: (first row, rows). The last chunk can be shorter.
fn chunks(total: u32, size: u32) -> impl Iterator<Item = (u32, u32)> {
    // Start at 0, size, 2 x size, and so on. Each chunk ends at the next start or at the end.
    (0..total).step_by(size as usize).map(move |start| (start, size.min(total - start)))
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function into the tests.
    use super::chunks;

    // A wrong last chunk would leave the last rows of the output at 0, or read past the queries.
    #[test]
    fn last_chunk_holds_the_remaining_rows() {
        // 4096 rows in chunks of 1000: four full chunks and 96 rows left.
        let expected = vec![(0, 1000), (1000, 1000), (2000, 1000), (3000, 1000), (4000, 96)];
        assert_eq!(chunks(4096, 1000).collect::<Vec<_>>(), expected);
        // A size that divides evenly has no short chunk.
        assert_eq!(chunks(512, 256).collect::<Vec<_>>(), vec![(0, 256), (256, 256)]);
    }
}
