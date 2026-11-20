// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// A kernel with its buffers.
use crate::dispatch::BoundKernel;
// The in-place add.
use crate::kernels::bind_add;
// The slots of the plan, and a view of a tensor in a buffer.
use crate::plan::{ActivationPool, SlotTensor};
// The weight buffers of the slot kernels.
use crate::weights::WeightBuffers;

// The sizes of the text encoder, and the slot of layer 0's input.
use super::plan::{EMBEDDINGS, TOKENS, WIDTH};

// The table of 49,408 token embeddings, 1024 values each. It stays on the GPU during the stage. A run copies 77 rows of it.
pub(super) const TOKEN_TABLE: &str = "text_model.embeddings.token_embedding.weight";
// The 77 position embeddings.
const POSITIONS: &str = "text_model.embeddings.position_embedding.weight";
// Bytes in one f32 value.
const F32_BYTES: u64 = 4;

// Bind `embeddings += position_embedding`, the second half of CLIPTextEmbeddings. The run copies the token rows in first.
// It is the same one f32 add per value as PyTorch, so the embeddings stay bit for bit the same.
pub(super) fn bind_positions(gpu: &Gpu, weights: &impl WeightBuffers, pool: &ActivationPool) -> Result<BoundKernel, EngineError> {
    // The position table and its shape.
    let positions = weights.float(gpu, POSITIONS)?;
    // One row per token, one value per channel.
    if positions.shape != [TOKENS as usize, WIDTH as usize] {
        return Err(EngineError::WeightShape { name: POSITIONS.to_string(), shape: positions.shape, expected: format!("[{TOKENS}, {WIDTH}]") });
    }
    // The table seen as a tensor of the size of the embeddings, so the add kernel can read it.
    let addend = SlotTensor { name: POSITIONS, buffer: positions.buffer.buffer(), dims: [TOKENS, WIDTH, 1] };
    // The add, in place on the embeddings slot.
    let kernel = bind_add(gpu, &pool.tensor(EMBEDDINGS)?, &addend)?;
    // Keep the table if it was made for this kernel.
    Ok(kernel.keeping(positions.buffer.into_owned()))
}

// Fail unless `ids` has exactly 77 ids, each a row of a table of `table_rows` rows.
pub(super) fn check_ids(ids: &[u32], table_rows: usize) -> Result<(), EngineError> {
    // A prompt has exactly 77 tokens, padded by the tokenizer.
    if ids.len() != TOKENS as usize {
        return Err(EngineError::LengthMismatch { actual: ids.len(), expected: TOKENS as usize });
    }
    // An id past the table would read bytes that are not a token.
    if let Some(&id) = ids.iter().find(|&&id| id as usize >= table_rows) {
        return Err(EngineError::WeightShape { name: TOKEN_TABLE.to_string(), shape: vec![table_rows, WIDTH as usize], expected: format!("a row below {table_rows}, not {id}") });
    }
    // All good.
    Ok(())
}

// Copy row ids[p] of the token table to row p of `embeddings`, for the 77 positions, in one submit.
// A buffer copy needs no bind group and no new buffer. `table_rows` is the row count of the table, for the id check.
pub(super) fn copy_token_rows(gpu: &Gpu, table: &wgpu::Buffer, table_rows: usize, ids: &[u32], embeddings: &wgpu::Buffer) -> Result<(), EngineError> {
    // 77 ids, each inside the table.
    check_ids(ids, table_rows)?;
    // The bytes of one row.
    let row_bytes = u64::from(WIDTH) * F32_BYTES;
    // Record the 77 copies.
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for (position, &id) in ids.iter().enumerate() {
        encoder.copy_buffer_to_buffer(table, u64::from(id) * row_bytes, embeddings, position as u64 * row_bytes, row_bytes);
    }
    // Send them. The next submit sees the rows, because the GPU runs submits in order.
    gpu.queue.submit([encoder.finish()]);
    // Done.
    Ok(())
}
