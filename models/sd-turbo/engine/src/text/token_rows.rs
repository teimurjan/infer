// The packed token table: a kernel that unpacks the rows of the prompt's tokens into the embeddings slot.
// The float32 table does not need it: its rows are copied with buffer copies (copy_token_rows in embed.rs).

// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// A kernel ready to run, and the function that records runs of it.
use crate::dispatch::{Dispatch, Work, encode_runs};
// Buffers that count their bytes.
use crate::memory::GpuBuffer;
// The bit width of a packed weight.
use crate::layers::WeightBits;

// The sizes of the text encoder.
use super::plan::{TOKENS, WIDTH};
// The checks of the 77 ids, shared with the float copy.
use super::embed::check_ids;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("token_rows_packed.wgsl");
// Invocations per workgroup. It must match @workgroup_size(256) in token_rows_packed.wgsl.
const WORKGROUP_SIZE: u32 = 256;
// Bits in one packed word.
const WORD_BITS: u32 = 32;

// The kernel that writes the 77 token rows of a prompt from the packed table, with the buffer of the ids.
pub(super) struct PackedTokenRows {
    // The kernel, bound to the ids, the table, the scales and the embeddings slot.
    dispatch: Dispatch,
    // The 77 ids of the prompt. Each run writes them first.
    ids: GpuBuffer,
    // The sizes. The bind group reads it, so it lives as long as the kernel.
    _shape: GpuBuffer,
    // The rows of the table, for the id check.
    table_rows: usize,
}

impl PackedTokenRows {
    // Bind the kernel to the packed table `words` with its `scale` words, and to the embeddings slot `embeddings`.
    pub(super) fn bind(gpu: &Gpu, words: &wgpu::Buffer, scale: &wgpu::Buffer, bits: WeightBits, table_rows: usize, embeddings: &wgpu::Buffer) -> Result<Self, EngineError> {
        // The rows of the table must fit the u32 of the shader.
        let rows = u32::try_from(table_rows).map_err(|_| EngineError::TooManyValues { count: table_rows, max: u32::MAX as usize })?;
        // Bits per integer, and words per row: each row starts on a word, as pack_weights.py writes it.
        let bits = bits.value();
        let row_words = (WIDTH * bits).div_ceil(WORD_BITS);
        // The field order of the Shape struct in token_rows_packed.wgsl.
        let shape = gpu.uniform_buffer("token rows shape", &[WIDTH, bits, row_words, rows]);
        // One u32 per position. The buffer helper counts 4-byte values.
        let ids = gpu.empty_storage_buffer("token ids", TOKENS as usize);
        // One invocation per output value: 77 x 1024.
        let workgroups = [(TOKENS * WIDTH).div_ceil(WORKGROUP_SIZE), 1, 1];
        // It writes 77 rows of floats. The packed bytes that it reads are far fewer, so the output is the work.
        let work = Work::f32_values("token rows", u64::from(TOKENS * WIDTH));
        // Bindings: the sizes, the ids, the words, the scales, the embeddings.
        let dispatch = Dispatch::new(gpu, "token rows", SHADER, &[&shape, &ids, words, scale, embeddings], workgroups, work)?;
        // Ready to run.
        Ok(Self { dispatch, ids, _shape: shape, table_rows })
    }

    // Write the 77 ids, then submit the kernel. The next submit sees the rows, because the GPU runs submits in order.
    pub(super) fn run(&self, gpu: &Gpu, ids: &[u32]) -> Result<(), EngineError> {
        // 77 ids, each inside the table.
        check_ids(ids, self.table_rows)?;
        // The ids as little-endian bytes, the byte order of the GPU buffer.
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        gpu.queue.write_buffer(&self.ids, 0, &bytes);
        // One run of the kernel.
        gpu.queue.submit([encode_runs(gpu, std::slice::from_ref(&self.dispatch), 1)]);
        // Done.
        Ok(())
    }
}
