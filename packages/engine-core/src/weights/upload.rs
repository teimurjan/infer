// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;
// A buffer that counts its bytes.
use crate::memory::GpuBuffer;

// The cursor that cuts the stream into aligned pieces.
use super::cursor::{Piece, StreamCursor};
// The tensors on the GPU, which the upload gives back at the end.
use super::gpu_weights::{GpuTensor, GpuWeights};
// One line of the index, and the index parser.
use super::index::{WeightEntry, WeightIndex};
// Where each tensor sits in the file, and the iPhone buffer limit.
use super::layout::{BUFFER_LIMIT_BYTES, Span, spans};

// After this many bytes, the caller waits for the GPU (Gpu::finish_uploads).
// write_buffer copies each piece into a staging area of the browser or of wgpu. The wait lets that area empty, so it cannot grow to the whole file.
const SETTLE_BYTES: u64 = 32 * 1024 * 1024;

// One weight file that streams into GPU buffers, one chunk at a time.
// Each tensor gets its own buffer when its first byte arrives, so the memory counter grows with the stream.
pub struct WeightUpload {
    // The index entries, in file order.
    entries: Vec<WeightEntry>,
    // Where each entry sits in the file, and its buffer size.
    spans: Vec<Span>,
    // The buffer of each entry, or None before its first byte arrived.
    buffers: Vec<Option<GpuBuffer>>,
    // Where the stream is now.
    cursor: StreamCursor,
    // Bytes written since the last time that the caller waited for the GPU.
    unsettled: u64,
}

impl WeightUpload {
    // Parse the index of one weight file and check its layout. No GPU buffer exists yet.
    pub fn begin(gpu: &Gpu, index_json: &str) -> Result<Self, EngineError> {
        // The entries from the first byte of the file to the last.
        let entries = WeightIndex::parse(index_json)?.into_file_order();
        // A buffer must fit both the iPhone limit and the limit of this device.
        let spans = spans(&entries, BUFFER_LIMIT_BYTES.min(gpu.max_buffer_bytes()))?;
        // No buffers yet: one empty slot per entry.
        let buffers = entries.iter().map(|_| None).collect();
        // The stream starts at byte 0.
        Ok(Self { entries, spans, buffers, cursor: StreamCursor::default(), unsettled: 0 })
    }

    // Write one chunk of the file into the GPU buffers. The chunk can have any size and end anywhere.
    // Returns true when the caller must wait for the GPU before the next chunk.
    pub fn write(&mut self, gpu: &Gpu, chunk: &[u8]) -> bool {
        // Borrow the fields apart, so the closure can change `buffers` while the cursor moves.
        let Self { entries, spans, buffers, cursor, .. } = self;
        // Cut the chunk into pieces and copy each piece into its tensor buffer.
        cursor.push(spans, chunk, &mut |piece| write_piece(gpu, entries, spans, buffers, piece));
        // Count the bytes since the last wait.
        self.unsettled += chunk.len() as u64;
        // Not time to wait yet.
        if self.unsettled < SETTLE_BYTES {
            return false;
        }
        // The caller waits now, so the count starts again.
        self.unsettled = 0;
        true
    }

    // The stream has ended. Check that every tensor arrived, and give back the tensors on the GPU.
    // The caller must still wait for the GPU (Gpu::finish_uploads) before a kernel reads them.
    pub fn finish(mut self, gpu: &Gpu) -> Result<GpuWeights, EngineError> {
        // Borrow the fields apart, as in `write`.
        let Self { entries, spans, buffers, cursor, .. } = &mut self;
        // Write the last 1 to 3 bytes, if the file length is not a multiple of 4.
        let received = cursor.finish(spans, &mut |piece| write_piece(gpu, entries, spans, buffers, piece));
        // The file must reach the last byte of the last tensor. A broken download or a cut cache file stops here.
        let needed = match entries.last() {
            // The end of the last tensor, without its padding.
            Some(entry) => entry.offset + entry.stored_byte_length()?,
            // An index with no tensors needs no bytes.
            None => 0,
        };
        if received < needed {
            return Err(EngineError::StreamEnded { received, needed });
        }
        // Pair each entry with its buffer.
        let tensors = self
            .entries
            .into_iter()
            .zip(self.buffers)
            .zip(self.spans)
            .map(|((entry, buffer), span)| {
                // The stored bytes, for the summary and the tests.
                let bytes = entry.stored_byte_length()?;
                // Every byte arrived, so only a tensor of 0 bytes can still have no buffer. It gets an empty one.
                let buffer = buffer.unwrap_or_else(|| gpu.empty_weight_buffer(&entry.name, span.length));
                // Key the tensor by its name.
                Ok((entry.name, GpuTensor { buffer, shape: entry.shape, dtype: entry.dtype, bytes }))
            })
            .collect::<Result<_, EngineError>>()?;
        // All tensors of the file.
        Ok(GpuWeights::new(tensors))
    }
}

// Copy one piece into the buffer of its tensor. Make the buffer first if this is the first piece of that tensor.
fn write_piece(gpu: &Gpu, entries: &[WeightEntry], spans: &[Span], buffers: &mut [Option<GpuBuffer>], piece: Piece) {
    // The buffer of this tensor, made now if it does not exist yet. Its size is the span length.
    let buffer = buffers[piece.tensor].get_or_insert_with(|| gpu.empty_weight_buffer(&entries[piece.tensor].name, spans[piece.tensor].length));
    // Queue the copy. In the browser this is GPUQueue.writeBuffer on a view of the piece in WASM memory.
    gpu.queue.write_buffer(buffer, piece.offset, piece.bytes);
}
