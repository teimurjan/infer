// Where each tensor sits in the file, and the write size rule.
use super::layout::{Span, WRITE_ALIGNMENT};

// The width of the carry, as a usize for slices.
const WORD: usize = WRITE_ALIGNMENT as usize;

// One write: some bytes of the stream that belong to one tensor.
#[derive(Debug)]
pub(crate) struct Piece<'a> {
    // The position of the tensor in the span list.
    pub(crate) tensor: usize,
    // Where the bytes go in the tensor's buffer. Always a multiple of 4.
    pub(crate) offset: u64,
    // The bytes. Their count is always a multiple of 4.
    pub(crate) bytes: &'a [u8],
}

// Follows a weight file that arrives in chunks of any size, and cuts it into 4-byte aligned writes, one tensor at a time.
// A chunk can end in the middle of a 4-byte word. Those 1 to 3 bytes wait in `carry` for the next chunk.
#[derive(Default)]
pub(crate) struct StreamCursor {
    // The file position of the first byte that has not been cut into pieces yet. Always a multiple of 4.
    position: u64,
    // The bytes after `position` that do not fill a whole word yet. At most 3.
    carry: Vec<u8>,
    // All bytes received so far.
    received: u64,
}

impl StreamCursor {
    // Take the next chunk of the file, and call `write` for each piece of it that falls inside a tensor.
    pub(crate) fn push(&mut self, spans: &[Span], chunk: &[u8], write: &mut impl FnMut(Piece)) {
        // Count the bytes now, before the chunk is cut.
        self.received += chunk.len() as u64;
        // The bytes that the carry needs to become one whole word. 0 if the carry is empty.
        let missing = (WORD - self.carry.len()) % WORD;
        // Take them from the front of the chunk, or the whole chunk if it is shorter.
        let (front, rest) = chunk.split_at(missing.min(chunk.len()));
        // Add them to the carry.
        self.carry.extend_from_slice(front);
        // A full carry is one aligned word: cut it. If the carry is still not full, `rest` is empty.
        if self.carry.len() == WORD {
            // Take the word out of the carry, so the carry is empty again.
            let word = std::mem::take(&mut self.carry);
            // Cut and write it.
            self.advance(spans, &word, write);
        }
        // The largest part of the rest that is whole words.
        let (whole, tail) = rest.split_at(rest.len() / WORD * WORD);
        // Cut and write it.
        self.advance(spans, whole, write);
        // The 0 to 3 bytes after the last whole word wait for the next chunk.
        self.carry.extend_from_slice(tail);
    }

    // The stream has ended. Write the carry, padded with zeros to one word, and give the number of bytes received.
    // The zeros land in the padding at the end of the last tensor buffer, which is zero already.
    pub(crate) fn finish(&mut self, spans: &[Span], write: &mut impl FnMut(Piece)) -> u64 {
        // A carry exists only if the file length is not a multiple of 4.
        if !self.carry.is_empty() {
            // Take the carry out and pad it to one word.
            let mut word = std::mem::take(&mut self.carry);
            word.resize(WORD, 0);
            // Cut and write it.
            self.advance(spans, &word, write);
        }
        // Every byte of the stream.
        self.received
    }

    // Cut `bytes`, which start at `position`, into pieces, write them, and move `position` past them.
    fn advance(&mut self, spans: &[Span], bytes: &[u8], write: &mut impl FnMut(Piece)) {
        // Write each piece.
        pieces(spans, self.position, bytes).for_each(&mut *write);
        // The next bytes start after these.
        self.position += bytes.len() as u64;
    }
}

// The pieces of `bytes`, which start at file position `position`, that fall inside a span.
// A span ends on a multiple of 4, so it also takes the 0 to 3 file bytes after its tensor. convert_weights.py writes zeros there.
// The other bytes between tensors are skipped.
fn pieces<'a>(spans: &'a [Span], position: u64, bytes: &'a [u8]) -> impl Iterator<Item = Piece<'a>> {
    // One past the last byte.
    let end = position + bytes.len() as u64;
    // The first span that ends after `position`. Spans are in file order and do not overlap, so their ends are in order too.
    let first = spans.partition_point(|span| span.end() <= position);
    // Walk the spans from there, while they start before `end`.
    spans[first..]
        .iter()
        .enumerate()
        .take_while(move |(_, span)| span.start < end)
        .map(move |(step, span)| {
            // The part of the bytes inside this span.
            let from = span.start.max(position);
            let to = span.end().min(end);
            // `from` and `to` lie inside `bytes`, so the differences fit in usize.
            Piece { tensor: first + step, offset: from - span.start, bytes: &bytes[(from - position) as usize..(to - position) as usize] }
        })
        // A span of 0 bytes gives an empty piece. There is nothing to write.
        .filter(|piece| !piece.bytes.is_empty())
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the cursor and the span into the tests.
    use super::{Span, StreamCursor};

    // A small file: tensor 0 has 6 bytes (padded to 8) at 0, tensor 1 has 4 bytes at 16, tensor 2 has 9 bytes at 32 and ends the file.
    // Bytes 6 and 7 go into the padding of buffer 0. The bytes 8..16 and 20..32 belong to no buffer.
    const SPANS: [Span; 3] = [Span { start: 0, length: 8 }, Span { start: 16, length: 4 }, Span { start: 32, length: 12 }];
    // The file length: tensor 2 ends at 32 + 9.
    const FILE_LENGTH: usize = 41;

    // Feed the file in chunks of `chunk_size` bytes, and build each tensor buffer from the pieces, as the GPU would.
    fn stream_in_chunks(file: &[u8], chunk_size: usize) -> (Vec<Vec<u8>>, u64) {
        // Zeroed buffers, as WebGPU makes them.
        let mut buffers: Vec<Vec<u8>> = SPANS.iter().map(|span| vec![0; span.length as usize]).collect();
        // Copy each piece into its buffer, and check the write rules of write_buffer.
        let mut write = |piece: super::Piece| {
            assert!(piece.offset % 4 == 0 && piece.bytes.len() % 4 == 0, "unaligned piece {piece:?}");
            buffers[piece.tensor][piece.offset as usize..][..piece.bytes.len()].copy_from_slice(piece.bytes);
        };
        // A fresh cursor.
        let mut cursor = StreamCursor::default();
        // Push the chunks.
        file.chunks(chunk_size).for_each(|chunk| cursor.push(&SPANS, chunk, &mut write));
        // End the stream.
        let received = cursor.finish(&SPANS, &mut write);
        (buffers, received)
    }

    // For every chunk size, each buffer must hold the file bytes of its span, then zeros where the file ends, and nothing from the gaps.
    #[test]
    fn every_chunk_size_gives_the_same_buffers() {
        // File bytes 1, 2, 3, ... so a byte in the wrong place is easy to see. No byte is 0.
        let file: Vec<u8> = (1..=FILE_LENGTH as u8).collect();
        // The expected buffers: the file bytes of each span, cut at the end of the file, then zeros up to the span length.
        let expected: Vec<Vec<u8>> = SPANS
            .iter()
            .map(|span| {
                // The file bytes of this span. Only the last span reaches past the end of the file.
                let mut bytes = file[span.start as usize..span.end().min(FILE_LENGTH as u64) as usize].to_vec();
                // Past the end of the file, the buffer stays zero.
                bytes.resize(span.length as usize, 0);
                bytes
            })
            .collect();
        // Chunk sizes 1 to the whole file, which covers every way a chunk can end inside a word.
        for chunk_size in 1..=FILE_LENGTH {
            // Stream the file.
            let (buffers, received) = stream_in_chunks(&file, chunk_size);
            // Same buffers as expected.
            assert_eq!(buffers, expected, "chunk size {chunk_size}");
            // Every byte was counted.
            assert_eq!(received, FILE_LENGTH as u64);
        }
    }

    // A stream that stops early must report fewer bytes, so the loader can refuse it.
    #[test]
    fn a_short_stream_reports_its_length() {
        // Only the first 20 bytes of the file arrive.
        let (_, received) = stream_in_chunks(&[7; 20], 3);
        // 20 bytes, so tensor 2 never arrived.
        assert_eq!(received, 20);
    }
}
