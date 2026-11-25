// Our error type.
use crate::error::EngineError;

// One line of the weight index.
use super::index::WeightEntry;

// write_buffer needs offsets and sizes that are multiples of 4 bytes (WebGPU's COPY_BUFFER_ALIGNMENT).
pub const WRITE_ALIGNMENT: u64 = 4;
// CLAUDE.md: one GPU buffer on the iPhone can be at most about 1073 MB.
pub const BUFFER_LIMIT_BYTES: u64 = 1_073_000_000;

// Where one tensor sits in the weight file. Its GPU buffer has exactly `length` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    // The first byte of the tensor in the file.
    pub start: u64,
    // The stored bytes of the tensor, rounded up to a multiple of 4.
    pub length: u64,
}

impl Span {
    // One past the last byte of the span.
    pub fn end(self) -> u64 {
        self.start + self.length
    }
}

// The span of every entry, in file order. Fails if write_buffer cannot fill a tensor, or if a buffer is too large.
// `entries` must be in file order. `max_buffer` is the smaller of the iPhone limit and the device limit.
pub fn spans(entries: &[WeightEntry], max_buffer: u64) -> Result<Vec<Span>, EngineError> {
    // Where the previous tensor ended, without its padding. Nothing came before the first tensor.
    let mut previous_end = 0;
    // Build the spans one after another, so each one can be checked against the one before.
    entries
        .iter()
        .map(|entry| {
            // The bytes that the file stores for this tensor.
            let stored = entry.stored_byte_length()?;
            // A tensor at an odd offset cannot be written with write_buffer. convert_weights.py aligns to 256.
            if !entry.offset.is_multiple_of(WRITE_ALIGNMENT) {
                return Err(EngineError::WeightLayout(format!("{} starts at byte {}, not a multiple of {WRITE_ALIGNMENT}", entry.name, entry.offset)));
            }
            // Two tensors that share bytes mean the index is wrong.
            if entry.offset < previous_end {
                return Err(EngineError::WeightLayout(format!("{} starts at byte {}, before the tensor before it ends at byte {previous_end}", entry.name, entry.offset)));
            }
            // The buffer size: the stored bytes rounded up to a multiple of 4.
            let length = stored.next_multiple_of(WRITE_ALIGNMENT);
            // Each tensor has its own buffer, so this is the only buffer size check that the loader needs.
            if length > max_buffer {
                return Err(EngineError::TensorTooLarge { name: entry.name.clone(), bytes: length, max: max_buffer });
            }
            // The next tensor must start after this one.
            previous_end = entry.offset + stored;
            // The rounded length cannot reach the next tensor: the next offset is a multiple of 4 and is at least `previous_end`.
            Ok(Span { start: entry.offset, length })
        })
        .collect()
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the layout and the index parser into the tests.
    use super::{EngineError, Span, spans};
    use crate::weights::WeightIndex;

    // Parse an index and give its entries in file order.
    fn entries(json: &str) -> Vec<super::WeightEntry> {
        WeightIndex::parse(json).unwrap().into_file_order()
    }

    // A good index gives one span per tensor, with the length rounded up to 4 bytes.
    #[test]
    fn rounds_each_tensor_up_to_whole_words() {
        // 3 float16 values (6 bytes), then 2 float32 values. The index lists them out of order.
        let list = entries(r#"[{"name": "b", "offset": 256, "shape": [2], "dtype": "float32"}, {"name": "a", "offset": 0, "shape": [3], "dtype": "float16"}]"#);
        // 6 bytes become 8.
        assert_eq!(spans(&list, 1000).unwrap(), vec![Span { start: 0, length: 8 }, Span { start: 256, length: 8 }]);
    }

    // A wrong index must stop the loader before any GPU buffer is made.
    #[test]
    fn refuses_bad_layouts_and_large_tensors() {
        // A tensor at byte 2.
        let unaligned = entries(r#"[{"name": "a", "offset": 2, "shape": [1], "dtype": "float32"}]"#);
        assert!(matches!(spans(&unaligned, 1000), Err(EngineError::WeightLayout(_))));
        // The second tensor starts inside the first one, which needs 16 bytes.
        let overlap = entries(r#"[{"name": "a", "offset": 0, "shape": [4], "dtype": "float32"}, {"name": "b", "offset": 8, "shape": [1], "dtype": "float32"}]"#);
        assert!(matches!(spans(&overlap, 1000), Err(EngineError::WeightLayout(_))));
        // 300 float32 values need 1200 bytes, over a limit of 1000.
        let large = entries(r#"[{"name": "a", "offset": 0, "shape": [300], "dtype": "float32"}]"#);
        assert!(matches!(spans(&large, 1000), Err(EngineError::TensorTooLarge { bytes: 1200, max: 1000, .. })));
    }
}
