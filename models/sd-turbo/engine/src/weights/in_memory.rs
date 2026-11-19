// Our error type.
use crate::error::EngineError;

// The index format and the shared reader interface.
use super::{WeightIndex, WeightSource, WeightTensor};

// A whole .bin file from convert_weights.py, already in memory, and its parsed .json index.
// The browser fetches both files and passes them in. This works for small networks such as TAESD (about 10 MB).
pub struct WeightBytes {
    // The parsed index.
    index: WeightIndex,
    // Every byte of the .bin file.
    bytes: Vec<u8>,
}

impl WeightBytes {
    // Parse the index text and keep the bytes.
    pub fn new(index_json: &str, bytes: Vec<u8>) -> Result<Self, EngineError> {
        // A broken index stops here, before any tensor is read.
        Ok(Self { index: WeightIndex::parse(index_json)?, bytes })
    }
}

impl WeightSource for WeightBytes {
    // Copy the bytes of one tensor out of the file bytes.
    fn read(&self, name: &str) -> Result<WeightTensor, EngineError> {
        // Find the tensor in the index.
        let entry = self.index.entry(name)?;
        // The first byte. An offset that does not fit in usize is past the end of any file in memory.
        let start = usize::try_from(entry.offset).map_err(|_| out_of_range(name, entry.offset, self.bytes.len()))?;
        // One past the last byte. An overflow is past the end too.
        let end = start.checked_add(entry.byte_length()?).ok_or_else(|| out_of_range(name, entry.offset, self.bytes.len()))?;
        // The bytes of this tensor. A short file, or the wrong .bin for this index, fails here.
        let bytes = self.bytes.get(start..end).ok_or_else(|| out_of_range(name, entry.offset, self.bytes.len()))?;
        // The shape comes from the index. The bytes are copied, because the GPU upload needs its own list.
        Ok(WeightTensor { shape: entry.shape.clone(), bytes: bytes.to_vec() })
    }

    // Copy the bytes of one row out of the file bytes.
    fn read_row(&self, name: &str, row: usize) -> Result<Vec<u8>, EngineError> {
        // Where the row starts, and its length.
        let (offset, length) = self.index.entry(name)?.row_range(row)?;
        // The first byte. An offset that does not fit in usize is past the end of any file in memory.
        let start = usize::try_from(offset).map_err(|_| out_of_range(name, offset, self.bytes.len()))?;
        // The bytes of the row. A short file fails here.
        let bytes = start.checked_add(length).and_then(|end| self.bytes.get(start..end)).ok_or_else(|| out_of_range(name, offset, self.bytes.len()))?;
        // A copy of them.
        Ok(bytes.to_vec())
    }
}

// The error for a tensor that does not fit inside the file bytes.
fn out_of_range(name: &str, offset: u64, file_length: usize) -> EngineError {
    // Say which tensor, where it starts, and how long the file is.
    EngineError::WeightRead(format!("{name} starts at byte {offset} and does not fit in the {file_length} bytes of the file"))
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the reader and the interface into the tests.
    use super::{EngineError, WeightBytes, WeightSource};

    // One float32 tensor of 2 values at offset 4, in the format of convert_weights.py.
    const INDEX: &str = r#"[{"name": "bias", "offset": 4, "shape": [2], "dtype": "float32"}]"#;

    // The reader must return exactly the bytes at the offset, with the shape from the index.
    #[test]
    fn reads_the_bytes_at_the_offset() {
        // 4 bytes of padding, then 8 bytes of tensor.
        let weights = WeightBytes::new(INDEX, (0..12).collect()).unwrap();
        // Read the tensor.
        let tensor = weights.read("bias").unwrap();
        // Bytes 4 to 11.
        assert_eq!(tensor.bytes, vec![4, 5, 6, 7, 8, 9, 10, 11]);
        // The shape from the index.
        assert_eq!(tensor.shape, vec![2]);
    }

    // A file that is too short must fail, not return part of a tensor.
    #[test]
    fn refuses_a_tensor_past_the_end() {
        // 11 bytes: the tensor needs bytes 4 to 11, so one byte is missing.
        let weights = WeightBytes::new(INDEX, (0..11).collect()).unwrap();
        // The read must fail with a read error.
        assert!(matches!(weights.read("bias"), Err(EngineError::WeightRead(_))));
    }
}
