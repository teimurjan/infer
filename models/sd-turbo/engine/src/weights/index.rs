// HashMap finds an entry by its tensor name.
use std::collections::HashMap;

// Deserialize builds a WeightEntry from one JSON object.
use serde::Deserialize;

// Our error type.
use crate::error::EngineError;

// Size of one float32 value in bytes.
const F32_BYTES: usize = 4;
// Size of one float16 value in bytes.
const F16_BYTES: u64 = 2;
// Bits in one packed word. pack_weights.py packs the integers of each row into u32 words.
const WORD_BITS: u64 = 32;
// The only dtype that the kernels read today.
const FLOAT32: &str = "float32";

// One line of the index that convert_weights.py writes.
#[derive(Deserialize)]
pub(crate) struct WeightEntry {
    // The PyTorch name, for example "conv_in.weight".
    pub(crate) name: String,
    // Where the tensor starts in the .bin file, in bytes.
    pub(crate) offset: u64,
    // The size of each dimension, in PyTorch's order.
    pub(crate) shape: Vec<usize>,
    // "float32", "float16", or "int8" and "int4" for packed weights.
    pub(crate) dtype: String,
}

impl WeightEntry {
    // The number of bytes of this tensor. Fails for any dtype other than float32.
    pub(crate) fn byte_length(&self) -> Result<usize, EngineError> {
        // The kernels read float32 only, so a float16 file must stop here.
        if self.dtype != FLOAT32 {
            return Err(EngineError::WeightDtype { name: self.name.clone(), dtype: self.dtype.clone() });
        }
        // Values times 4 bytes. An empty shape is one value, as in PyTorch.
        Ok(self.shape.iter().product::<usize>() * F32_BYTES)
    }

    // Where row `row` of this float32 tensor starts in the .bin file, and its length in bytes. A row is everything after dimension 0.
    // The text encoder reads 77 rows of its token table this way, not all 49,408.
    pub(crate) fn row_range(&self, row: usize) -> Result<(u64, usize), EngineError> {
        // Dimension 0 counts the rows.
        let &rows = self.shape.first().ok_or_else(|| EngineError::WeightShape { name: self.name.clone(), shape: self.shape.clone(), expected: "at least one dimension for rows".to_string() })?;
        // A row past the end is an error, not bytes of the next tensor.
        if row >= rows {
            return Err(EngineError::WeightShape { name: self.name.clone(), shape: self.shape.clone(), expected: format!("a row below {rows}, not {row}") });
        }
        // The bytes of one row. byte_length checks that the tensor is float32.
        let row_bytes = self.byte_length()? / rows;
        // The start of the row, and its length.
        Ok((self.offset + (row * row_bytes) as u64, row_bytes))
    }

    // The number of bytes that this tensor takes in the .bin file, for every dtype that convert_weights.py writes.
    pub(crate) fn stored_byte_length(&self) -> Result<u64, EngineError> {
        // The number of values. An empty shape is one value, as in PyTorch.
        let values = self.shape.iter().map(|&size| size as u64).product::<u64>();
        // Pick the size rule of this dtype.
        match self.dtype.as_str() {
            // 4 bytes per value.
            FLOAT32 => Ok(values * F32_BYTES as u64),
            // 2 bytes per value.
            "float16" => Ok(values * F16_BYTES),
            // Packed rows of 8-bit or 4-bit integers, see pack_weights.py.
            "int8" => self.packed_byte_length(8),
            "int4" => self.packed_byte_length(4),
            // A dtype that the converter does not write.
            _ => Err(EngineError::UnknownDtype { name: self.name.clone(), dtype: self.dtype.clone() }),
        }
    }

    // The bytes of a packed weight: one row per output channel, each row padded to whole u32 words.
    fn packed_byte_length(&self, bits: u64) -> Result<u64, EngineError> {
        // Dim 0 is the output channel. A packed weight with no dimensions cannot have rows.
        let (&rows, rest) = self.shape.split_first().ok_or_else(|| EngineError::WeightShape { name: self.name.clone(), shape: self.shape.clone(), expected: "at least one dimension for packed rows".to_string() })?;
        // The values in one row: all dimensions after the first.
        let row_values = rest.iter().map(|&size| size as u64).product::<u64>();
        // Round the bits of one row up to whole words.
        let row_words = (row_values * bits).div_ceil(WORD_BITS);
        // 4 bytes per word, for every row.
        Ok(rows as u64 * row_words * (WORD_BITS / 8))
    }
}

// Every entry of one index, found by name.
pub(crate) struct WeightIndex {
    // Tensor name to entry.
    entries: HashMap<String, WeightEntry>,
}

impl WeightIndex {
    // Parse the JSON text of an index: a list of {name, offset, shape, dtype} objects.
    pub(crate) fn parse(json: &str) -> Result<Self, EngineError> {
        // serde_json checks that every object has the four fields with the right types.
        let list: Vec<WeightEntry> = serde_json::from_str(json).map_err(|error| EngineError::WeightIndex(error.to_string()))?;
        // Key each entry by its name.
        Ok(Self { entries: list.into_iter().map(|entry| (entry.name.clone(), entry)).collect() })
    }

    // Every entry, from the first byte of the file to the last. The weight stream needs this order.
    pub(crate) fn into_file_order(self) -> Vec<WeightEntry> {
        // Take the entries out of the map.
        let mut entries: Vec<WeightEntry> = self.entries.into_values().collect();
        // Sort them by where they start.
        entries.sort_by_key(|entry| entry.offset);
        entries
    }

    // The entry with this name.
    pub(crate) fn entry(&self, name: &str) -> Result<&WeightEntry, EngineError> {
        // A missing name usually means a typo in the engine or the wrong .bin file.
        self.entries.get(name).ok_or_else(|| EngineError::MissingWeight(name.to_string()))
    }
}

// The bytes that the tensors of one index take on the GPU: the sum of the stored lengths, without the padding between tensors.
pub fn weight_bytes_on_gpu(index_json: &str) -> Result<u64, EngineError> {
    // Parse the index.
    let index = WeightIndex::parse(index_json)?;
    // Add up the stored length of every tensor. The first unknown dtype stops the sum.
    index.entries.values().map(WeightEntry::stored_byte_length).sum()
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the index and the error into the tests.
    use super::{EngineError, WeightIndex};

    // Two lines in the format of convert_weights.py.
    const INDEX: &str = r#"[
{"name": "conv_in.bias", "offset": 0, "shape": [320], "dtype": "float32"},
{"name": "half", "offset": 1280, "shape": [2, 3], "dtype": "float16"}
]"#;

    // The entry must give back its offset and its byte length.
    #[test]
    fn finds_an_entry_and_its_byte_length() {
        // Parse the two lines.
        let index = WeightIndex::parse(INDEX).unwrap();
        // The bias starts at 0.
        assert_eq!(index.entry("conv_in.bias").unwrap().offset, 0);
        // 320 values x 4 bytes.
        assert_eq!(index.entry("conv_in.bias").unwrap().byte_length().unwrap(), 1280);
    }

    // The stored size must follow the rules of convert_weights.py and pack_weights.py for each dtype.
    #[test]
    fn stored_byte_length_of_each_dtype() {
        // One entry of each dtype. The int shapes have rows that do not fill a whole word.
        let index = WeightIndex::parse(r#"[
{"name": "f32", "offset": 0, "shape": [3], "dtype": "float32"},
{"name": "f16", "offset": 256, "shape": [3], "dtype": "float16"},
{"name": "q8", "offset": 512, "shape": [2, 5], "dtype": "int8"},
{"name": "q4", "offset": 768, "shape": [2, 9], "dtype": "int4"},
{"name": "odd", "offset": 1024, "shape": [2], "dtype": "bfloat16"}
]"#)
        .unwrap();
        // The stored length of one entry.
        let length = |name: &str| index.entry(name).unwrap().stored_byte_length();
        // 3 x 4 bytes.
        assert_eq!(length("f32").unwrap(), 12);
        // 3 x 2 bytes.
        assert_eq!(length("f16").unwrap(), 6);
        // 2 rows of 5 bytes, each rounded up to 2 words of 4 bytes.
        assert_eq!(length("q8").unwrap(), 16);
        // 2 rows of 9 x 4 = 36 bits, each rounded up to 2 words.
        assert_eq!(length("q4").unwrap(), 16);
        // An unknown dtype must stop the loader.
        assert!(matches!(length("odd"), Err(EngineError::UnknownDtype { .. })));
    }

    // In the real files, each tensor must end in the 256 bytes before the next one starts.
    // convert_weights.py pads only to the next multiple of 256, so a wrong size rule shows up as a larger gap or an overlap.
    #[test]
    fn stored_lengths_match_the_real_files() {
        // The weights folder is in the project folder, one level above engine/.
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../weights");
        // Every index that the converter wrote. unet_packed.json has the int4 and int8 tensors.
        for name in ["taesd", "unet_packed", "text_encoder", "unet", "vae"] {
            // Read and parse the index.
            let json = std::fs::read_to_string(folder.join(format!("{name}.json"))).unwrap();
            // The entries in file order.
            let entries = WeightIndex::parse(&json).unwrap().into_file_order();
            // Each tensor and the one after it.
            for pair in entries.windows(2) {
                // Where the first tensor ends.
                let end = pair[0].offset + pair[0].stored_byte_length().unwrap();
                // The gap is the alignment padding: 0 to 255 bytes.
                let gap = pair[1].offset.checked_sub(end).unwrap_or_else(|| panic!("{name}: {} overlaps {}", pair[0].name, pair[1].name));
                assert!(gap < 256, "{name}: {} bytes between {} and {}", gap, pair[0].name, pair[1].name);
            }
        }
    }

    // A wrong row offset would give the text encoder the embedding of another token.
    #[test]
    fn a_row_starts_after_the_earlier_rows() {
        // A [3, 2] float32 tensor at byte 256: 8 bytes per row.
        let index = WeightIndex::parse(r#"[{"name": "table", "offset": 256, "shape": [3, 2], "dtype": "float32"}]"#).unwrap();
        // The table.
        let entry = index.entry("table").unwrap();
        // Row 2 starts 2 x 8 bytes after the tensor.
        assert_eq!(entry.row_range(2).unwrap(), (272, 8));
        // There is no row 3.
        assert!(matches!(entry.row_range(3), Err(EngineError::WeightShape { .. })));
    }

    // A float16 tensor and a missing name must fail, not return wrong bytes.
    #[test]
    fn refuses_float16_and_missing_names() {
        // Parse the two lines.
        let index = WeightIndex::parse(INDEX).unwrap();
        // float16 is not read by the kernels yet.
        assert!(matches!(index.entry("half").unwrap().byte_length(), Err(EngineError::WeightDtype { .. })));
        // No such name.
        assert!(matches!(index.entry("nothing"), Err(EngineError::MissingWeight(_))));
    }
}
