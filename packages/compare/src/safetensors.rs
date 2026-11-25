// fs reads the whole file into memory.
use std::fs;
// Path is the file path we read.
use std::path::Path;

// Value is one JSON value from the header.
use serde_json::Value;

// Our error type.
use crate::error::CompareError;
// Our hand-written float16 decoder.
use crate::float16::f16_to_f32;

// One tensor from a safetensors file, with every value turned into f32.
pub struct Tensor {
    // The key of the tensor in the JSON header, for example "mid_block".
    pub name: String,
    // The size of each dimension, for example [1, 1280, 8, 8].
    pub shape: Vec<usize>,
    // All values in row-major order (the last dimension changes fastest).
    pub values: Vec<f32>,
}

// The first 8 bytes of the file hold the length of the JSON header.
const LENGTH_BYTES: usize = 8;

// Read every tensor in a safetensors file.
// File layout: [8-byte header length, little-endian] [JSON header] [raw tensor bytes].
pub fn read_tensors(path: &Path) -> Result<Vec<Tensor>, CompareError> {
    // Read the whole file. Golden files are small enough for this.
    let bytes = fs::read(path).map_err(|source| CompareError::Io { path: path.to_path_buf(), source })?;
    // The first 8 bytes. A shorter file cannot be safetensors.
    let length_bytes: [u8; LENGTH_BYTES] = bytes
        // Take bytes 0..8 if they exist.
        .get(..LENGTH_BYTES)
        // Turn the slice into a fixed array of 8 bytes.
        .and_then(|slice| slice.try_into().ok())
        // No 8 bytes means the file is too short.
        .ok_or_else(|| header_error(path, "file is shorter than 8 bytes"))?;
    // Read the 8 bytes as an unsigned 64-bit number, lowest byte first.
    let header_length = usize::try_from(u64::from_le_bytes(length_bytes))
        // A length that does not fit in memory is not a real header.
        .map_err(|_| header_error(path, "header length is too large"))?;
    // The JSON header ends here, and the raw data starts here.
    let data_start = LENGTH_BYTES
        // Add without overflow.
        .checked_add(header_length)
        // An overflow means a broken length.
        .ok_or_else(|| header_error(path, "header length is too large"))?;
    // The JSON text of the header.
    let header_bytes = bytes
        // Take the bytes between the length and the data.
        .get(LENGTH_BYTES..data_start)
        // The file ended before the header did.
        .ok_or_else(|| header_error(path, "file is shorter than its header length"))?;
    // Parse the JSON. The top level is an object: tensor name -> {dtype, shape, data_offsets}.
    let header: serde_json::Map<String, Value> = serde_json::from_slice(header_bytes)
        // Bad JSON means a broken file.
        .map_err(|error| header_error(path, &format!("header is not a JSON object: {error}")))?;
    // Everything after the header is raw tensor bytes. data_offsets count from here.
    let data = &bytes[data_start..];
    // Read each tensor described in the header.
    header
        // Go through each key and its value.
        .iter()
        // "__metadata__" is optional free text, not a tensor.
        .filter(|(name, _)| name.as_str() != "__metadata__")
        // Decode one tensor.
        .map(|(name, entry)| read_tensor(path, name, entry, data))
        // Stop at the first error, or return all tensors.
        .collect()
}

// Decode one tensor from its header entry and the raw data.
fn read_tensor(path: &Path, name: &str, entry: &Value, data: &[u8]) -> Result<Tensor, CompareError> {
    // The type of each number, for example "F32" or "F16".
    let dtype = entry["dtype"]
        // It must be a string.
        .as_str()
        // Missing or wrong type means a broken header.
        .ok_or_else(|| header_error(path, &format!("tensor {name} has no dtype string")))?;
    // The size of each dimension.
    let shape = usize_list(&entry["shape"])
        // Missing or wrong type means a broken header.
        .ok_or_else(|| header_error(path, &format!("tensor {name} has no valid shape")))?;
    // Where this tensor starts and ends inside the data section, in bytes.
    let offsets = usize_list(&entry["data_offsets"])
        // Missing or wrong type means a broken header.
        .ok_or_else(|| header_error(path, &format!("tensor {name} has no valid data_offsets")))?;
    // There must be exactly two numbers: start and end.
    let [start, end] = offsets[..] else {
        // Any other count is a broken header.
        return Err(header_error(path, &format!("tensor {name} needs 2 data_offsets")));
    };
    // The raw bytes of this tensor.
    let raw = data
        // Take bytes start..end of the data section.
        .get(start..end)
        // Offsets outside the file mean a broken header.
        .ok_or_else(|| header_error(path, &format!("tensor {name} offsets are outside the file")))?;
    // The number of values is the product of the dimensions. checked_mul stops on overflow.
    let count = shape.iter().try_fold(1usize, |total, &size| total.checked_mul(size))
        // An overflow means a broken shape.
        .ok_or_else(|| header_error(path, &format!("tensor {name} shape is too large")))?;
    // How many bytes one value takes. Any type other than F32, F16 and I64 is not supported.
    let value_size = bytes_per_value(dtype)
        // Unknown type: say which tensor and which type.
        .ok_or_else(|| CompareError::UnsupportedDtype { path: path.to_path_buf(), name: name.to_owned(), dtype: dtype.to_owned() })?;
    // The byte length must be exactly one value per element of the shape.
    if Some(raw.len()) != count.checked_mul(value_size) {
        // Wrong length means the header and the data do not agree.
        return Err(header_error(path, &format!("tensor {name} has {} bytes, which does not fit shape {shape:?}", raw.len())));
    }
    // Build the tensor, with the raw bytes turned into f32 values.
    Ok(Tensor { name: name.to_owned(), shape, values: decode(raw, dtype) })
}

// How many bytes one value of this type takes. None means the type is not supported.
fn bytes_per_value(dtype: &str) -> Option<usize> {
    // Pick the size for this type.
    match dtype {
        // float32 is 4 bytes.
        "F32" => Some(4),
        // float16 is 2 bytes.
        "F16" => Some(2),
        // int64 is 8 bytes. Token ids are int64 in PyTorch.
        "I64" => Some(8),
        // Any other type is not supported.
        _ => None,
    }
}

// Turn raw little-endian bytes into f32 values. The caller already checked the dtype and the length.
fn decode(raw: &[u8], dtype: &str) -> Vec<f32> {
    // F16: every 2 bytes are one float16, which we widen to f32.
    if dtype == "F16" {
        // Read 2 bytes at a time, lowest byte first.
        return raw.chunks_exact(2).map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))).collect();
    }
    // I64: every 8 bytes are one whole number. Token ids are below 2^24, so f32 holds them exactly.
    if dtype == "I64" {
        return raw.chunks_exact(8).map(|chunk| i64::from_le_bytes(chunk.try_into().expect("chunks_exact gives 8 bytes")) as f32).collect();
    }
    // F32: every 4 bytes are one f32, lowest byte first.
    raw.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
}

// Read a JSON array of whole numbers, such as [1, 1280, 8, 8]. Returns None if it is not one.
fn usize_list(value: &Value) -> Option<Vec<usize>> {
    // It must be an array.
    value
        // Get the array.
        .as_array()?
        // Go through each item.
        .iter()
        // Each item must be a whole number that fits in usize.
        .map(|item| item.as_u64().and_then(|number| usize::try_from(number).ok()))
        // Fail if any item fails.
        .collect()
}

// Build a header error for this file.
fn header_error(path: &Path, reason: &str) -> CompareError {
    // Keep the path, so the message says which file is broken.
    CompareError::Header { path: path.to_path_buf(), reason: reason.to_owned() }
}
