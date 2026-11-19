// Size of one f32 value in bytes.
const F32_BYTES: usize = 4;

// Turn raw bytes into f32 values. Each group of 4 bytes is one value, lowest byte first (little-endian).
// The GPU, the Mac and WASM all store f32 this way. Extra bytes at the end (fewer than 4) are ignored.
pub fn f32_from_le_bytes(bytes: &[u8]) -> Vec<f32> {
    // Split the bytes into groups of exactly 4.
    bytes
        .chunks_exact(F32_BYTES)
        // Build one f32 from each group of 4 bytes.
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        // Put all values into one list.
        .collect()
}

// Turn f32 values into raw bytes, 4 bytes per value, lowest byte first.
pub fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    // Each value becomes 4 bytes, and all bytes go into one flat list.
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}
