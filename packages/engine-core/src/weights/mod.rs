// index.rs parses the .json index that convert_weights.py writes.
mod index;
// in_memory.rs reads tensors from a .bin file that is already in memory. The browser uses it after a fetch.
mod in_memory;
// file.rs reads tensors from a .bin file on disk. The browser has no files, so it is native only.
#[cfg(not(target_arch = "wasm32"))]
mod file;
// layout.rs finds where each tensor sits in the file and checks the buffer limits.
mod layout;
// cursor.rs cuts a file that arrives in chunks into 4-byte aligned writes, one tensor at a time.
mod cursor;
// upload.rs streams one weight file into GPU buffers with queue.write_buffer.
mod upload;
// gpu_weights.rs holds the tensors of one weight file on the GPU.
mod gpu_weights;
// buffers.rs gives the slot kernels their weights as GPU buffers, from a CPU reader or from GpuWeights.
mod buffers;
// bits.rs holds the bit width of a packed weight.
mod bits;

// Our error type.
use crate::error::EngineError;

// The parsed index.
pub use index::WeightIndex;
// The GPU bytes of one weight file.
pub use index::weight_bytes_on_gpu;
// The in-memory reader.
pub use in_memory::WeightBytes;
// The native reader.
#[cfg(not(target_arch = "wasm32"))]
pub use file::{WeightFile, load_to_gpu};
// The streaming loader and what it gives back.
pub use gpu_weights::{GpuTensor, GpuWeights};
pub use upload::WeightUpload;
// The weight buffers of the slot kernels.
pub use buffers::{Matrix, MatrixWeight, Uploading, WeightBuffer, WeightBuffers};
// The bit width of a packed weight.
pub use bits::WeightBits;

// One tensor as it is stored: its shape and its raw float32 bytes, lowest byte first.
pub struct WeightTensor {
    // The size of each dimension, in PyTorch's order.
    pub shape: Vec<usize>,
    // 4 bytes per value.
    pub bytes: Vec<u8>,
}

// Anything that gives the bytes of a weight tensor by name. The Mac reads a file. The browser will fetch.
pub trait WeightSource {
    // Read one tensor.
    fn read(&self, name: &str) -> Result<WeightTensor, EngineError>;
    // Read row `row` of one float32 tensor: everything after dimension 0, as raw bytes.
    fn read_row(&self, name: &str, row: usize) -> Result<Vec<u8>, EngineError>;
}
