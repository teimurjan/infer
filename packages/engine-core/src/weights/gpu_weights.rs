// HashMap finds a tensor by its name.
use std::collections::HashMap;
// Display prints the summary line.
use std::fmt;

// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;
// A buffer that counts its bytes.
use crate::memory::GpuBuffer;

// One weight tensor that lives in its own GPU buffer.
pub struct GpuTensor {
    // The buffer. Its size is `bytes` rounded up to a multiple of 4.
    pub buffer: GpuBuffer,
    // The size of each dimension, in PyTorch's order.
    pub shape: Vec<usize>,
    // "float32", "float16", "int8" or "int4", as in the index.
    pub dtype: String,
    // The bytes that the file stores for this tensor, without the padding of the buffer.
    pub bytes: u64,
}

// Every tensor of one weight file, on the GPU. Dropping it frees all its buffers.
pub struct GpuWeights {
    // Tensor name to tensor.
    tensors: HashMap<String, GpuTensor>,
}

impl GpuWeights {
    // Keep the tensors of a finished upload.
    pub fn new(tensors: HashMap<String, GpuTensor>) -> Self {
        Self { tensors }
    }

    // The tensor with this name.
    pub fn tensor(&self, name: &str) -> Result<&GpuTensor, EngineError> {
        // A missing name usually means a typo in the engine or the wrong weight file.
        self.tensors.get(name).ok_or_else(|| EngineError::MissingWeight(name.to_string()))
    }

    // The number of tensors, which is also the number of GPU buffers.
    pub fn count(&self) -> usize {
        self.tensors.len()
    }

    // The bytes of all buffers together.
    pub fn buffer_bytes(&self) -> u64 {
        self.tensors.values().map(|tensor| tensor.buffer.size()).sum()
    }

    // The size of the largest buffer. It must stay under the iPhone limit of about 1073 MB.
    pub fn largest_buffer(&self) -> u64 {
        self.tensors.values().map(|tensor| tensor.buffer.size()).max().unwrap_or(0)
    }

    // Copy one tensor back to the CPU, with the padding of its buffer. For tests and debugging.
    pub async fn read_back(&self, gpu: &Gpu, name: &str) -> Result<Vec<u8>, EngineError> {
        // Find the tensor, then read its whole buffer.
        gpu.read_bytes(&self.tensor(name)?.buffer).await
    }
}

// One line, for example "968 tensors in 968 GPU buffers, 700016128 bytes, largest buffer 52428800 bytes".
impl fmt::Display for GpuWeights {
    // Write the line.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{} tensors in {} GPU buffers, {} bytes, largest buffer {} bytes", self.count(), self.count(), self.buffer_bytes(), self.largest_buffer())
    }
}
