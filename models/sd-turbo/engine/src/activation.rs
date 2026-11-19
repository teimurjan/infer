// The kinds of GPU buffer that the memory counter keeps apart.
use crate::memory::MemoryCategory;
// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// Buffers that count their bytes.
use crate::memory::GpuBuffer;
// The shared tensor length check.
use crate::checks::check_length;

// One tensor between two layers, on the GPU, in PyTorch's layout [channels][height][width] with batch 1.
pub(crate) struct Activation {
    // The values.
    pub(crate) buffer: GpuBuffer,
    // Channels.
    pub(crate) channels: u32,
    // Height in pixels.
    pub(crate) height: u32,
    // Width in pixels.
    pub(crate) width: u32,
}

impl Activation {
    // Check that `values` has channels x height x width values, and copy them to the GPU.
    pub(crate) fn upload(gpu: &Gpu, label: &'static str, values: &[f32], [channels, height, width]: [u32; 3]) -> Result<Self, EngineError> {
        // The number of values that the shape needs.
        let length = channels as usize * height as usize * width as usize;
        // A wrong length would make the kernels read the wrong values.
        check_length(label, values, length)?;
        // The values as a GPU activation.
        Ok(Self { buffer: gpu.storage_buffer_from(label, values, MemoryCategory::Activations), channels, height, width })
    }

    // The number of values: channels x height x width. usize, so large tensors cannot overflow u32.
    pub(crate) fn value_count(&self) -> usize {
        self.channels as usize * self.height as usize * self.width as usize
    }
}
