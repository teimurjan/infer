// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The weight buffers of the slot kernels.
use crate::weights::{WeightBuffer, WeightBuffers};

// Get one weight of shape [channels], such as the scale or the shift of a norm, as f32 values on the GPU.
pub(super) fn channel_weight<'a>(gpu: &Gpu, weights: &'a impl WeightBuffers, name: &str, channels: u32) -> Result<WeightBuffer<'a>, EngineError> {
    // The f32 values and the shape.
    let tensor = weights.float(gpu, name)?;
    // One value per channel.
    if tensor.shape != [channels as usize] {
        return Err(EngineError::WeightShape { name: name.to_string(), shape: tensor.shape, expected: format!("[{channels}]") });
    }
    // The buffer.
    Ok(tensor.buffer)
}
