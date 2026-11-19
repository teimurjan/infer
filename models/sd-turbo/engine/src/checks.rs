// Our error type.
use crate::error::EngineError;

// Size of one f32 value in bytes.
const F32_BYTES: u64 = 4;

// Fail if a tensor does not have the length that the shape needs. A wrong length would make a kernel read the wrong values.
pub(crate) fn check_length(tensor: &'static str, values: &[f32], expected: usize) -> Result<(), EngineError> {
    // Equal lengths are fine.
    if values.len() == expected {
        return Ok(());
    }
    // Say which tensor and both lengths.
    Err(EngineError::TensorLength { tensor, actual: values.len(), expected })
}

// A size that only has to be 1 or more.
pub(crate) fn positive_dimension(dimension: &'static str, value: u32) -> Result<u32, EngineError> {
    // 0 is not allowed.
    if value == 0 {
        return Err(EngineError::ZeroDimension { dimension });
    }
    // The value is fine.
    Ok(value)
}

// Fail if a GPU buffer does not hold exactly `expected` f32 values.
pub(crate) fn check_buffer(tensor: &'static str, buffer: &wgpu::Buffer, expected: usize) -> Result<(), EngineError> {
    // The number of f32 values that fit in the buffer. A buffer is never larger than memory, so this fits in usize.
    let actual = (buffer.size() / F32_BYTES) as usize;
    // Equal lengths are fine.
    if actual == expected {
        return Ok(());
    }
    // Say which tensor and both lengths.
    Err(EngineError::TensorLength { tensor, actual, expected })
}

// Fail if a GPU buffer holds fewer than `expected` f32 values. A slot of the activation plan can hold more, because it is
// as large as its largest tensor. The kernels take their sizes from a uniform, so the extra values are never read.
pub(crate) fn check_buffer_holds(tensor: &'static str, buffer: &wgpu::Buffer, expected: usize) -> Result<(), EngineError> {
    // The number of f32 values that fit in the buffer.
    let actual = (buffer.size() / F32_BYTES) as usize;
    // Enough room is fine.
    if actual >= expected {
        return Ok(());
    }
    // Say which tensor and both lengths.
    Err(EngineError::TensorLength { tensor, actual, expected })
}
