// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// A kernel ready to record, and a kernel with its buffers.
use crate::dispatch::{BoundKernel, Dispatch, Work};
// The slot size check.
use crate::checks::check_buffer_holds;
// A tensor in a slot of the activation plan.
use crate::plan::SlotTensor;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("transpose_add.wgsl");
// Invocations per workgroup. It must match @workgroup_size(256) in transpose_add.wgsl.
const WORKGROUP_SIZE: u32 = 256;

// Connect `output = rows (as [channels][pixels]) + residual`. rows is [pixels, channels, 1]. residual and output are [channels, height, width].
pub(crate) fn bind_transpose_add(gpu: &Gpu, rows: &SlotTensor, residual: &SlotTensor, output: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // The image size.
    let [channels, height, width] = residual.dims;
    // Pixels per channel.
    let pixels = height * width;
    // The rows must be one per pixel, with every channel.
    if rows.dims != [pixels, channels, 1] {
        return Err(EngineError::PlanMismatch { tensor: rows.name.to_string(), planned: rows.dims, actual: [pixels, channels, 1] });
    }
    // The output has the size of the residual.
    if output.dims != residual.dims {
        return Err(EngineError::PlanMismatch { tensor: output.name.to_string(), planned: output.dims, actual: residual.dims });
    }
    // The values to write.
    let count = channels * pixels;
    // The slots must hold the three tensors.
    for (name, tensor) in [("transpose add rows", rows), ("transpose add residual", residual), ("transpose add output", output)] {
        check_buffer_holds(name, tensor.buffer, count as usize)?;
    }
    // The field order of the Shape struct in transpose_add.wgsl.
    let uniform = gpu.uniform_buffer("transpose add", &[channels, pixels]);
    // One invocation per output value. Bindings: sizes, rows, residual, output.
    // It reads the rows and the residual, and writes the output.
    let work = Work::f32_values("transpose add", 3 * u64::from(count));
    let dispatch = Dispatch::new(gpu, "transpose add", SHADER, &[&uniform, rows.buffer, residual.buffer, output.buffer], [count.div_ceil(WORKGROUP_SIZE), 1, 1], work)?;
    // Keep the sizes with the kernel.
    Ok(BoundKernel::new(vec![dispatch], vec![uniform]))
}
