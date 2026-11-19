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
const SHADER: &str = include_str!("concat.wgsl");
// Invocations per workgroup. It must match @workgroup_size(256) in concat.wgsl.
const WORKGROUP_SIZE: u32 = 256;

// Connect `output = cat(first, second)` along the channels. All three are [channels, height, width] with the same height and width.
pub(crate) fn bind_concat(gpu: &Gpu, first: &SlotTensor, second: &SlotTensor, output: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // The two inputs.
    let ([first_channels, height, width], [second_channels, _, _]) = (first.dims, second.dims);
    // The output has the channels of both, at the same size.
    let expected = [first_channels + second_channels, height, width];
    if output.dims != expected || second.dims[1..] != first.dims[1..] {
        return Err(EngineError::PlanMismatch { tensor: output.name.to_string(), planned: output.dims, actual: expected });
    }
    // The values of the first input and of the output.
    let first_count = first_channels * height * width;
    let count = expected[0] * height * width;
    // The slots must hold the three tensors.
    check_buffer_holds("concat first", first.buffer, first_count as usize)?;
    check_buffer_holds("concat second", second.buffer, (count - first_count) as usize)?;
    check_buffer_holds("concat output", output.buffer, count as usize)?;
    // The field order of the Shape struct in concat.wgsl.
    let uniform = gpu.uniform_buffer("concat", &[first_count, count]);
    // One invocation per output value. Bindings: sizes, first, second, output.
    // It reads both inputs, which together have the size of the output, and writes the output.
    let work = Work::f32_values("concat", 2 * u64::from(count));
    let dispatch = Dispatch::new(gpu, "concat", SHADER, &[&uniform, first.buffer, second.buffer, output.buffer], [count.div_ceil(WORKGROUP_SIZE), 1, 1], work)?;
    // Keep the sizes with the kernel.
    Ok(BoundKernel::new(vec![dispatch], vec![uniform]))
}
