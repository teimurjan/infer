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
const SHADER: &str = include_str!("add.wgsl");
// Invocations per workgroup. It must match @workgroup_size(256) in add.wgsl.
const WORKGROUP_SIZE: u32 = 256;

// Connect `values += addend` in place. The addend has the size of `values` (a residual add),
// or the size [channels, 1, 1] (one value per channel, such as the time vector).
pub(crate) fn bind_add(gpu: &Gpu, values: &SlotTensor, addend: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // The size of the tensor that changes.
    let [channels, height, width] = values.dims;
    // Pick the kind of add from the sizes.
    let per_channel = if addend.dims == values.dims {
        // One addend value per value.
        false
    } else if addend.dims == [channels, 1, 1] {
        // One addend value per channel.
        true
    } else {
        // Any other size has no meaning here.
        return Err(EngineError::PlanMismatch { tensor: addend.name.to_string(), planned: addend.dims, actual: values.dims });
    };
    // The values to change.
    let count = channels * height * width;
    // The slots must hold both tensors.
    check_buffer_holds("add values", values.buffer, count as usize)?;
    check_buffer_holds("add addend", addend.buffer, if per_channel { channels as usize } else { count as usize })?;
    // The field order of the Shape struct in add.wgsl.
    let uniform = gpu.uniform_buffer("add", &[count, height * width, u32::from(per_channel)]);
    // One invocation per value. Bindings: sizes, values, addend.
    // In place: it reads the values and the addend, and writes the values.
    let work = Work::f32_values("add", 2 * u64::from(count) + if per_channel { u64::from(channels) } else { u64::from(count) });
    let dispatch = Dispatch::new(gpu, "add", SHADER, &[&uniform, values.buffer, addend.buffer], [count.div_ceil(WORKGROUP_SIZE), 1, 1], work)?;
    // Keep the sizes with the kernel.
    Ok(BoundKernel::new(vec![dispatch], vec![uniform]))
}
