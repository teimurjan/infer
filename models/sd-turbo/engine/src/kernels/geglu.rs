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

// The WGSL source, copied into the program when it compiles: erf first, then the kernel.
const SHADER: &str = concat!(include_str!("erf.wgsl"), include_str!("geglu.wgsl"));
// Invocations per workgroup. It must match @workgroup_size(256) in geglu.wgsl.
const WORKGROUP_SIZE: u32 = 256;

// Connect GEGLU from `projected` [rows, 2 * inner, 1] to `output` [rows, inner, 1].
pub(crate) fn bind_geglu(gpu: &Gpu, projected: &SlotTensor, output: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // The output size: rows of inner values.
    let [rows, inner, _] = output.dims;
    // The projection must be twice as wide, with the same rows.
    if projected.dims != [rows, 2 * inner, 1] {
        return Err(EngineError::PlanMismatch { tensor: projected.name.to_string(), planned: projected.dims, actual: [rows, 2 * inner, 1] });
    }
    // The values to write.
    let count = rows * inner;
    // The slots must hold both tensors.
    check_buffer_holds("geglu input", projected.buffer, 2 * count as usize)?;
    check_buffer_holds("geglu output", output.buffer, count as usize)?;
    // The field order of the Shape struct in geglu.wgsl.
    let uniform = gpu.uniform_buffer("geglu", &[rows, inner]);
    // One invocation per output value. Bindings: sizes, projection, output.
    // It reads the projection, 2 values per output, and writes the output.
    let work = Work::f32_values("geglu", 3 * u64::from(count));
    let dispatch = Dispatch::new(gpu, "geglu", SHADER, &[&uniform, projected.buffer, output.buffer], [count.div_ceil(WORKGROUP_SIZE), 1, 1], work)?;
    // Keep the sizes with the kernel.
    Ok(BoundKernel::new(vec![dispatch], vec![uniform]))
}
