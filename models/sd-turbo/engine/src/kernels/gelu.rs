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
const SHADER: &str = concat!(include_str!("erf.wgsl"), include_str!("gelu.wgsl"));
// Invocations per workgroup. It must match @workgroup_size(256) in gelu.wgsl.
const WORKGROUP_SIZE: u32 = 256;

// Connect GELU in place on `values`.
pub(crate) fn bind_gelu(gpu: &Gpu, values: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // Every value of the tensor.
    let count = values.dims.iter().product::<u32>();
    // The slot must hold the tensor.
    check_buffer_holds("gelu values", values.buffer, count as usize)?;
    // The field order of the Shape struct in gelu.wgsl.
    let uniform = gpu.uniform_buffer("gelu", &[count]);
    // One invocation per value. Bindings: sizes, values.
    // In place: it reads every value and writes it back.
    let work = Work::f32_values("gelu", 2 * u64::from(count));
    let dispatch = Dispatch::new(gpu, "gelu", SHADER, &[&uniform, values.buffer], [count.div_ceil(WORKGROUP_SIZE), 1, 1], work)?;
    // Keep the sizes with the kernel.
    Ok(BoundKernel::new(vec![dispatch], vec![uniform]))
}
