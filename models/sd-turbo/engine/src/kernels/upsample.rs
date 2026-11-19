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
// The nearest 2x upsample shader of TAESD. It takes its sizes from a uniform, so it works on slots.
use crate::taesd::UPSAMPLE_SHADER;

// Invocations per workgroup. It must match @workgroup_size(256) in taesd/upsample.wgsl.
const WORKGROUP_SIZE: u32 = 256;
// One invocation writes 4 output values.
const VEC4: u32 = 4;

// Connect nearest 2x upsampling (F.interpolate with scale_factor=2, mode="nearest") from `input` to `output`.
pub(crate) fn bind_upsample(gpu: &Gpu, input: &SlotTensor, output: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // The input size.
    let [channels, height, width] = input.dims;
    // Twice as high and twice as wide.
    let expected = [channels, height * 2, width * 2];
    if output.dims != expected {
        return Err(EngineError::PlanMismatch { tensor: output.name.to_string(), planned: output.dims, actual: expected });
    }
    // The shader reads 2 input pixels per output vec4, so the input width must be even.
    if !width.is_multiple_of(2) {
        return Err(EngineError::BadDimension { dimension: "upsample input width", value: width });
    }
    // The output values.
    let count = expected.iter().product::<u32>();
    // The slots must hold both tensors.
    check_buffer_holds("upsample input", input.buffer, (count / 4) as usize)?;
    check_buffer_holds("upsample output", output.buffer, count as usize)?;
    // The field order of the Shape struct: output vec4s, input width.
    let uniform = gpu.uniform_buffer("upsample", &[count / VEC4, width]);
    // One invocation per output vec4. Bindings: sizes, input, output.
    // It reads the input, a quarter of the output, and writes the output.
    let work = Work::f32_values("upsample", u64::from(count) + u64::from(count / 4));
    let dispatch = Dispatch::new(gpu, "upsample", UPSAMPLE_SHADER, &[&uniform, input.buffer, output.buffer], [(count / VEC4).div_ceil(WORKGROUP_SIZE), 1, 1], work)?;
    // Keep the sizes with the kernel.
    Ok(BoundKernel::new(vec![dispatch], vec![uniform]))
}
