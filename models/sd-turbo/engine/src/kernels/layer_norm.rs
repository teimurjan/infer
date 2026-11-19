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
// The weight buffers of the slot kernels.
use crate::weights::WeightBuffers;

// The reader of a scale or shift with one value per channel.
use super::weight::channel_weight;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("layer_norm.wgsl");
// The eps of every LayerNorm in the transformer blocks (norm_eps of BasicTransformerBlock).
const EPS: f32 = 1e-5;

// Read `<prefix>.weight` and `<prefix>.bias` and connect LayerNorm from `input` [rows, channels, 1] to `output` of the same size.
pub(crate) fn bind_layer_norm(gpu: &Gpu, weights: &impl WeightBuffers, prefix: &str, input: &SlotTensor, output: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // A token tensor: rows of channels.
    let [rows, channels, 1] = input.dims else {
        return Err(EngineError::PlanMismatch { tensor: input.name.to_string(), planned: input.dims, actual: [input.dims[0], input.dims[1], 1] });
    };
    // LayerNorm keeps the size.
    if output.dims != input.dims {
        return Err(EngineError::PlanMismatch { tensor: output.name.to_string(), planned: output.dims, actual: input.dims });
    }
    // The slots must hold the whole tensor.
    let values = rows as usize * channels as usize;
    check_buffer_holds("layer norm input", input.buffer, values)?;
    check_buffer_holds("layer norm output", output.buffer, values)?;
    // The scale and the shift.
    let gamma = channel_weight(gpu, weights, &format!("{prefix}.weight"), channels)?;
    let beta = channel_weight(gpu, weights, &format!("{prefix}.bias"), channels)?;
    // The field order of the Shape struct in layer_norm.wgsl.
    let uniform = gpu.uniform_buffer(prefix, &[channels, EPS.to_bits()]);
    // The input and the output once each, and the scale and the shift.
    let work = Work::f32_values("layer_norm", 2 * values as u64 + 2 * u64::from(channels));
    // One workgroup per row. Bindings: sizes, input, scale, shift, output.
    let dispatch = Dispatch::new(gpu, prefix, SHADER, &[&uniform, input.buffer, gamma.buffer(), beta.buffer(), output.buffer], [rows, 1, 1], work)?;
    // Keep the buffers with the kernel.
    Ok(BoundKernel::new(vec![dispatch], vec![uniform]).keeping(gamma.into_owned()).keeping(beta.into_owned()))
}
