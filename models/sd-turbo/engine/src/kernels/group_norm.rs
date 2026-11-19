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
const SHADER: &str = include_str!("group_norm.wgsl");
// The UNet uses 32 groups everywhere (norm_num_groups in unet/config.json).
const GROUPS: u32 = 32;

// The layout of the GroupNorm output.
#[derive(Clone, Copy)]
pub(crate) enum NormLayout {
    // [channels, height, width], as the input. The ResNet blocks use this.
    Image,
    // [pixels, channels, 1], one row per token. The transformers use this, because proj_in reads rows.
    Tokens,
}

// Read `<prefix>.weight` and `<prefix>.bias` (one value per channel) and connect GroupNorm from `input` to `output`.
// `silu` applies SiLU after the norm. `eps` is the number added to the variance. `layout` picks the output layout.
pub(crate) fn bind_group_norm_into(gpu: &Gpu, weights: &impl WeightBuffers, prefix: &str, input: &SlotTensor, silu: bool, eps: f32, layout: NormLayout, output: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // The input size.
    let [channels, height, width] = input.dims;
    // The output size of each layout.
    let expected = match layout {
        // The same size as the input.
        NormLayout::Image => input.dims,
        // One row of channels per pixel.
        NormLayout::Tokens => [height * width, channels, 1],
    };
    // The plan must have that size.
    if output.dims != expected {
        return Err(EngineError::PlanMismatch { tensor: output.name.to_string(), planned: output.dims, actual: expected });
    }
    // Every group must have the same number of channels.
    if channels == 0 || !channels.is_multiple_of(GROUPS) {
        return Err(EngineError::BadDimension { dimension: "group norm channels", value: channels });
    }
    // Pixels per channel.
    let pixels = height * width;
    // The slots must hold the whole tensor.
    let values = channels as usize * pixels as usize;
    check_buffer_holds("group norm input", input.buffer, values)?;
    check_buffer_holds("group norm output", output.buffer, values)?;
    // The scale and the shift, one value per channel.
    let gamma = channel_weight(gpu, weights, &format!("{prefix}.weight"), channels)?;
    let beta = channel_weight(gpu, weights, &format!("{prefix}.bias"), channels)?;
    // The field order of the Shape struct in group_norm.wgsl. eps goes in as the bits of its f32.
    let uniform = gpu.uniform_buffer(prefix, &[channels / GROUPS, pixels, u32::from(silu), eps.to_bits(), u32::from(matches!(layout, NormLayout::Tokens)), channels]);
    // The input and the output once each, and the scale and the shift.
    let work = Work::f32_values("group_norm", 2 * values as u64 + 2 * u64::from(channels));
    // One workgroup per group. Bindings: sizes, input, scale, shift, output.
    let dispatch = Dispatch::new(gpu, prefix, SHADER, &[&uniform, input.buffer, gamma.buffer(), beta.buffer(), output.buffer], [GROUPS, 1, 1], work)?;
    // Keep the buffers with the kernel.
    Ok(BoundKernel::new(vec![dispatch], vec![uniform]).keeping(gamma.into_owned()).keeping(beta.into_owned()))
}
