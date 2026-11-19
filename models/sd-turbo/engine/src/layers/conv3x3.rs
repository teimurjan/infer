// Our error type.
use crate::error::EngineError;

// The shared check for sizes that only have to be 1 or more.
use crate::checks::positive_dimension;

// The shared layer helpers.
use super::{BLOCK, LayerPlan, Lengths, block_dimension, product, workgroups_for};

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("conv3x3.wgsl");

// The sizes of one 3x3 convolution with padding 1 and batch 1: input [in_channels, height, width].
pub struct Conv3x3Shape {
    // Input channels.
    pub in_channels: u32,
    // Output channels.
    pub out_channels: u32,
    // Input height in pixels.
    pub height: u32,
    // Input width in pixels.
    pub width: u32,
    // 1 keeps the size. 2 halves it, as in the UNet downsamplers.
    pub stride: u32,
}

// Check the sizes and plan the 3x3 convolution kernel.
pub(super) fn plan_conv3x3(shape: &Conv3x3Shape) -> Result<LayerPlan, EngineError> {
    // A stride of 0 would never move.
    if shape.stride == 0 {
        return Err(EngineError::ZeroStride);
    }
    // The kernel walks input channels one by one, so any count of 1 or more works.
    let in_channels = positive_dimension("in_channels", shape.in_channels)?;
    // Input height of 1 or more.
    let height = positive_dimension("height", shape.height)?;
    // Input width of 1 or more.
    let width = positive_dimension("width", shape.width)?;
    // Each invocation takes 4 output channels.
    let out_channels = block_dimension("out_channels", shape.out_channels)?;
    // The output height for this stride.
    let out_height = conv3x3_output_size(height, shape.stride);
    // Each invocation takes 4 neighbor pixels of one row, so the output width must be a multiple of 4.
    let out_width = block_dimension("output width", conv3x3_output_size(width, shape.stride))?;
    // The length of each tensor. The weight is [out_channels, in_channels, 3, 3].
    let lengths = Lengths {
        input: product(&[in_channels, height, width]),
        weight: product(&[out_channels, in_channels, 3, 3]),
        bias: product(&[out_channels]),
        output: product(&[out_channels, out_height, out_width]),
    };
    // x covers pixel blocks, y output rows, z output channel blocks, as in conv3x3.wgsl. z has 1 invocation per workgroup.
    let workgroups = [workgroups_for(out_width / BLOCK), workgroups_for(out_height), out_channels / BLOCK];
    // The field order of the Shape struct in conv3x3.wgsl.
    let sizes = vec![in_channels, height, width, out_channels, out_height, out_width, shape.stride];
    // The plan, without any data.
    Ok(LayerPlan { label: "conv3x3", shader: SHADER, sizes, lengths, workgroups })
}

// The output height or width of a 3x3 convolution with padding 1. The size must be 1 or more.
pub(crate) fn conv3x3_output_size(size: u32, stride: u32) -> u32 {
    // The PyTorch formula with kernel 3 and padding 1: (size + 2 - 3) / stride + 1.
    (size - 1) / stride + 1
}
