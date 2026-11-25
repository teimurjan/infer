// Our error type.
use crate::error::EngineError;

// The shared layer helpers.
use super::{BLOCK, LayerPlan, Lengths, block_dimension, product, workgroups_for};

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("conv1x1.wgsl");

// The sizes of one 1x1 convolution with batch 1: input [in_channels, height, width], output [out_channels, height, width].
pub struct Conv1x1Shape {
    // Input channels.
    pub in_channels: u32,
    // Output channels.
    pub out_channels: u32,
    // Height in pixels. A 1x1 kernel keeps the size.
    pub height: u32,
    // Width in pixels.
    pub width: u32,
}

// Check the sizes and plan the 1x1 convolution kernel.
pub(super) fn plan_conv1x1(shape: &Conv1x1Shape) -> Result<LayerPlan, EngineError> {
    // The kernel reads input channels 4 at a time.
    let in_channels = block_dimension("in_channels", shape.in_channels)?;
    // Each invocation takes 4 output channels.
    let out_channels = block_dimension("out_channels", shape.out_channels)?;
    // Each invocation takes 4 pixels, so the pixel count must be a multiple of 4.
    let pixels = block_dimension("height * width", shape.height * shape.width)?;
    // The length of each tensor. The weight is [out_channels, in_channels, 1, 1].
    let lengths = Lengths {
        input: product(&[in_channels, pixels]),
        weight: product(&[out_channels, in_channels]),
        bias: product(&[out_channels]),
        output: product(&[out_channels, pixels]),
    };
    // x covers the pixel blocks and y the output channel blocks, as in conv1x1.wgsl.
    let workgroups = [workgroups_for(pixels / BLOCK), workgroups_for(out_channels / BLOCK), 1];
    // The field order of the Shape struct in conv1x1.wgsl.
    Ok(LayerPlan { label: "conv1x1", shader: SHADER, sizes: vec![in_channels, out_channels, pixels], lengths, workgroups })
}
