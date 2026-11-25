// The output size formula of a 3x3 convolution.
use crate::layers::{Conv3x3Shape, conv3x3_output_size};

// Output pixels of one row that one invocation computes. It must match the shader.
const PIXELS: u32 = 4;
// Output channels that one invocation computes. It must match the shader.
const CHANNELS: u32 = 4;
// Invocations per workgroup along x (pixel blocks), y (rows) and z (channel blocks). It must match @workgroup_size in conv3x3.wgsl.
// 2 x 8 x 4: the smallest UNet image, 8 x 8, has 2 pixel blocks per row, so no lane of a workgroup is idle on any UNet image.
const WORKGROUP: [u32; 3] = [2, 8, 4];

// The workgroups that cover every output of the packed 3x3 convolution.
pub(super) fn conv3x3_workgroups(shape: &Conv3x3Shape) -> [u32; 3] {
    // Pixel blocks per row, rows, and channel blocks.
    let invocations = [conv3x3_output_size(shape.width, shape.stride) / PIXELS, conv3x3_output_size(shape.height, shape.stride), shape.out_channels / CHANNELS];
    // Round each axis up, so the last invocations also get a workgroup. The shader skips the ones past the end.
    [0, 1, 2].map(|axis| invocations[axis].div_ceil(WORKGROUP[axis]))
}
