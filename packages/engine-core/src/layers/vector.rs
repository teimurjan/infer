// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The size check for the output count.
use crate::checks::positive_dimension;

// The shared layer helpers.
use super::{LayerPlan, Lengths, bind_plan_into, block_dimension, product};
// A kernel with its buffers.
use crate::dispatch::BoundKernel;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("vector.wgsl");
// Invocations per workgroup. It must match @workgroup_size(64) in vector.wgsl.
const WORKGROUP_SIZE: u32 = 64;

// The sizes of nn.Linear on one row: input [in_features], weight [out_features, in_features], output [out_features].
pub struct VectorShape {
    // Values in the input row.
    pub in_features: u32,
    // Values in the output row.
    pub out_features: u32,
    // Apply SiLU to the input values as the kernel reads them.
    pub silu_input: bool,
}

// Check the sizes and plan the kernel. The packed one-row plan starts from it.
pub(super) fn plan_vector(shape: &VectorShape) -> Result<LayerPlan, EngineError> {
    // The kernel reads the input 4 values at a time.
    let in_features = block_dimension("in_features", shape.in_features)?;
    // One invocation per output value, so any count of 1 or more works.
    let out_features = positive_dimension("out_features", shape.out_features)?;
    // The length of each tensor. One row, so the input is in_features values.
    let lengths = Lengths { input: product(&[in_features]), weight: product(&[out_features, in_features]), bias: product(&[out_features]), output: product(&[out_features]) };
    // One invocation per output value, 64 per workgroup.
    let workgroups = [out_features.div_ceil(WORKGROUP_SIZE), 1, 1];
    // The field order of the Shape struct in vector.wgsl.
    Ok(LayerPlan { label: "vector linear", shader: SHADER, sizes: vec![in_features, out_features, u32::from(shape.silu_input)], lengths, workgroups })
}

// Connect the kernel from the `input` slot to the `output` slot. The slots can be longer than the tensors.
pub fn bind_vector_into(gpu: &Gpu, shape: &VectorShape, input: &wgpu::Buffer, weight: &wgpu::Buffer, bias: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // Check the sizes, plan the kernel, and connect the slots.
    bind_plan_into(gpu, &plan_vector(shape)?, input, weight, bias, output)
}
