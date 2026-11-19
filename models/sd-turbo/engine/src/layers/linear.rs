// Our error type.
use crate::error::EngineError;
// The check for a size that only has to be 1 or more.
use crate::checks::positive_dimension;

// The shared layer helpers.
use super::{BLOCK, LayerPlan, Lengths, block_dimension, product, workgroups_for};

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("linear.wgsl");

// The sizes of one nn.Linear call: input [rows, in_features], weight [out_features, in_features], output [rows, out_features].
pub struct LinearShape {
    // Input rows, for example 4096 tokens.
    pub rows: u32,
    // Values in one input row.
    pub in_features: u32,
    // Values in one output row.
    pub out_features: u32,
}

// Check the sizes and plan the linear kernel.
pub(super) fn plan_linear(shape: &LinearShape) -> Result<LayerPlan, EngineError> {
    // Each invocation takes 4 rows. The last block of rows can be short, so any count of 1 or more works.
    let rows = positive_dimension("rows", shape.rows)?;
    // The kernel reads rows 4 values at a time.
    let in_features = block_dimension("in_features", shape.in_features)?;
    // Each invocation takes 4 columns.
    let out_features = block_dimension("out_features", shape.out_features)?;
    // The length of each tensor.
    let lengths = Lengths {
        input: product(&[rows, in_features]),
        weight: product(&[out_features, in_features]),
        bias: product(&[out_features]),
        output: product(&[rows, out_features]),
    };
    // x covers the column blocks and y the row blocks, as in linear.wgsl.
    // A short last block of rows still needs its invocation, so round up.
    let workgroups = [workgroups_for(out_features / BLOCK), workgroups_for(rows.div_ceil(BLOCK)), 1];
    // The field order of the Shape struct in linear.wgsl.
    Ok(LayerPlan { label: "linear", shader: SHADER, sizes: vec![rows, in_features, out_features], lengths, workgroups })
}
