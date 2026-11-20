// The sizes of a linear layer.
use crate::layers::LinearShape;

// Output columns that one invocation computes. It must match the shader.
const COLUMNS: u32 = 4;
// Input rows that one invocation computes. It must match the 8 rows of linear.wgsl.
const ROWS: u32 = 8;
// Invocations per workgroup along x (column blocks) and y (row blocks). It must match @workgroup_size in linear.wgsl.
const WORKGROUP: [u32; 2] = [8, 8];

// The workgroups that cover every output of the packed linear layer.
pub(super) fn linear_workgroups(shape: &LinearShape) -> [u32; 3] {
    // Column blocks, and row blocks. The last row block can be short, so the rows round up.
    let invocations = [shape.out_features / COLUMNS, shape.rows.div_ceil(ROWS)];
    // Round each axis up, so the last invocations also get a workgroup. The shader skips the ones past the end.
    [invocations[0].div_ceil(WORKGROUP[0]), invocations[1].div_ceil(WORKGROUP[1]), 1]
}
