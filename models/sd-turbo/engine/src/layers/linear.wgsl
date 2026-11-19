// output[row][column] = bias[column] + sum over k of input[row][k] * weight[column][k]. This is PyTorch's nn.Linear.

// The sizes of this layer. Rust writes them once, before the dispatch.
struct Shape {
    // Input rows, for example 4096 tokens.
    rows: u32,
    // Values in one input row, for example 320.
    in_features: u32,
    // Values in one output row, for example 320.
    out_features: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input, row by row. One vec4 holds 4 neighbor values of a row: [rows][in_features / 4].
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
// The weights in PyTorch's layout, one row per output column: [out_features][in_features / 4].
@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;
// One bias per output column, 4 per vec4: [out_features / 4].
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
// The output, row by row: [rows][out_features / 4].
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

// 8 x 8 = 64 invocations per workgroup. Rust's WORKGROUP_SIDE must match 8.
@compute @workgroup_size(8, 8, 1)
// Each invocation computes a block of 4 rows x 4 columns = 16 outputs.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // x picks the 4 output columns. As a vec4 index, column block c covers columns 4c to 4c + 3.
    let column_block = id.x;
    // y picks the 4 rows. The first one is 4 * id.y.
    let row0 = id.y * 4u;
    // vec4s in one input row.
    let in_vectors = shape.in_features / 4u;
    // vec4s in one output row.
    let out_vectors = shape.out_features / 4u;
    // The last workgroups can stick out past the end. Those invocations do nothing.
    if (column_block >= out_vectors || row0 >= shape.rows) {
        return;
    }
    // The first of the 4 weight rows that we use.
    let column0 = column_block * 4u;
    // The last input row. The row count does not have to be a multiple of 4: the cross attention reads 77 text rows.
    let last_row = shape.rows - 1u;
    // The first vec4 of each of our 4 input rows. A row past the last one reads the last row instead, and its output is not written.
    // The rows are picked once here, so the loop below has no extra work.
    let start0 = row0 * in_vectors;
    let start1 = min(row0 + 1u, last_row) * in_vectors;
    let start2 = min(row0 + 2u, last_row) * in_vectors;
    let start3 = min(row0 + 3u, last_row) * in_vectors;
    // sum0 holds the 4 outputs of row row0, sum1 of row row0 + 1, and so on. 16 running sums in total.
    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);
    var sum2 = vec4<f32>(0.0);
    var sum3 = vec4<f32>(0.0);
    // Walk along the rows, 4 values at a time.
    for (var k = 0u; k < in_vectors; k++) {
        // 4 values from each of our 4 input rows.
        let x0 = input[start0 + k];
        let x1 = input[start1 + k];
        let x2 = input[start2 + k];
        let x3 = input[start3 + k];
        // 4 values from each of our 4 weight rows.
        let w0 = weight[(column0 + 0u) * in_vectors + k];
        let w1 = weight[(column0 + 1u) * in_vectors + k];
        let w2 = weight[(column0 + 2u) * in_vectors + k];
        let w3 = weight[(column0 + 3u) * in_vectors + k];
        // Each input vec4 meets all 4 weight vec4s, and each weight vec4 meets all 4 input vec4s.
        // So 8 loads feed 16 dot products of 4 values each: 64 multiply-adds.
        sum0 += vec4<f32>(dot(x0, w0), dot(x0, w1), dot(x0, w2), dot(x0, w3));
        sum1 += vec4<f32>(dot(x1, w0), dot(x1, w1), dot(x1, w2), dot(x1, w3));
        sum2 += vec4<f32>(dot(x2, w0), dot(x2, w1), dot(x2, w2), dot(x2, w3));
        sum3 += vec4<f32>(dot(x3, w0), dot(x3, w1), dot(x3, w2), dot(x3, w3));
    }
    // The 4 biases of our 4 columns.
    let b = bias[column_block];
    // Add the bias and write 4 outputs per row with one vec4 store. Rows past the last one are not written.
    output[row0 * out_vectors + column_block] = sum0 + b;
    if (row0 + 1u <= last_row) {
        output[(row0 + 1u) * out_vectors + column_block] = sum1 + b;
    }
    if (row0 + 2u <= last_row) {
        output[(row0 + 2u) * out_vectors + column_block] = sum2 + b;
    }
    if (row0 + 3u <= last_row) {
        output[(row0 + 3u) * out_vectors + column_block] = sum3 + b;
    }
}
