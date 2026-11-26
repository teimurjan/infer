// output[row][column] = bias[column] + sum over groups g of scale[column][g] * sum over k in g of input[row][k] * level[column][k].
// level is the packed signed integer. This is nn.Linear with weight = level * scale.
// A group is a run of neighbor values of a weight row that share one scale. SD-Turbo has one group per row, LFM3 one per 32 values.
// The scale is the same for a whole group, so it is applied once to the sum of the group, not once per product.

// The sizes of this layer. Rust writes them once, before the dispatch.
struct Shape {
    // Input rows, for example 4096 tokens.
    rows: u32,
    // Values in one input row, for example 320.
    in_features: u32,
    // Values in one output row, for example 320.
    out_features: u32,
    // Bits per packed weight: 4 or 8.
    bits: u32,
    // u32 words in one packed weight row.
    row_words: u32,
    // vec4s of input that share one scale: in_features / 4 for one scale per row, 8 for one scale per 32 values.
    group_vectors: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input, row by row. One vec4 holds 4 neighbor values of a row: [rows][in_features / 4].
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
// The packed weights, one row of row_words words per output column.
@group(0) @binding(2) var<storage, read> weight: array<u32>;
// The float16 scales, [out_features][groups], two per word, the even one in the low 16 bits.
@group(0) @binding(3) var<storage, read> scale: array<u32>;
// One bias per output column, 4 per vec4: [out_features / 4].
@group(0) @binding(4) var<storage, read> bias: array<vec4<f32>>;
// The output, row by row: [rows][out_features / 4].
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

// Weights 4k to 4k + 3 of one packed row, as f32 integers.
fn weight_vec4(column: u32, k: u32) -> vec4<f32> {
    // The bit where weight 4k starts inside the row.
    let first_bit = k * 4u * shape.bits;
    // 4 weights take 16 or 32 bits, so they always sit in one word. bitcast keeps the bits and makes it signed.
    let word = bitcast<i32>(weight[column * shape.row_words + first_bit / 32u]);
    // Where weight 4k starts inside that word.
    let offset = first_bit % 32u;
    // extractBits on an i32 copies the top bit of the field upward, so 0b1111 at 4 bits becomes -1.
    return vec4<f32>(
        f32(extractBits(word, offset, shape.bits)),
        f32(extractBits(word, offset + shape.bits, shape.bits)),
        f32(extractBits(word, offset + 2u * shape.bits, shape.bits)),
        f32(extractBits(word, offset + 3u * shape.bits, shape.bits)),
    );
}

// The float16 scale of group `group` of one output column, as f32.
fn group_scale(column: u32, group: u32, groups: u32) -> f32 {
    // The place of this scale among all scales.
    let index = column * groups + group;
    // unpack2x16float gives the low half first, so scale 2j is .x and scale 2j + 1 is .y.
    return unpack2x16float(scale[index / 2u])[index % 2u];
}

// 8 x 8 = 64 invocations per workgroup. Rust's WORKGROUP in packed/linear.rs must match.
@compute @workgroup_size(8, 8, 1)
// Each invocation computes a block of 8 rows x 4 columns = 32 outputs.
// With 4 rows, each unpacked weight fed 4 rows. With 8 rows, the same unpacking feeds twice the multiply-adds.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // x picks the 4 output columns. As a vec4 index, column block c covers columns 4c to 4c + 3.
    let column_block = id.x;
    // y picks the 8 rows. The first one is 8 * id.y. Rust's ROWS in packed/linear.rs must match 8.
    let row0 = id.y * 8u;
    // vec4s in one input row.
    let in_vectors = shape.in_features / 4u;
    // vec4s in one output row.
    let out_vectors = shape.out_features / 4u;
    // The last workgroups can stick out past the end. Those invocations do nothing.
    if (column_block >= out_vectors || row0 >= shape.rows) {
        return;
    }
    // The first of our 4 output columns.
    let column0 = column_block * 4u;
    // The last input row. The row count does not have to be a multiple of 8: the cross attention reads 77 text rows.
    let last_row = shape.rows - 1u;
    // The first vec4 of each of our 8 input rows. A row past the last one reads the last row instead, and its output is not written.
    // The rows are picked once here, so the loop below has no extra work.
    let start0 = row0 * in_vectors;
    let start1 = min(row0 + 1u, last_row) * in_vectors;
    let start2 = min(row0 + 2u, last_row) * in_vectors;
    let start3 = min(row0 + 3u, last_row) * in_vectors;
    let start4 = min(row0 + 4u, last_row) * in_vectors;
    let start5 = min(row0 + 5u, last_row) * in_vectors;
    let start6 = min(row0 + 6u, last_row) * in_vectors;
    let start7 = min(row0 + 7u, last_row) * in_vectors;
    // The groups of one row.
    let groups = in_vectors / shape.group_vectors;
    // total0 holds the 4 scaled outputs of row row0, total1 of row row0 + 1, and so on.
    var total0 = vec4<f32>(0.0);
    var total1 = vec4<f32>(0.0);
    var total2 = vec4<f32>(0.0);
    var total3 = vec4<f32>(0.0);
    var total4 = vec4<f32>(0.0);
    var total5 = vec4<f32>(0.0);
    var total6 = vec4<f32>(0.0);
    var total7 = vec4<f32>(0.0);
    // One group at a time: its unscaled sums, then its scale.
    for (var group = 0u; group < groups; group++) {
        // sum0 holds the 4 unscaled sums of this group for row row0, sum1 for row row0 + 1, and so on. 32 running sums in total.
        var sum0 = vec4<f32>(0.0);
        var sum1 = vec4<f32>(0.0);
        var sum2 = vec4<f32>(0.0);
        var sum3 = vec4<f32>(0.0);
        var sum4 = vec4<f32>(0.0);
        var sum5 = vec4<f32>(0.0);
        var sum6 = vec4<f32>(0.0);
        var sum7 = vec4<f32>(0.0);
        // Walk along the values of this group, 4 at a time.
        let first = group * shape.group_vectors;
        for (var k = first; k < first + shape.group_vectors; k++) {
            // 4 integers from each of our 4 weight rows, unpacked once for all 8 input rows.
            let w0 = weight_vec4(column0 + 0u, k);
            let w1 = weight_vec4(column0 + 1u, k);
            let w2 = weight_vec4(column0 + 2u, k);
            let w3 = weight_vec4(column0 + 3u, k);
            // 4 values from each of our 8 input rows, and 4 dot products of 4 values for each row: 128 multiply-adds.
            let x0 = input[start0 + k];
            sum0 += vec4<f32>(dot(x0, w0), dot(x0, w1), dot(x0, w2), dot(x0, w3));
            let x1 = input[start1 + k];
            sum1 += vec4<f32>(dot(x1, w0), dot(x1, w1), dot(x1, w2), dot(x1, w3));
            let x2 = input[start2 + k];
            sum2 += vec4<f32>(dot(x2, w0), dot(x2, w1), dot(x2, w2), dot(x2, w3));
            let x3 = input[start3 + k];
            sum3 += vec4<f32>(dot(x3, w0), dot(x3, w1), dot(x3, w2), dot(x3, w3));
            let x4 = input[start4 + k];
            sum4 += vec4<f32>(dot(x4, w0), dot(x4, w1), dot(x4, w2), dot(x4, w3));
            let x5 = input[start5 + k];
            sum5 += vec4<f32>(dot(x5, w0), dot(x5, w1), dot(x5, w2), dot(x5, w3));
            let x6 = input[start6 + k];
            sum6 += vec4<f32>(dot(x6, w0), dot(x6, w1), dot(x6, w2), dot(x6, w3));
            let x7 = input[start7 + k];
            sum7 += vec4<f32>(dot(x7, w0), dot(x7, w1), dot(x7, w2), dot(x7, w3));
        }
        // The scales of this group in our 4 columns.
        let s = vec4<f32>(group_scale(column0, group, groups), group_scale(column0 + 1u, group, groups), group_scale(column0 + 2u, group, groups), group_scale(column0 + 3u, group, groups));
        // Scale each sum of this group once, and add it to the total of its row.
        total0 += sum0 * s;
        total1 += sum1 * s;
        total2 += sum2 * s;
        total3 += sum3 * s;
        total4 += sum4 * s;
        total5 += sum5 * s;
        total6 += sum6 * s;
        total7 += sum7 * s;
    }
    // The 4 biases of our 4 columns.
    let b = bias[column_block];
    // Add the bias, and write 4 outputs per row with one vec4 store. Rows past the last one are not written.
    output[row0 * out_vectors + column_block] = total0 + b;
    if (row0 + 1u <= last_row) {
        output[(row0 + 1u) * out_vectors + column_block] = total1 + b;
    }
    if (row0 + 2u <= last_row) {
        output[(row0 + 2u) * out_vectors + column_block] = total2 + b;
    }
    if (row0 + 3u <= last_row) {
        output[(row0 + 3u) * out_vectors + column_block] = total3 + b;
    }
    if (row0 + 4u <= last_row) {
        output[(row0 + 4u) * out_vectors + column_block] = total4 + b;
    }
    if (row0 + 5u <= last_row) {
        output[(row0 + 5u) * out_vectors + column_block] = total5 + b;
    }
    if (row0 + 6u <= last_row) {
        output[(row0 + 6u) * out_vectors + column_block] = total6 + b;
    }
    if (row0 + 7u <= last_row) {
        output[(row0 + 7u) * out_vectors + column_block] = total7 + b;
    }
}
