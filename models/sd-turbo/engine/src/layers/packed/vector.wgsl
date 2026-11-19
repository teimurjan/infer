// output[column] = bias[column] + scale[column] * sum over k of f(input[k]) * level[column][k]. This is nn.Linear on one row,
// with packed weights: weight = level * scale. f is SiLU when shape.silu_input is 1, and nothing when it is 0.
// The UNet uses it for the time embedding and for time_emb_proj, which apply SiLU to the time embedding first.

// The sizes of this layer. Rust writes them once, before the first dispatch.
struct Shape {
    // Values in the input row, a multiple of 4.
    in_features: u32,
    // Values in the output row.
    out_features: u32,
    // 1: apply SiLU to each input value first. 0: use the input as it is.
    silu_input: u32,
    // Bits per packed weight: 4 or 8.
    bits: u32,
    // u32 words in one packed weight row.
    row_words: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input row, 4 values per vec4: [in_features / 4]. The slot can be longer.
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
// The packed weights, one row of row_words words per output column.
@group(0) @binding(2) var<storage, read> weight: array<u32>;
// Two float16 scales per word, the even column in the low 16 bits: [out_features / 2].
@group(0) @binding(3) var<storage, read> scale: array<u32>;
// One bias per output column: [out_features].
@group(0) @binding(4) var<storage, read> bias: array<f32>;
// The output row: [out_features]. The slot can be longer.
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

// SiLU of 4 values at once: x times sigmoid(x).
fn silu(x: vec4<f32>) -> vec4<f32> {
    return x / (vec4<f32>(1.0) + exp(-x));
}

// Weights 4k to 4k + 3 of one packed row, as f32 integers. The same unpacking as packed/linear.wgsl.
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

// 64 invocations per workgroup. Rust's WORKGROUP_SIZE must match 64.
@compute @workgroup_size(64)
// Each invocation computes one output column.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The output column of this invocation.
    let column = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (column >= shape.out_features) {
        return;
    }
    // The running sum of 4 partial sums, one per vec4 lane.
    var sum = vec4<f32>(0.0);
    // Walk along the row, 4 values at a time.
    for (var k = 0u; k < shape.in_features / 4u; k++) {
        // 4 input values.
        var x = input[k];
        // The flag is the same for every invocation, so this branch costs almost nothing.
        if (shape.silu_input == 1u) {
            x = silu(x);
        }
        // Multiply 4 input values by 4 integers and add them up per lane.
        sum += x * weight_vec4(column, k);
    }
    // The float16 scale of this column. unpack2x16float gives the low half first.
    let column_scale = unpack2x16float(scale[column / 2u])[column % 2u];
    // Scale the sum once, then add the bias.
    output[column] = bias[column] + (sum.x + sum.y + sum.z + sum.w) * column_scale;
}
