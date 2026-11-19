// output[column] = bias[column] + sum over k of f(input[k]) * weight[column][k]. This is PyTorch's nn.Linear on one row.
// f is SiLU when shape.silu_input is 1, and nothing when it is 0. The UNet applies SiLU to the time embedding before
// each linear layer that reads it, so this kernel does it while it reads, and no SiLU tensor is needed.

// The sizes of this layer. Rust writes them once, before the first dispatch.
struct Shape {
    // Values in the input row, a multiple of 4.
    in_features: u32,
    // Values in the output row.
    out_features: u32,
    // 1: apply SiLU to each input value first. 0: use the input as it is.
    silu_input: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input row, 4 values per vec4: [in_features / 4]. The slot can be longer. We read only in_features values.
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
// The weights in PyTorch's layout, one row per output column: [out_features][in_features / 4].
@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;
// One bias per output column: [out_features].
@group(0) @binding(3) var<storage, read> bias: array<f32>;
// The output row: [out_features]. The slot can be longer. We write only out_features values.
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// SiLU of 4 values at once: x times sigmoid(x).
fn silu(x: vec4<f32>) -> vec4<f32> {
    return x / (vec4<f32>(1.0) + exp(-x));
}

// 64 invocations per workgroup. Rust's WORKGROUP_SIZE must match 64.
@compute @workgroup_size(64)
// Each invocation computes one output column: one dot product of the input row with one weight row.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The output column of this invocation.
    let column = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (column >= shape.out_features) {
        return;
    }
    // vec4s in one row.
    let vectors = shape.in_features / 4u;
    // The first vec4 of this column's weight row.
    let row = column * vectors;
    // The running sum of 4 partial sums, one per vec4 lane.
    var sum = vec4<f32>(0.0);
    // Walk along the row, 4 values at a time.
    for (var k = 0u; k < vectors; k++) {
        // 4 input values.
        var x = input[k];
        // The flag is the same for every invocation, so this branch costs almost nothing.
        if (shape.silu_input == 1u) {
            x = silu(x);
        }
        // Multiply 4 input values by 4 weights and add them up per lane.
        sum += x * weight[row + k];
    }
    // Add the 4 lanes and the bias.
    output[column] = bias[column] + sum.x + sum.y + sum.z + sum.w;
}
