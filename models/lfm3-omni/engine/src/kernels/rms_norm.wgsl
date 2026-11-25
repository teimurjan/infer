// out[r][c] = weight[c] * x[r][c] / sqrt(mean over c of x[r][c]^2 + eps), the RMSNorm of LFM3, one row at a time.
// With accumulate = 1 the result is added to the output instead of replacing it. The layer uses that for the residual add
// after its sandwich norms: hidden += norm(branch).

// The sizes. Rust writes them once, before the first run.
struct Shape {
    // Rows, one per token.
    rows: u32,
    // Values in one row, 2048 in LFM3.
    width: u32,
    // 1: add to the output. 0: replace it.
    accumulate: u32,
    // eps, as the bits of an f32, because the uniform buffer holds u32 values.
    eps_bits: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input rows: [rows][width].
@group(0) @binding(1) var<storage, read> input: array<f32>;
// One scale per value of a row: [width].
@group(0) @binding(2) var<storage, read> weight: array<f32>;
// The output rows, in the same layout as the input.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Invocations per workgroup. Rust's WORKGROUP_SIZE in rms_norm.rs must match.
const LANES: u32 = 256u;
// The partial sums of squares of the lanes, so they can add them up together.
var<workgroup> partial: array<f32, 256>;

// One workgroup per row.
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    // The row of this workgroup. The grid has exactly one workgroup per row.
    let row = group.x;
    // Where the row starts.
    let start = row * shape.width;
    // Lane l adds the squares of values l, l + 256, l + 512, and so on.
    var sum = 0.0;
    for (var column = lane; column < shape.width; column += LANES) {
        let value = input[start + column];
        sum += value * value;
    }
    partial[lane] = sum;
    // Halve the number of partial sums until one is left: 256, 128, ..., 1.
    for (var step = LANES / 2u; step > 0u; step /= 2u) {
        // Wait until every lane has written its partial sum of the last round.
        workgroupBarrier();
        if (lane < step) {
            partial[lane] += partial[lane + step];
        }
    }
    // Wait for the last round.
    workgroupBarrier();
    // 1 / sqrt(mean square + eps), as torch.rsqrt(variance + eps).
    let scale = inverseSqrt(partial[0] / f32(shape.width) + bitcast<f32>(shape.eps_bits));
    // Each lane writes its own values of the row.
    for (var column = lane; column < shape.width; column += LANES) {
        // PyTorch first scales x, then multiplies by the weight.
        let normed = weight[column] * (input[start + column] * scale);
        // Add to the residual stream, or write a new tensor.
        if (shape.accumulate == 1u) {
            output[start + column] += normed;
        } else {
            output[start + column] = normed;
        }
    }
}
