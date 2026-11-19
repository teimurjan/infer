// LayerNorm over the channels of each row: PyTorch's nn.LayerNorm with affine weights, on [rows][channels].
// One workgroup handles one row: it finds the mean, then the variance, then writes the normalized row.

// The sizes of this kernel. Rust writes them once, before the first dispatch.
struct Shape {
    // Values in one row.
    channels: u32,
    // The small number added to the variance (eps), as the bits of an f32.
    eps_bits: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input rows. The slot can be longer. We read only our rows.
@group(0) @binding(1) var<storage, read> input: array<f32>;
// The scale of each channel (LayerNorm weight).
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
// The shift of each channel (LayerNorm bias).
@group(0) @binding(3) var<storage, read> beta: array<f32>;
// The output rows, the same size as the input.
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Invocations per workgroup. Rust's THREADS must match 64.
const THREADS: u32 = 64u;
// One partial sum per invocation, shared by the workgroup.
var<workgroup> partial: array<f32, 64>;

// Add up `value` over the 64 invocations and give the sum to every invocation.
fn workgroup_sum(local: u32, value: f32) -> f32 {
    // Each invocation puts its own sum into the shared list.
    partial[local] = value;
    // Wait until every invocation has written.
    workgroupBarrier();
    // Halve the list each round: entry i adds entry i + stride. 6 rounds for 64 entries.
    for (var stride = THREADS / 2u; stride > 0u; stride = stride / 2u) {
        // Only the first half of the remaining entries adds.
        if (local < stride) {
            partial[local] += partial[local + stride];
        }
        // Wait until this round is done.
        workgroupBarrier();
    }
    // Entry 0 holds the sum.
    let total = partial[0];
    // Wait until every invocation has read it, so the next call can write the list again.
    workgroupBarrier();
    // The sum, for every invocation.
    return total;
}

// 64 invocations per workgroup, one workgroup per row.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) workgroup: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    // The first value of this row.
    let start = workgroup.x * shape.channels;
    // Pass 1: each invocation adds every 64th value of the row.
    var sum = 0.0;
    for (var index = local; index < shape.channels; index += THREADS) {
        sum += input[start + index];
    }
    // The mean of the row.
    let mean = workgroup_sum(local, sum) / f32(shape.channels);
    // Pass 2: the sum of squared distances from the mean.
    var squares = 0.0;
    for (var index = local; index < shape.channels; index += THREADS) {
        // The distance of this value from the mean.
        let distance = input[start + index] - mean;
        squares += distance * distance;
    }
    // The variance, divided by the count as PyTorch does.
    let variance = workgroup_sum(local, squares) / f32(shape.channels);
    // 1 / sqrt(variance + eps).
    let scale = inverseSqrt(variance + bitcast<f32>(shape.eps_bits));
    // Pass 3: normalize, scale and shift each value.
    for (var index = local; index < shape.channels; index += THREADS) {
        output[start + index] = (input[start + index] - mean) * scale * gamma[index] + beta[index];
    }
}
