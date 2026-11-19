// GroupNorm, then SiLU if the flag is set. This is PyTorch's nn.GroupNorm with affine weights.
// The tensor is [channels][pixels]. A group is channels_per_group neighbor channels, so its values are one block in memory.
// One workgroup handles one group: it finds the mean, then the variance, then writes every normalized value.
// With the transpose flag, the output is [pixels][channels]: one row of channels per token, as the transformer reads it.

// The sizes of this kernel. Rust writes them once, before the first dispatch.
struct Shape {
    // Channels in one group, for example 320 / 32 = 10.
    channels_per_group: u32,
    // Pixels per channel, height x width.
    pixels: u32,
    // 1: apply SiLU after the norm. 0: no SiLU.
    silu: u32,
    // The small number added to the variance (eps), as the bits of an f32. A uniform struct of u32 values is simpler to fill.
    eps_bits: u32,
    // 1: write [pixels][channels]. 0: write [channels][pixels], the layout of the input.
    transpose: u32,
    // All channels, for the transposed layout.
    channels: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input tensor. The slot can be longer. We read only our groups.
@group(0) @binding(1) var<storage, read> input: array<f32>;
// The scale of each channel (GroupNorm weight).
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
// The shift of each channel (GroupNorm bias).
@group(0) @binding(3) var<storage, read> beta: array<f32>;
// The output tensor, the same size as the input, in the layout that the flag picks.
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Invocations per workgroup. Rust's THREADS must match 256.
const THREADS: u32 = 256u;
// One partial sum per invocation, shared by the workgroup.
var<workgroup> partial: array<f32, 256>;

// Add up `value` over the 256 invocations and give the sum to every invocation.
fn workgroup_sum(local: u32, value: f32) -> f32 {
    // Each invocation puts its own sum into the shared list.
    partial[local] = value;
    // Wait until every invocation has written.
    workgroupBarrier();
    // Halve the list each round: entry i adds entry i + stride. 8 rounds for 256 entries.
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

// 256 invocations per workgroup, one workgroup per group.
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    // The group of this workgroup.
    let group = workgroup.x;
    // Values in one group.
    let size = shape.channels_per_group * shape.pixels;
    // The first value of the group.
    let start = group * size;
    // Pass 1: each invocation adds every 256th value of the group.
    var sum = 0.0;
    for (var index = local; index < size; index += THREADS) {
        sum += input[start + index];
    }
    // The mean of the group.
    let mean = workgroup_sum(local, sum) / f32(size);
    // Pass 2: the sum of squared distances from the mean. Two passes are more exact than sum and sum of squares in one.
    var squares = 0.0;
    for (var index = local; index < size; index += THREADS) {
        // The distance of this value from the mean.
        let distance = input[start + index] - mean;
        squares += distance * distance;
    }
    // The variance, divided by the count as PyTorch does (not by count - 1).
    let variance = workgroup_sum(local, squares) / f32(size);
    // 1 / sqrt(variance + eps).
    let scale = inverseSqrt(variance + bitcast<f32>(shape.eps_bits));
    // Pass 3: normalize, scale and shift each value, then SiLU.
    for (var index = local; index < size; index += THREADS) {
        // The channel of this value. Each channel has `pixels` values.
        let channel = group * shape.channels_per_group + index / shape.pixels;
        // The normalized value with the channel's scale and shift.
        var value = (input[start + index] - mean) * scale * gamma[channel] + beta[channel];
        // SiLU: x times sigmoid(x). The flag is the same for every invocation.
        if (shape.silu == 1u) {
            value = value / (1.0 + exp(-value));
        }
        // The pixel of this value, for the transposed layout.
        let pixel = index % shape.pixels;
        // The same position as the input, or row `pixel`, column `channel`.
        let position = select(start + index, pixel * shape.channels + channel, shape.transpose == 1u);
        // Write it.
        output[position] = value;
    }
}
