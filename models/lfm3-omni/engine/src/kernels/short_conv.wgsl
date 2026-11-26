// The second part of the gated short convolution of LFM3 (Lfm3ShortConv): a causal depthwise convolution of width 7
// over the history of h = B * x, then y = C * conv(h). The token at position p sees positions p - 6 to p.
// Positions before 0 count as 0. conv_gate.wgsl wrote h of the new tokens into the history first.

// The sizes. Rust writes them once, before the first run.
struct Shape {
    // New tokens in this run.
    rows: u32,
    // Channels, 2048. One row of bcx holds B, C and x: 3 x width values.
    width: u32,
    // The convolution width, 7.
    kernel: u32,
}

// Where this run starts. The CPU writes it before each run (memory.rs).
struct Step {
    start: u32,
    keys: u32,
}

@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<uniform> step: Step;
// The output of in_proj for the new tokens, for C: [rows][B | C | x].
@group(0) @binding(2) var<storage, read> bcx: array<f32>;
// h of every position so far: [capacity][width].
@group(0) @binding(3) var<storage, read> history: array<f32>;
// The convolution weights: [width][kernel].
@group(0) @binding(4) var<storage, read> taps: array<f32>;
// The output: [rows][width], the input of out_proj.
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

// Invocations per workgroup. Rust's WORKGROUP_SIZE in kernels/mod.rs must match.
@compute @workgroup_size(256)
// Each invocation writes one value: new token t, channel c.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= shape.rows * shape.width) {
        return;
    }
    let token = index / shape.width;
    let channel = index % shape.width;
    // The position of this token.
    let position = step.start + token;
    // Tap k reads position p - (kernel - 1) + k, as nn.Conv1d with padding kernel - 1 and the first rows kept.
    var sum = 0.0;
    for (var tap = 0u; tap < shape.kernel; tap++) {
        // The position under this tap, as a signed number, because it can be before position 0.
        let source = i32(position) - i32(shape.kernel - 1u) + i32(tap);
        if (source >= 0) {
            sum += taps[channel * shape.kernel + tap] * history[u32(source) * shape.width + channel];
        }
    }
    // y = C * conv(h), with the C of this token.
    output[index] = bcx[token * 3u * shape.width + shape.width + channel] * sum;
}
