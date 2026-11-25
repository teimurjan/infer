// The gated short convolution of LFM3 (Lfm3ShortConv), between its two linear layers:
// in_proj gives B, C and x for each token. Then h = B * x, then a causal depthwise convolution of width 7 over the
// tokens, then y = C * conv(h). Token t sees tokens t - 6 to t. Tokens before 0 count as 0.

// The sizes. Rust writes them once, before the first run.
struct Shape {
    // Tokens.
    rows: u32,
    // Channels, 2048 in LFM3. One row of the input holds B, C and x: 3 x width values.
    width: u32,
    // The convolution width, 7 in LFM3.
    kernel: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The output of in_proj: [rows][B | C | x], each part width values.
@group(0) @binding(1) var<storage, read> bcx: array<f32>;
// The convolution weights: [width][kernel], one row of taps per channel.
@group(0) @binding(2) var<storage, read> taps: array<f32>;
// The output: [rows][width], the input of out_proj.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Invocations per workgroup. Rust's WORKGROUP_SIZE in short_conv.rs must match.
@compute @workgroup_size(256)
// Each invocation writes one value: token t, channel c.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The value of this invocation.
    let index = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (index >= shape.rows * shape.width) {
        return;
    }
    let token = index / shape.width;
    let channel = index % shape.width;
    // The values in one input row: B, C and x.
    let row_width = 3u * shape.width;
    // Tap k of the convolution reads token t - (kernel - 1) + k, as nn.Conv1d with padding kernel - 1 and the first rows kept.
    var sum = 0.0;
    for (var tap = 0u; tap < shape.kernel; tap++) {
        // The token under this tap, as a signed number, because it can be before the first token.
        let source = i32(token) - i32(shape.kernel - 1u) + i32(tap);
        // Tokens before the first one are the zero padding.
        if (source >= 0) {
            let row = u32(source) * row_width;
            // h = B * x of that token and channel.
            let gated = bcx[row + channel] * bcx[row + 2u * shape.width + channel];
            sum += taps[channel * shape.kernel + tap] * gated;
        }
    }
    // y = C * conv(h), with the C of this token.
    output[index] = bcx[token * row_width + shape.width + channel] * sum;
}
