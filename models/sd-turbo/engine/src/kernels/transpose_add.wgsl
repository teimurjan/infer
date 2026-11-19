// output[channel][pixel] = rows[pixel][channel] + residual[channel][pixel].
// The end of a transformer: proj_out wrote token rows, the UNet wants [channels][pixels], and the transformer adds its input.

// The sizes of this kernel. Rust writes them once, before the first dispatch.
struct Shape {
    // Channels, the length of one token row.
    channels: u32,
    // Pixels, the number of token rows.
    pixels: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The token rows, [pixels][channels].
@group(0) @binding(1) var<storage, read> rows: array<f32>;
// The transformer input, [channels][pixels].
@group(0) @binding(2) var<storage, read> residual: array<f32>;
// The transformer output, [channels][pixels].
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE must match 256.
@compute @workgroup_size(256)
// Each invocation writes one output value.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The output value of this invocation, in [channels][pixels] order.
    let index = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (index >= shape.channels * shape.pixels) {
        return;
    }
    // Its channel and its pixel.
    let channel = index / shape.pixels;
    let pixel = index % shape.pixels;
    // The same value in token-row order, plus the input.
    output[index] = rows[pixel * shape.channels + channel] + residual[index];
}
