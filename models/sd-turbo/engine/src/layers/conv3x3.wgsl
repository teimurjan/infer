// output[o][y][x] = bias[o] + sum over c, ky, kx of weight[o][c][ky][kx] * input[c][y * stride + ky - 1][x * stride + kx - 1].
// This is nn.Conv2d with a 3x3 kernel and padding 1: pixels outside the image count as 0.
// Tensors are in PyTorch's NCHW layout with batch 1.

// The sizes of this layer. Rust writes them once, before the dispatch.
struct Shape {
    // Input channels, for example 320.
    in_channels: u32,
    // Input height in pixels.
    height: u32,
    // Input width in pixels.
    width: u32,
    // Output channels.
    out_channels: u32,
    // Output height: (height - 1) / stride + 1.
    out_height: u32,
    // Output width: (width - 1) / stride + 1.
    out_width: u32,
    // Step between output pixels, measured in input pixels: 1 keeps the size, 2 halves it.
    stride: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input, one value at a time: [in_channels][height][width].
@group(0) @binding(1) var<storage, read> input: array<f32>;
// The weights in PyTorch's layout: [out_channels][in_channels][3][3].
@group(0) @binding(2) var<storage, read> weight: array<f32>;
// One bias per output channel, 4 per vec4: [out_channels / 4].
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
// The output. One vec4 holds 4 neighbor pixels of a row: [out_channels][out_height][out_width / 4].
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

// One input value, or 0 for a position outside the image (the padding).
fn load_input(channel: u32, y: i32, x: i32) -> f32 {
    // Outside the image on any side.
    if (y < 0 || x < 0 || y >= i32(shape.height) || x >= i32(shape.width)) {
        return 0.0;
    }
    // Inside: row y of this channel, then column x.
    return input[(channel * shape.height + u32(y)) * shape.width + u32(x)];
}

// 8 x 8 = 64 invocations per workgroup. Rust's WORKGROUP_SIDE must match 8.
@compute @workgroup_size(8, 8, 1)
// Each invocation computes a block of 4 output channels x 4 neighbor pixels of one row = 16 outputs.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // x picks the 4 pixels of the row. The first one is 4 * id.x.
    let x0 = id.x * 4u;
    // y picks the output row.
    let y = id.y;
    // z picks the 4 output channels. The first one is 4 * id.z.
    let channel0 = id.z * 4u;
    // The last workgroups can stick out past the end. Those invocations do nothing.
    if (x0 >= shape.out_width || y >= shape.out_height || channel0 >= shape.out_channels) {
        return;
    }
    // Weights of one output channel: in_channels * 3 * 3. The next output channel starts this far away.
    let weights_per_channel = shape.in_channels * 9u;
    // The input step between our 4 pixels: 1 for stride 1, 2 for stride 2.
    let step = i32(shape.stride);
    // sum0 holds our 4 pixels of output channel channel0, sum1 of channel0 + 1, and so on.
    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);
    var sum2 = vec4<f32>(0.0);
    var sum3 = vec4<f32>(0.0);
    // Walk over every input channel.
    for (var c = 0u; c < shape.in_channels; c++) {
        // Walk over the 3 kernel rows.
        for (var ky = 0u; ky < 3u; ky++) {
            // The input row. The - 1 is the padding: kernel row 0 looks one row above.
            let input_y = i32(y * shape.stride + ky) - 1;
            // Walk over the 3 kernel columns.
            for (var kx = 0u; kx < 3u; kx++) {
                // The input column under our first pixel.
                let input_x = i32(x0 * shape.stride + kx) - 1;
                // The input value under each of our 4 pixels.
                let pixels = vec4<f32>(
                    load_input(c, input_y, input_x),
                    load_input(c, input_y, input_x + step),
                    load_input(c, input_y, input_x + 2 * step),
                    load_input(c, input_y, input_x + 3 * step),
                );
                // Where weight[channel0][c][ky][kx] is.
                let w = channel0 * weights_per_channel + c * 9u + ky * 3u + kx;
                // 4 input loads and 4 weight loads feed 16 multiply-adds.
                sum0 += weight[w] * pixels;
                sum1 += weight[w + weights_per_channel] * pixels;
                sum2 += weight[w + 2u * weights_per_channel] * pixels;
                sum3 += weight[w + 3u * weights_per_channel] * pixels;
            }
        }
    }
    // The 4 biases of our 4 output channels.
    let b = bias[id.z];
    // vec4s in one output channel.
    let plane = shape.out_height * shape.out_width / 4u;
    // Our 4 pixels as a vec4 index inside one output channel.
    let position = (y * shape.out_width + x0) / 4u;
    // Add the bias and write 4 pixels per channel with one vec4 store.
    output[(channel0 + 0u) * plane + position] = sum0 + b.x;
    output[(channel0 + 1u) * plane + position] = sum1 + b.y;
    output[(channel0 + 2u) * plane + position] = sum2 + b.z;
    output[(channel0 + 3u) * plane + position] = sum3 + b.w;
}
