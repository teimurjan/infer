// output[o][y][x] = bias[o] + scale[o] * sum over c, ky, kx of level[o][c][ky][kx] * input[c][y * stride + ky - 1][x * stride + kx - 1].
// level is the packed signed integer. This is nn.Conv2d with a 3x3 kernel, padding 1, and weight = level * scale.
// The scale is the same for a whole output channel, so it is applied once to the sum, not once per product.

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
    // Bits per packed weight: 4 or 8.
    bits: u32,
    // u32 words in one packed weight row. One row holds in_channels * 9 integers.
    row_words: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input, one value at a time: [in_channels][height][width].
@group(0) @binding(1) var<storage, read> input: array<f32>;
// The packed weights, one row of row_words words per output channel, in the order [c][ky][kx].
@group(0) @binding(2) var<storage, read> weight: array<u32>;
// Two float16 scales per word, the even channel in the low 16 bits: [out_channels / 2].
@group(0) @binding(3) var<storage, read> scale: array<u32>;
// One bias per output channel, 4 per vec4: [out_channels / 4].
@group(0) @binding(4) var<storage, read> bias: array<vec4<f32>>;
// The output. One vec4 holds 4 neighbor pixels of a row: [out_channels][out_height][out_width / 4].
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

// One input value, or 0 for a position outside the image (the padding).
fn load_input(channel: u32, y: i32, x: i32) -> f32 {
    // Outside the image on any side.
    if (y < 0 || x < 0 || y >= i32(shape.height) || x >= i32(shape.width)) {
        return 0.0;
    }
    // Inside: row y of this channel, then column x.
    return input[(channel * shape.height + u32(y)) * shape.width + u32(x)];
}

// Integer number `index` of one packed row, as f32.
fn weight_value(channel: u32, index: u32) -> f32 {
    // The bit where this integer starts inside the row.
    let first_bit = index * shape.bits;
    // 32 is a multiple of 4 and of 8, so an integer never crosses a word. bitcast keeps the bits and makes it signed.
    let word = bitcast<i32>(weight[channel * shape.row_words + first_bit / 32u]);
    // extractBits on an i32 copies the top bit of the field upward, so 0b1111 at 4 bits becomes -1.
    return f32(extractBits(word, first_bit % 32u, shape.bits));
}

// One input value of a row that starts at index `row`, or 0 for a column outside the image (the padding).
// The caller checks the row, so only the column is checked here.
fn load_column(row: u32, x: i32) -> f32 {
    // Left or right of the image.
    if (x < 0 || x >= i32(shape.width)) {
        return 0.0;
    }
    // Inside: column x of the row.
    return input[row + u32(x)];
}

// The integers at position `index` of the packed rows of output channels channel0 to channel0 + 3, as f32.
fn four_weights(channel0: u32, index: u32) -> vec4<f32> {
    // One integer per channel. Component x is channel0.
    return vec4<f32>(weight_value(channel0, index), weight_value(channel0 + 1u, index), weight_value(channel0 + 2u, index), weight_value(channel0 + 3u, index));
}

// The float16 scale of one output channel, as f32.
fn channel_scale(channel: u32) -> f32 {
    // unpack2x16float gives the low half first, so channel 2j is .x and channel 2j + 1 is .y.
    return unpack2x16float(scale[channel / 2u])[channel % 2u];
}

// 2 x 8 x 4 = 64 invocations per workgroup: 2 pixel blocks, 8 rows, 4 channel blocks. Rust's WORKGROUP in packed/conv3x3.rs must match.
// The 8 x 8 x 1 shape left half of the lanes idle on 16 x 16 images and three quarters on 8 x 8 images.
@compute @workgroup_size(2, 8, 4)
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
    // The input step between our 4 pixels: 1 for stride 1, 2 for stride 2.
    let step = i32(shape.stride);
    // sum0 holds our 4 pixels of output channel channel0, sum1 of channel0 + 1, and so on.
    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);
    var sum2 = vec4<f32>(0.0);
    var sum3 = vec4<f32>(0.0);
    // Stride 1 is 49 of the 52 UNet convolutions. Its fast path loads each input value once per input row.
    if (shape.stride == 1u) {
        // Walk over every input channel.
        for (var c = 0u; c < shape.in_channels; c++) {
            // Walk over the 3 kernel rows.
            for (var ky = 0u; ky < 3u; ky++) {
                // The input row. The - 1 is the padding: kernel row 0 looks one row above.
                let input_y = i32(y + ky) - 1;
                // A row above or below the image is all padding, so it adds nothing to any sum.
                if (input_y < 0 || input_y >= i32(shape.height)) {
                    continue;
                }
                // The index of the first value of this input row.
                let row = (c * shape.height + u32(input_y)) * shape.width;
                // The 3 kernel columns under our 4 pixels touch only 6 input columns: x0 - 1 to x0 + 4.
                let first = i32(x0) - 1;
                let v0 = load_column(row, first);
                let v1 = load_column(row, first + 1);
                let v2 = load_column(row, first + 2);
                let v3 = load_column(row, first + 3);
                let v4 = load_column(row, first + 4);
                let v5 = load_column(row, first + 5);
                // The position of [c][ky][0] inside each packed row.
                let index = c * 9u + ky * 3u;
                // Kernel column 0 sees columns x0 - 1 to x0 + 2, column 1 sees x0 to x0 + 3, column 2 sees x0 + 1 to x0 + 4.
                let left = vec4<f32>(v0, v1, v2, v3);
                let middle = vec4<f32>(v1, v2, v3, v4);
                let right = vec4<f32>(v2, v3, v4, v5);
                // The weights of our 4 output channels at each kernel column. Each component is one channel.
                let w_left = four_weights(channel0, index);
                let w_middle = four_weights(channel0, index + 1u);
                let w_right = four_weights(channel0, index + 2u);
                // 48 multiply-adds from 6 loads and 12 unpacked integers.
                sum0 += w_left.x * left + w_middle.x * middle + w_right.x * right;
                sum1 += w_left.y * left + w_middle.y * middle + w_right.y * right;
                sum2 += w_left.z * left + w_middle.z * middle + w_right.z * right;
                sum3 += w_left.w * left + w_middle.w * middle + w_right.w * right;
            }
        }
    } else {
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
                    // The position of [c][ky][kx] inside each packed row.
                    let index = c * 9u + ky * 3u + kx;
                    // 4 integers, unpacked here in the loop, feed 16 multiply-adds.
                    sum0 += weight_value(channel0 + 0u, index) * pixels;
                    sum1 += weight_value(channel0 + 1u, index) * pixels;
                    sum2 += weight_value(channel0 + 2u, index) * pixels;
                    sum3 += weight_value(channel0 + 3u, index) * pixels;
                }
            }
        }
    }
    // The 4 biases of our 4 output channels.
    let b = bias[id.z];
    // vec4s in one output channel.
    let plane = shape.out_height * shape.out_width / 4u;
    // Our 4 pixels as a vec4 index inside one output channel.
    let position = (y * shape.out_width + x0) / 4u;
    // Scale each channel sum, add the bias, and write 4 pixels per channel with one vec4 store.
    output[(channel0 + 0u) * plane + position] = sum0 * channel_scale(channel0 + 0u) + b.x;
    output[(channel0 + 1u) * plane + position] = sum1 * channel_scale(channel0 + 1u) + b.y;
    output[(channel0 + 2u) * plane + position] = sum2 * channel_scale(channel0 + 2u) + b.z;
    output[(channel0 + 3u) * plane + position] = sum3 * channel_scale(channel0 + 3u) + b.w;
}
