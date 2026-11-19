// Nearest upsampling by 2, as nn.Upsample(scale_factor=2, mode="nearest"): every input pixel becomes a 2 x 2 square.
// Layout [channels][height][width]. One invocation writes one output vec4: 4 neighbor pixels of one output row.

// The sizes. Rust writes them once, before the dispatch.
struct Shape {
    // How many output vec4 values to write.
    count: u32,
    // The input width in pixels. The output is twice as wide.
    in_width: u32,
}

// The sizes.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input, one value at a time.
@group(0) @binding(1) var<storage, read> input: array<f32>;
// The output, 4 neighbor pixels per vec4.
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE in kernels.rs must match 256.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The index of our output vec4.
    let index = id.x;
    // Invocations past the end do nothing.
    if (index >= shape.count) {
        return;
    }
    // An output row is 2 * in_width pixels, so it holds in_width / 2 vec4s.
    let vec4s_per_row = shape.in_width / 2u;
    // The output row, counted over all channels: channel * out_height + y.
    let out_row = index / vec4s_per_row;
    // Our vec4 inside that row.
    let column = index % vec4s_per_row;
    // out_height = 2 * in_height, so channel * out_height + y, halved, is channel * in_height + y / 2: the input row.
    let in_row = out_row / 2u;
    // Our 4 output pixels start at 4 * column. They come from the 2 input pixels at 2 * column and 2 * column + 1.
    let first = in_row * shape.in_width + column * 2u;
    // Each input pixel fills 2 output pixels.
    let left = input[first];
    let right = input[first + 1u];
    // Write the 4 output pixels.
    output[index] = vec4<f32>(left, left, right, right);
}
