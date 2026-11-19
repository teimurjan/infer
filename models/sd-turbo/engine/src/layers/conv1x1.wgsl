// output[o][p] = bias[o] + sum over c of weight[o][c] * input[c][p]. This is nn.Conv2d with a 1x1 kernel.
// Tensors are in PyTorch's NCHW layout with batch 1, so p is one pixel: y * width + x.

// The sizes of this layer. Rust writes them once, before the dispatch.
struct Shape {
    // Input channels, for example 320.
    in_channels: u32,
    // Output channels, for example 640.
    out_channels: u32,
    // Pixels per channel: height * width, for example 32 * 32 = 1024.
    pixels: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The input, one channel after another. One vec4 holds 4 neighbor pixels: [in_channels][pixels / 4].
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
// The weights in PyTorch's layout [out_channels][in_channels][1][1], read 4 input channels at a time.
@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;
// One bias per output channel, 4 per vec4: [out_channels / 4].
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
// The output, one channel after another: [out_channels][pixels / 4].
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

// 8 x 8 = 64 invocations per workgroup. Rust's WORKGROUP_SIDE must match 8.
@compute @workgroup_size(8, 8, 1)
// Each invocation computes a block of 4 output channels x 4 pixels = 16 outputs.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // x picks the 4 pixels. As a vec4 index, pixel block p covers pixels 4p to 4p + 3.
    let pixel_block = id.x;
    // y picks the 4 output channels. The first one is 4 * id.y.
    let channel0 = id.y * 4u;
    // vec4s in one channel.
    let pixel_vectors = shape.pixels / 4u;
    // vec4s in one weight row.
    let in_vectors = shape.in_channels / 4u;
    // The last workgroups can stick out past the end. Those invocations do nothing.
    if (pixel_block >= pixel_vectors || channel0 >= shape.out_channels) {
        return;
    }
    // sum0 holds our 4 pixels of output channel channel0, sum1 of channel0 + 1, and so on.
    var sum0 = vec4<f32>(0.0);
    var sum1 = vec4<f32>(0.0);
    var sum2 = vec4<f32>(0.0);
    var sum3 = vec4<f32>(0.0);
    // Walk over the input channels, 4 at a time.
    for (var k = 0u; k < in_vectors; k++) {
        // The first of the 4 input channels in this step.
        let c = k * 4u;
        // Our 4 pixels in each of the 4 input channels.
        let x0 = input[(c + 0u) * pixel_vectors + pixel_block];
        let x1 = input[(c + 1u) * pixel_vectors + pixel_block];
        let x2 = input[(c + 2u) * pixel_vectors + pixel_block];
        let x3 = input[(c + 3u) * pixel_vectors + pixel_block];
        // For each of our 4 output channels, its 4 weights for these 4 input channels.
        let w0 = weight[(channel0 + 0u) * in_vectors + k];
        let w1 = weight[(channel0 + 1u) * in_vectors + k];
        let w2 = weight[(channel0 + 2u) * in_vectors + k];
        let w3 = weight[(channel0 + 3u) * in_vectors + k];
        // 8 loads feed 16 vec4 multiply-adds: 64 multiply-adds in total.
        sum0 += w0.x * x0 + w0.y * x1 + w0.z * x2 + w0.w * x3;
        sum1 += w1.x * x0 + w1.y * x1 + w1.z * x2 + w1.w * x3;
        sum2 += w2.x * x0 + w2.y * x1 + w2.z * x2 + w2.w * x3;
        sum3 += w3.x * x0 + w3.y * x1 + w3.z * x2 + w3.w * x3;
    }
    // The 4 biases of our 4 output channels.
    let b = bias[id.y];
    // Add the bias and write 4 pixels per channel with one vec4 store.
    output[(channel0 + 0u) * pixel_vectors + pixel_block] = sum0 + b.x;
    output[(channel0 + 1u) * pixel_vectors + pixel_block] = sum1 + b.y;
    output[(channel0 + 2u) * pixel_vectors + pixel_block] = sum2 + b.z;
    output[(channel0 + 3u) * pixel_vectors + pixel_block] = sum3 + b.w;
}
