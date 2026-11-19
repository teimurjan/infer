// output = input * 2 - 1. TAESD makes values in [0, 1]. diffusers maps them to [-1, 1], the range of the full VAE.
// The input can have more channels than the output. The kernel reads only the first `count` vec4s, which are the first channels.

// The sizes. Rust writes them once, before the dispatch.
struct Shape {
    // How many output vec4 values to write.
    count: u32,
}

// The sizes.
@group(0) @binding(0) var<uniform> shape: Shape;
// The last convolution output, with a 4th channel of zeros at the end.
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
// The image: 3 channels.
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE in kernels.rs must match 256.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The index of our vec4.
    let index = id.x;
    // Invocations past the end do nothing.
    if (index >= shape.count) {
        return;
    }
    // Scale and shift each of the 4 values.
    output[index] = input[index] * 2.0 - 1.0;
}
