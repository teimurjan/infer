// output = tanh(input / 3) * 3. TAESD squeezes every latent value into (-3, 3) before its first convolution.
// One invocation handles one vec4: 4 neighbor values.

// The sizes. Rust writes them once, before the dispatch.
struct Shape {
    // How many vec4 values to process.
    count: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The latent, 4 values per vec4.
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
// The clamped latent, same layout.
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE in kernels.rs must match 256.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // We dispatch along x only, so x is the index of our vec4.
    let index = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (index >= shape.count) {
        return;
    }
    // tanh works on each of the 4 values.
    output[index] = tanh(input[index] / 3.0) * 3.0;
}
