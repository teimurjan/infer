// values = max(values + skip, 0), in place. This ends a TAESD block: the convolution output plus the block input, then ReLU.

// The sizes. Rust writes them once, before the dispatch.
struct Shape {
    // How many vec4 values to process.
    count: u32,
}

// The sizes.
@group(0) @binding(0) var<uniform> shape: Shape;
// The block input, which skips over the three convolutions.
@group(0) @binding(1) var<storage, read> skip: array<vec4<f32>>;
// The output of the last convolution. The result goes back into it.
@group(0) @binding(2) var<storage, read_write> values: array<vec4<f32>>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE in kernels.rs must match 256.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The index of our vec4.
    let index = id.x;
    // Invocations past the end do nothing.
    if (index >= shape.count) {
        return;
    }
    // Add, then ReLU, on each of the 4 values.
    values[index] = max(values[index] + skip[index], vec4<f32>(0.0));
}
