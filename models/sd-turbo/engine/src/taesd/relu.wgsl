// values = max(values, 0), in place. ReLU keeps positive values and turns negative values into 0.
// In place means no second buffer: at 512 x 512 x 64 channels, one buffer is 67 MB.

// The sizes. Rust writes them once, before the dispatch.
struct Shape {
    // How many vec4 values to process.
    count: u32,
}

// The sizes.
@group(0) @binding(0) var<uniform> shape: Shape;
// The values. The kernel reads each one and writes the result back to the same place.
@group(0) @binding(1) var<storage, read_write> values: array<vec4<f32>>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE in kernels.rs must match 256.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The index of our vec4.
    let index = id.x;
    // Invocations past the end do nothing.
    if (index >= shape.count) {
        return;
    }
    // max works on each of the 4 values.
    values[index] = max(values[index], vec4<f32>(0.0));
}
