// values[i] = GELU(values[i]), in place. GELU is the exact one of PyTorch: 0.5 * x * (1 + erf(x / sqrt(2))).
// erf comes from erf.wgsl, which Rust puts in front of this text. The text encoder MLP uses it after fc1.

// The sizes of this kernel. Rust writes them once, before the first dispatch.
struct Shape {
    // Values to change.
    count: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The tensor that changes in place. The slot can be longer. We change only `count` values.
@group(0) @binding(1) var<storage, read_write> values: array<f32>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE must match 256.
@compute @workgroup_size(256)
// Each invocation changes one value.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The value of this invocation.
    let index = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (index >= shape.count) {
        return;
    }
    // The value.
    let x = values[index];
    // 1 / sqrt(2).
    let inverse_root_two = 0.70710678118654752;
    // GELU of it, in place.
    values[index] = 0.5 * x * (1.0 + erf(x * inverse_root_two));
}
