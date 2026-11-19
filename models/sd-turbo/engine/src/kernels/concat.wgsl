// torch.cat along the channels: output = first, then second. The tensors are [channels][pixels],
// so the output is the values of `first`, followed by the values of `second`.

// The sizes of this kernel. Rust writes them once, before the first dispatch.
struct Shape {
    // Values in `first`.
    first_count: u32,
    // Values in the output: first and second together.
    count: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The first tensor, the up tensor of the UNet.
@group(0) @binding(1) var<storage, read> first: array<f32>;
// The second tensor, the skip tensor.
@group(0) @binding(2) var<storage, read> second: array<f32>;
// The joined tensor.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE must match 256.
@compute @workgroup_size(256)
// Each invocation copies one value.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The output value of this invocation.
    let index = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (index >= shape.count) {
        return;
    }
    // The first part comes from `first`, the rest from `second`.
    if (index < shape.first_count) {
        output[index] = first[index];
    } else {
        output[index] = second[index - shape.first_count];
    }
}
