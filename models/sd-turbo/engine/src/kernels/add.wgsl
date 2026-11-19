// values[i] += addend[j], in place. The addend has either the same size as `values` (j = i),
// or one value per channel (j = i / pixels), such as the time vector of a ResNet block.

// The sizes of this kernel. Rust writes them once, before the first dispatch.
struct Shape {
    // Values to change.
    count: u32,
    // Pixels per channel. Used only when per_channel is 1.
    pixels: u32,
    // 1: one addend value per channel. 0: one addend value per value.
    per_channel: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The tensor that changes in place. The slot can be longer. We change only `count` values.
@group(0) @binding(1) var<storage, read_write> values: array<f32>;
// The tensor that is added. It is in another slot, because both tensors are live at this op.
@group(0) @binding(2) var<storage, read> addend: array<f32>;

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
    // The addend position: the same position, or the channel of this value.
    let source = select(index, index / shape.pixels, shape.per_channel == 1u);
    // Add it in place.
    values[index] += addend[source];
}
