// GEGLU after its projection: output[row][j] = value[row][j] * GELU(gate[row][j]).
// The projection wrote [rows][2 * inner]: the values in the first half of each row, the gates in the second half.
// GELU is the exact one of PyTorch: 0.5 * x * (1 + erf(x / sqrt(2))). erf comes from erf.wgsl, which Rust puts in front of this text.

// The sizes of this kernel. Rust writes them once, before the first dispatch.
struct Shape {
    // Token rows.
    rows: u32,
    // Values per output row, half of a projected row.
    inner: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The projected rows, [rows][2 * inner].
@group(0) @binding(1) var<storage, read> projected: array<f32>;
// The output rows, [rows][inner].
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE must match 256.
@compute @workgroup_size(256)
// Each invocation writes one output value.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The output value of this invocation.
    let index = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (index >= shape.rows * shape.inner) {
        return;
    }
    // Its row and its position in the row.
    let row = index / shape.inner;
    let column = index % shape.inner;
    // The first value of the projected row.
    let start = row * 2u * shape.inner;
    // The value and its gate.
    let value = projected[start + column];
    let gate = projected[start + shape.inner + column];
    // 1 / sqrt(2).
    let inverse_root_two = 0.70710678118654752;
    // value * GELU(gate).
    output[index] = value * 0.5 * gate * (1.0 + erf(gate * inverse_root_two));
}
