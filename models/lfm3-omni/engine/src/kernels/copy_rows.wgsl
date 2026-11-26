// Copy `rows` rows of `width` values: source rows from `source_first` on, to target rows from a first row.
// The first target row is `target_first`, plus the start of this run when `after_start` is 1. The attention layers use
// that to store the keys and values of the new tokens at their positions in the memory of earlier tokens.

// The sizes. Rust writes them once, before the first run.
struct Shape {
    rows: u32,
    width: u32,
    source_first: u32,
    target_first: u32,
    // 1: add the start of this run to target_first.
    after_start: u32,
}

// Where this run starts. The CPU writes it before each run (memory.rs).
struct Step {
    start: u32,
    keys: u32,
}

@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<uniform> step: Step;
@group(0) @binding(2) var<storage, read> source: array<f32>;
@group(0) @binding(3) var<storage, read_write> destination: array<f32>;

// Invocations per workgroup. Rust's WORKGROUP_SIZE in kernels/mod.rs must match.
@compute @workgroup_size(256)
// Each invocation copies one value.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= shape.rows * shape.width) {
        return;
    }
    let row = index / shape.width;
    let column = index % shape.width;
    let first = shape.target_first + select(0u, step.start, shape.after_start == 1u);
    destination[(first + row) * shape.width + column] = source[(shape.source_first + row) * shape.width + column];
}
