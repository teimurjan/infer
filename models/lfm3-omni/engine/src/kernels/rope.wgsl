// Rotary positions of LFM3 (apply_rotary_pos_emb), in place on [rows][heads][head_dim]:
// for each pair (j, j + head_dim / 2) with j < angles: a' = a cos - b sin, b' = b cos + a sin.
// Pairs with j >= angles have a frequency of 0 in LFM3 (partial rotation), so they stay as they are.
// The cos and sin of each position and pair come from a table that Rust makes.

// The sizes. Rust writes them once, before the first run.
struct Shape {
    // Tokens.
    rows: u32,
    // Heads of this tensor: 16 for the queries, 4 for the keys.
    heads: u32,
    // Values per head, 128.
    head_dim: u32,
    // Rotated pairs per head: 32 in the full attention layers, 64 in the sliding ones.
    angles: u32,
}

// Where this run starts. The CPU writes it before each run (memory.rs). Row r is at position start + r.
struct Step {
    start: u32,
    keys: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<uniform> step: Step;
// The queries or the keys, changed in place.
@group(0) @binding(2) var<storage, read_write> values: array<f32>;
// cos of each position and pair: [positions][angles].
@group(0) @binding(3) var<storage, read> cos_table: array<f32>;
// sin in the same layout.
@group(0) @binding(4) var<storage, read> sin_table: array<f32>;

// Invocations per workgroup. Rust's WORKGROUP_SIZE in kernels/mod.rs must match.
@compute @workgroup_size(256)
// Each invocation rotates one pair of one head of one row.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (index >= shape.rows * shape.heads * shape.angles) {
        return;
    }
    // Which pair, head and row this is.
    let pair = index % shape.angles;
    let head = (index / shape.angles) % shape.heads;
    let row = index / (shape.angles * shape.heads);
    // The first value of the pair, and its partner half a head later, as rotate_half pairs them.
    let first = (row * shape.heads + head) * shape.head_dim + pair;
    let second = first + shape.head_dim / 2u;
    // The cos and sin of this position and pair.
    let table = (step.start + row) * shape.angles + pair;
    let c = cos_table[table];
    let s = sin_table[table];
    let a = values[first];
    let b = values[second];
    values[first] = a * c - b * s;
    values[second] = b * c + a * s;
}
