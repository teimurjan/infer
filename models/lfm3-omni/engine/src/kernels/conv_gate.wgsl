// The first part of the gated short convolution: h = B * x for each new token, written into the history of the layer
// at the token's position. The convolution (short_conv.wgsl) then reads the last 7 positions of the history.

// The sizes. Rust writes them once, before the first run.
struct Shape {
    // New tokens in this run.
    rows: u32,
    // Channels, 2048. One row of bcx holds B, C and x: 3 x width values.
    width: u32,
}

// Where this run starts. The CPU writes it before each run (memory.rs).
struct Step {
    // The position of the first new token.
    start: u32,
    // The tokens that exist after this run: start + rows.
    keys: u32,
}

@group(0) @binding(0) var<uniform> shape: Shape;
@group(0) @binding(1) var<uniform> step: Step;
// The output of in_proj for the new tokens: [rows][B | C | x].
@group(0) @binding(2) var<storage, read> bcx: array<f32>;
// h of every position so far: [capacity][width].
@group(0) @binding(3) var<storage, read_write> history: array<f32>;

// Invocations per workgroup. Rust's WORKGROUP_SIZE in kernels/mod.rs must match.
@compute @workgroup_size(256)
// Each invocation writes one value: new token t, channel c.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= shape.rows * shape.width) {
        return;
    }
    let token = index / shape.width;
    let channel = index % shape.width;
    let row = token * 3u * shape.width;
    // h = B * x, stored at the position of this token.
    history[(step.start + token) * shape.width + channel] = bcx[row + channel] * bcx[row + 2u * shape.width + channel];
}
