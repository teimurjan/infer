// gate[i] = silu(gate[i]) * up[i], the middle of the LFM3 feed-forward layer: w2(silu(w1(x)) * w3(x)).
// It writes over `gate` in place, so the feed-forward layer needs no third wide buffer.

// The output of w1, [rows][7680]. It becomes the input of w2.
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
// The output of w3, in the same layout.
@group(0) @binding(1) var<storage, read> up: array<f32>;

// Invocations per workgroup. Rust's WORKGROUP_SIZE in swiglu.rs must match.
@compute @workgroup_size(256)
// Each invocation changes one value.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (index >= arrayLength(&gate)) {
        return;
    }
    let value = gate[index];
    // silu(v) = v / (1 + exp(-v)), as torch.nn.functional.silu.
    gate[index] = value / (1.0 + exp(-value)) * up[index];
}
