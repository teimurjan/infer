// Causal attention of LFM3 with grouped keys and a sigmoid gate per head:
// out[t][h] = sigmoid(gate[t][h]) * sum over visible keys j of softmax_j(q[t][h] . k[j][h / group] * scale) * v[j][h / group].
// Key j is visible to the query at position p if j <= p, and with a window w also p - j < w (the sliding layers).
// The keys go by in chunks of 128 with a running maximum and a running total (online softmax).

// The sizes. Rust writes them once, before the first run.
struct Shape {
    // Query tokens.
    rows: u32,
    // Keys that exist: the prompt and the earlier tokens.
    keys: u32,
    // Query heads, 16.
    heads: u32,
    // Key and value heads, 4. Query head h uses key head h / (heads / kv_heads).
    kv_heads: u32,
    // The position of query row 0.
    first_position: u32,
    // 0: no window. Otherwise the number of keys that a query sees, its own included.
    window: u32,
    // 1 / sqrt(128), as the bits of an f32.
    scale_bits: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The queries after their norm and rotation: [rows][heads][128].
@group(0) @binding(1) var<storage, read> query: array<f32>;
// The keys after their norm and rotation: [keys][kv_heads][128].
@group(0) @binding(2) var<storage, read> key: array<f32>;
// The values: [keys][kv_heads][128].
@group(0) @binding(3) var<storage, read> value: array<f32>;
// gate_proj before the sigmoid: [rows][heads].
@group(0) @binding(4) var<storage, read> gate: array<f32>;
// The output: [rows][heads][128], the input of o_proj.
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

// Values per head. Rust checks that the weights have 128 too.
const HEAD_DIM: u32 = 128u;
// Keys per chunk: one per lane.
const CHUNK: u32 = 128u;
// The query of this workgroup, read once.
var<workgroup> query_row: array<f32, 128>;
// The scores of the chunk, and whether each key of the chunk is visible.
var<workgroup> scores: array<f32, 128>;
var<workgroup> visible: array<u32, 128>;
// exp(score - running maximum) of each key of the chunk, 0 for keys that are not visible.
var<workgroup> weights: array<f32, 128>;

// One workgroup per query row and head. Lane d owns value d of the output.
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.x;
    let head = group.y;
    let kv_head = head / (shape.heads / shape.kv_heads);
    let position = shape.first_position + row;
    let scale = bitcast<f32>(shape.scale_bits);
    // Copy the query row into workgroup memory, one value per lane.
    query_row[lane] = query[(row * shape.heads + head) * HEAD_DIM + lane];
    // The running maximum, total and weighted sum. All lanes keep the same maximum and total.
    var largest = -3.4e38;
    var total = 0.0;
    var sum = 0.0;
    // The loop count depends only on the uniform key count, so every lane reaches every barrier.
    for (var start = 0u; start < shape.keys; start += CHUNK) {
        // Wait until the query row is there, or the last chunk is done.
        workgroupBarrier();
        // Lane l scores key start + l, if it exists and is visible.
        let key_index = start + lane;
        var seen = key_index < shape.keys && key_index <= position;
        if (shape.window != 0u && seen) {
            seen = position - key_index < shape.window;
        }
        var score = 0.0;
        if (seen) {
            let base = (key_index * shape.kv_heads + kv_head) * HEAD_DIM;
            for (var d = 0u; d < HEAD_DIM; d++) {
                score += query_row[d] * key[base + d];
            }
            score *= scale;
        }
        scores[lane] = score;
        visible[lane] = select(0u, 1u, seen);
        workgroupBarrier();
        // The new maximum over the visible keys of the chunk.
        var new_largest = largest;
        for (var j = 0u; j < CHUNK; j++) {
            if (visible[j] == 1u) {
                new_largest = max(new_largest, scores[j]);
            }
        }
        // The weight of this lane's key. A key that is not visible weighs 0.
        weights[lane] = select(0.0, exp(score - new_largest), seen);
        // Moves the old total and sum to the new maximum. While nothing is visible yet, both are 0 and stay 0.
        let rescale = exp(largest - new_largest);
        largest = new_largest;
        workgroupBarrier();
        // Add the chunk to the total and to this lane's value of the sum.
        var chunk_total = 0.0;
        var chunk_sum = 0.0;
        for (var j = 0u; j < CHUNK; j++) {
            let weight = weights[j];
            if (weight != 0.0) {
                chunk_total += weight;
                chunk_sum += weight * value[((start + j) * shape.kv_heads + kv_head) * HEAD_DIM + lane];
            }
        }
        total = total * rescale + chunk_total;
        sum = sum * rescale + chunk_sum;
    }
    // Finish the softmax, then gate the head: torch.sigmoid(gate_proj(x)).
    let gated = 1.0 / (1.0 + exp(-gate[row * shape.heads + head]));
    output[(row * shape.heads + head) * HEAD_DIM + lane] = sum / total * gated;
}
