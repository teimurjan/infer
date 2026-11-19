// Attention for one chunk of query rows, for every head:
// output[i] = sum over keys j of softmax_j(query[i] . key[j] / sqrt(64)) * value[j].
// The tensors use the layout of the SD-Turbo UNet before it splits the heads: [tokens][heads * 64].
// Head h owns values h * 64 to h * 64 + 63 of every row.
// Self attention has as many keys as queries. Cross attention has other keys, for example 77 text tokens.
// The scores of the chunk go to a scratch buffer of [heads][key_tokens][chunk_rows] values,
// so memory grows with the chunk size, not with query_tokens x key_tokens.

// One head has 64 values, which is 16 vec4. Rust's HEAD_SIZE must be 64.
const HEAD_VEC4S: u32 = 16u;
// Query rows per workgroup. Rust's ROWS_PER_WORKGROUP must match.
const ROWS: u32 = 16u;
// Invocations that share one query row. For the scores, lane l takes keys l, l + 16, l + 32, and so on.
// For the output, lane l owns vec4 number l of the row. 16 rows x 16 lanes = 256 invocations.
const LANES: u32 = 16u;
// 1 / sqrt(64). It is a power of 2, so scaling by it adds no rounding error.
const SCALE: f32 = 0.125;
// Almost the lowest f32 (the exact limit is -3.40282347e38, and some compilers reject a literal written past it).
// The running maximum starts here. Any real score is larger.
const LOWEST: f32 = -3.4e38;

// The sizes of this chunk. Rust writes one of these per chunk, before the first run.
struct Shape {
    // Keys and values: the same as the queries for self attention, 77 for SD-Turbo cross attention.
    key_tokens: u32,
    // Heads, for example 5.
    heads: u32,
    // Rows of a full chunk. It is the step between keys in the scores buffer.
    chunk_rows: u32,
    // The first query row of this chunk.
    chunk_start: u32,
    // Query rows in this chunk. The last chunk can have fewer than chunk_rows.
    rows: u32,
    // 1: query row i sees only keys 0 to i (the causal mask of the text encoder). 0: every key.
    causal: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// All query rows: [query_tokens][heads * 16] vec4.
@group(0) @binding(1) var<storage, read> query: array<vec4<f32>>;
// All key rows: [key_tokens][heads * 16] vec4.
@group(0) @binding(2) var<storage, read> key: array<vec4<f32>>;
// All value rows, in the same layout as the keys.
@group(0) @binding(3) var<storage, read> value: array<vec4<f32>>;
// Scratch for the scores of this chunk: [heads][key_tokens][chunk_rows]. Every chunk reuses it.
@group(0) @binding(4) var<storage, read_write> scores: array<f32>;
// The output rows, in the same layout as the queries.
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;

// The largest score that each invocation found, so the 16 lanes of a row can combine theirs.
// "workgroup" memory is shared by the 256 invocations of a workgroup.
var<workgroup> lane_largest: array<f32, 256>;

// The index of vec4 number `part` of head `head` in row `token`.
fn row_vec4(token: u32, head: u32, part: u32) -> u32 {
    // Skip the earlier rows, then the earlier heads of this row.
    return (token * shape.heads + head) * HEAD_VEC4S + part;
}

// The index of the score of key `key_index` for query row `local_row` of the chunk.
// Neighbor rows are next to each other, so invocations of neighbor rows read and write neighbor addresses.
fn score_index(head: u32, key_index: u32, local_row: u32) -> u32 {
    // Skip the earlier heads and keys, then go to our row.
    return (head * shape.key_tokens + key_index) * shape.chunk_rows + local_row;
}

// 256 invocations per workgroup: 16 query rows x 16 lanes, for one head.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    // y picks the head.
    let head = group.y;
    // Our row in the workgroup, 0 to 15. The GPU runs 32 neighbor invocations together:
    // 16 rows x 2 lanes. They read the same 2 key rows, which the cache serves once.
    let row_in_group = local % ROWS;
    // Our lane, 0 to 15.
    let lane = local / ROWS;
    // x picks a group of 16 rows of the chunk.
    let local_row = group.x * ROWS + row_in_group;
    // The last workgroup can stick out past the chunk. Those invocations must still reach the barriers,
    // because every invocation of a workgroup must reach them. So they cannot return early.
    let in_chunk = local_row < shape.rows;
    // Our row in the full query tensor.
    let row = shape.chunk_start + local_row;
    // Our whole query row, scaled by 1 / 8, in registers. Every lane needs all of it for its dot products.
    // WGSL starts every var at 0.
    var q: array<vec4<f32>, HEAD_VEC4S>;
    // Only a real row reads the query.
    if (in_chunk) {
        // 16 vec4 loads.
        for (var part = 0u; part < HEAD_VEC4S; part++) {
            q[part] = query[row_vec4(row, head, part)] * SCALE;
        }
    }

    // The keys that this row sees: all of them, or with the causal mask only keys 0 to row. Key `row` is always seen, so the total of pass 2 is never 0.
    let keys = select(shape.key_tokens, min(row + 1u, shape.key_tokens), shape.causal == 1u);
    // Pass 1: our scores, and the largest of them. Softmax needs the largest score of the row before it can start.
    var largest = LOWEST;
    // Our keys: lane, lane + 16, lane + 32, and so on.
    for (var key_index = lane; key_index < keys; key_index += LANES) {
        // Rows outside the chunk do nothing.
        if (in_chunk) {
            // The dot product of our query row and this key. 4 separate sums in a vec4, added at the end,
            // so the GPU does not wait for each add to finish before the next one.
            var products = vec4<f32>(0.0);
            for (var part = 0u; part < HEAD_VEC4S; part++) {
                products += q[part] * key[row_vec4(key_index, head, part)];
            }
            let score = products.x + products.y + products.z + products.w;
            // Keep it for pass 2.
            scores[score_index(head, key_index, local_row)] = score;
            // Track the largest of our keys.
            largest = max(largest, score);
        }
    }
    // Share our largest score with the other lanes of the row.
    lane_largest[local] = largest;
    // Wait until all lanes have written their largest score (workgroup memory).
    workgroupBarrier();
    // Also wait until their scores are visible (storage memory). Pass 2 reads the scores of all lanes,
    // and workgroupBarrier alone does not cover storage memory.
    storageBarrier();
    // The largest score of the whole row: the largest of all lanes. Lane l of our row sits at row_in_group + l * ROWS.
    for (var other = 0u; other < LANES; other++) {
        largest = max(largest, lane_largest[row_in_group + other * ROWS]);
    }

    // Pass 2: softmax weights and the weighted sum of the values. Each lane goes over every key,
    // but only for its own vec4 of the output. All lanes compute the same weights and the same total.
    // exp(score - largest) is at most 1, so it cannot overflow. Dividing by the total at the end gives the softmax.
    var total = 0.0;
    // Our 4 values of the weighted sum.
    var sum = vec4<f32>(0.0);
    // Every key that this row sees.
    for (var key_index = 0u; key_index < keys; key_index++) {
        // Rows outside the chunk do nothing.
        if (in_chunk) {
            // The weight before dividing by the total. One of the 16 lanes wrote this score in pass 1.
            let weight = exp(scores[score_index(head, key_index, local_row)] - largest);
            // Add it to the total.
            total += weight;
            // Add weight x our vec4 of the value row.
            sum += weight * value[row_vec4(key_index, head, lane)];
        }
    }

    // Only a real row writes its output.
    if (in_chunk) {
        // Divide by the total to finish the softmax, and write our vec4.
        output[row_vec4(row, head, lane)] = sum / total;
    }
}
