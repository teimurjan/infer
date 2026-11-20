// Attention for one chunk of query rows, for every head:
// output[i] = sum over keys j of softmax_j(query[i] . key[j] / sqrt(64)) * value[j].
// The tensors use the layout of the SD-Turbo UNet before it splits the heads: [tokens][heads * 64].
// Head h owns values h * 64 to h * 64 + 63 of every row.
// Self attention has as many keys as queries. Cross attention has other keys, for example 77 text tokens.
// The kernel walks over the keys in tiles of 16, with an "online softmax": a running maximum and a running total.
// The scores live only in workgroup memory for one tile, so they never go to the storage buffers.

// One head has 64 values, which is 16 vec4. Rust's HEAD_SIZE must be 64.
const HEAD_VEC4S: u32 = 16u;
// Query rows per workgroup. Rust's ROWS_PER_WORKGROUP must match.
const ROWS: u32 = 16u;
// Invocations that share one query row. For the scores, lane l takes key l of the tile.
// For the output, lane l owns vec4 number l of the row. 16 rows x 16 lanes = 256 invocations.
const LANES: u32 = 16u;
// Keys per tile. It equals LANES, so each lane makes one score per tile.
const TILE_KEYS: u32 = 16u;
// 1 / sqrt(64). It is a power of 2, so scaling by it adds no rounding error.
const SCALE: f32 = 0.125;
// Almost the lowest f32 (the exact limit is -3.40282347e38, and some compilers reject a literal written past it).
// The running maximum starts here, and a key that the row must not see gets this score. Any real score is larger.
const LOWEST: f32 = -3.4e38;

// The sizes of this chunk. Rust writes one of these per chunk, before the first run.
struct Shape {
    // Keys and values: the same as the queries for self attention, 77 for SD-Turbo cross attention.
    key_tokens: u32,
    // Heads, for example 5.
    heads: u32,
    // The first query row of this chunk.
    chunk_start: u32,
    // Query rows in this chunk. The last chunk can have fewer rows than the others.
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
// The output rows, in the same layout as the queries.
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

// "workgroup" memory is shared by the 256 invocations of a workgroup. It is much faster than a storage buffer.
// The 16 key rows of this tile, for our head: [key in tile][vec4 part].
var<workgroup> tile_key: array<vec4<f32>, 256>;
// The 16 value rows of this tile, in the same layout.
var<workgroup> tile_value: array<vec4<f32>, 256>;
// The scores of the tile: [key in tile][row in group]. Neighbor rows sit next to each other,
// so the 16 rows that read the same key read 16 neighbor addresses.
var<workgroup> tile_score: array<f32, 256>;
// exp(score - running maximum) of each score, in the same layout. A separate array, because some lanes
// still read tile_score while the others write their weights.
var<workgroup> tile_weight: array<f32, 256>;

// The index of vec4 number `part` of head `head` in row `token`.
fn row_vec4(token: u32, head: u32, part: u32) -> u32 {
    // Skip the earlier rows, then the earlier heads of this row.
    return (token * shape.heads + head) * HEAD_VEC4S + part;
}

// The index of key `key_in_tile` and row `row_in_group` in tile_score and tile_weight.
fn tile_index(key_in_tile: u32, row_in_group: u32) -> u32 {
    // Skip the earlier keys, then go to our row.
    return key_in_tile * ROWS + row_in_group;
}

// 256 invocations per workgroup: 16 query rows x 16 lanes, for one head.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    // y picks the head.
    let head = group.y;
    // Our row in the workgroup, 0 to 15.
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
    // Our whole query row, scaled by 1 / 8, in registers. Every lane needs all of it for its dot product.
    // WGSL starts every var at 0.
    var q: array<vec4<f32>, HEAD_VEC4S>;
    // Only a real row reads the query.
    if (in_chunk) {
        // 16 vec4 loads.
        for (var part = 0u; part < HEAD_VEC4S; part++) {
            q[part] = query[row_vec4(row, head, part)] * SCALE;
        }
    }
    // The keys that this row sees: all of them, or with the causal mask only keys 0 to row.
    // Key 0 is always seen and sits in the first tile, so the running maximum is a real score after that tile.
    let keys = select(shape.key_tokens, min(row + 1u, shape.key_tokens), shape.causal == 1u);

    // Every lane of a row keeps the same three running values. They get the same inputs, so they stay equal.
    // The largest score of the row so far.
    var largest = LOWEST;
    // The sum of exp(score - largest) so far.
    var total = 0.0;
    // Our vec4 of the sum of exp(score - largest) x value row so far.
    var sum = vec4<f32>(0.0);
    // The tile in workgroup memory that this invocation fills: key `local / 16`, vec4 part `local % 16`.
    let fill_key = local / HEAD_VEC4S;
    let fill_part = local % HEAD_VEC4S;

    // The loop count depends only on the uniform key count, so every invocation reaches every barrier.
    for (var tile_start = 0u; tile_start < shape.key_tokens; tile_start += TILE_KEYS) {
        // Step 1: copy the key and value rows of this tile into workgroup memory, one vec4 per invocation.
        let source_key = tile_start + fill_key;
        // The last tile can stick out past the keys. Its extra rows stay 0, and their scores are masked below.
        var key_part = vec4<f32>(0.0);
        var value_part = vec4<f32>(0.0);
        if (source_key < shape.key_tokens) {
            key_part = key[row_vec4(source_key, head, fill_part)];
            value_part = value[row_vec4(source_key, head, fill_part)];
        }
        tile_key[local] = key_part;
        tile_value[local] = value_part;
        // Wait until the whole tile is in workgroup memory.
        workgroupBarrier();

        // Step 2: the score of our row and key `lane` of the tile.
        let key_index = tile_start + lane;
        // A key past the end, or past our row with the causal mask, gets the lowest score, so its weight is exp(lowest) = 0.
        var score = LOWEST;
        if (key_index < keys) {
            // The dot product of our query row and this key. 4 separate sums in a vec4, added at the end,
            // so the GPU does not wait for each add to finish before the next one.
            var products = vec4<f32>(0.0);
            for (var part = 0u; part < HEAD_VEC4S; part++) {
                products += q[part] * tile_key[lane * HEAD_VEC4S + part];
            }
            score = products.x + products.y + products.z + products.w;
        }
        // Share it with the other lanes of our row.
        tile_score[tile_index(lane, row_in_group)] = score;
        // Wait until all 16 scores of every row are there.
        workgroupBarrier();

        // Step 3: the new running maximum of our row, and the weight of our score.
        var new_largest = largest;
        for (var other = 0u; other < TILE_KEYS; other++) {
            new_largest = max(new_largest, tile_score[tile_index(other, row_in_group)]);
        }
        // One exp per score: each lane makes the weight of its own key. exp of a value of at most 0 cannot overflow.
        tile_weight[tile_index(lane, row_in_group)] = exp(score - new_largest);
        // The old total and sum were made with the old maximum. This factor moves them to the new one.
        // In the first tile, largest is LOWEST and the factor is 0, which leaves the empty total and sum at 0.
        let rescale = exp(largest - new_largest);
        largest = new_largest;
        // Wait until every weight of the tile is there.
        workgroupBarrier();

        // Step 4: add the 16 weights and the weighted value rows of the tile, for our vec4 of the output.
        var tile_total = 0.0;
        var tile_sum = vec4<f32>(0.0);
        for (var other = 0u; other < TILE_KEYS; other++) {
            let weight = tile_weight[tile_index(other, row_in_group)];
            tile_total += weight;
            tile_sum += weight * tile_value[other * HEAD_VEC4S + lane];
        }
        total = total * rescale + tile_total;
        sum = sum * rescale + tile_sum;
        // Wait until every lane is done with this tile, before the next tile overwrites it.
        workgroupBarrier();
    }

    // Only a real row writes its output.
    if (in_chunk) {
        // Divide by the total to finish the softmax, and write our vec4.
        output[row_vec4(row, head, lane)] = sum / total;
    }
}
