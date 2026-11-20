// output[p][c] = level[ids[p]][c] * scale[ids[p]]: the rows of the prompt's tokens, unpacked from the packed token table.
// This is the first half of CLIPTextEmbeddings when the token table is packed. The positions are added after it.
// level x scale is exact in f32: an 8-bit integer times a float16 needs at most 7 + 11 bits.

// The sizes. Rust writes them once, before the first run.
struct Shape {
    // Values in one row, 1024.
    width: u32,
    // Bits per packed integer: 4 or 8.
    bits: u32,
    // u32 words in one packed row.
    row_words: u32,
    // Rows of the table, 49408. An id at or past it writes zeros, but Rust checks the ids first.
    table_rows: u32,
}

// The sizes. "uniform" is a small read-only buffer that every invocation reads.
@group(0) @binding(0) var<uniform> shape: Shape;
// The 77 token ids of the prompt. Rust writes them before each run.
@group(0) @binding(1) var<storage, read> ids: array<u32>;
// The packed table, one row of row_words words per token.
@group(0) @binding(2) var<storage, read> words: array<u32>;
// Two float16 scales per word, the even row in the low 16 bits.
@group(0) @binding(3) var<storage, read> scale: array<u32>;
// The embeddings slot: [77][width].
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// 256 invocations per workgroup. Rust's WORKGROUP_SIZE in token_rows.rs must match.
@compute @workgroup_size(256)
// Each invocation writes one value of the output.
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // The value this invocation writes.
    let index = id.x;
    // The last workgroup can stick out past the end. Those invocations do nothing.
    if (index >= arrayLength(&output)) {
        return;
    }
    // The position in the prompt, and the column in the row.
    let position = index / shape.width;
    let column = index % shape.width;
    // The token of this position.
    let row = ids[position];
    // A row past the table cannot be read. Rust checks the ids, so this only guards the buffer.
    if (row >= shape.table_rows) {
        output[index] = 0.0;
        return;
    }
    // The bit where this integer starts inside the row. 32 is a multiple of 4 and of 8, so an integer never crosses a word.
    let first_bit = column * shape.bits;
    // bitcast keeps the bits and makes the word signed.
    let word = bitcast<i32>(words[row * shape.row_words + first_bit / 32u]);
    // extractBits on an i32 copies the top bit of the field upward, so the integer keeps its sign.
    let level = f32(extractBits(word, first_bit % 32u, shape.bits));
    // unpack2x16float gives the low half first, so row 2j is .x and row 2j + 1 is .y.
    output[index] = level * unpack2x16float(scale[row / 2u])[row % 2u];
}
