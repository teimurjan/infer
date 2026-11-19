// The graph that records the ops.
use crate::plan::{Graph, TensorId};

// Query rows per attention dispatch. The attention kernel puts the scores of one chunk into a scratch buffer
// of heads x key_tokens x chunk values. 256 is the chunk that ran on the iPhone in 145.0 ms per submit ("Attention on the iPhone").
const CHUNK_ROWS: u32 = 256;
// The feed-forward layer is 4 times wider than the block (GEGLU, mult=4 in diffusers).
const FEED_FORWARD_FACTOR: u32 = 4;

// One Transformer2DModel with use_linear_projection=True and one BasicTransformerBlock, as the ops that the engine will run.
// `text` is the text encoder output, [77, 1024, 1]. Returns the output, the same size as `input`.
pub(super) fn transformer(graph: &mut Graph, prefix: &str, input: TensorId, text: TensorId, heads: u32) -> TensorId {
    // The input size.
    let [channels, height, width] = graph.dims(input);
    // Every pixel is one token.
    let tokens = height * width;
    // GroupNorm with 32 groups, no SiLU. The kernel writes [tokens][channels], so proj_in can read rows.
    let norm = graph.op(format!("{prefix}.norm"), &[input], [tokens, channels, 1]);
    // proj_in: a linear layer on each token row.
    let hidden = graph.op(format!("{prefix}.proj_in"), &[norm], [tokens, channels, 1]);
    // The one block of this transformer.
    let block = format!("{prefix}.transformer_blocks.0");
    // Self attention: keys and values come from the block itself.
    attention(graph, &block, 1, hidden, None, heads);
    // Cross attention: keys and values come from the text.
    attention(graph, &block, 2, hidden, Some(text), heads);
    // The feed-forward part.
    feed_forward(graph, &block, hidden);
    // proj_out: a linear layer on each token row.
    let rows = graph.op(format!("{prefix}.proj_out"), &[hidden], [tokens, channels, 1]);
    // Back to [channels][pixels], plus the transformer input, into a new tensor.
    graph.op(format!("{prefix}.output"), &[rows, input], [channels, height, width])
}

// LayerNorm, then attention `number` (1 or 2) of the block, then add the result into `hidden` in place.
fn attention(graph: &mut Graph, block: &str, number: u32, hidden: TensorId, text: Option<TensorId>, heads: u32) {
    // [tokens, channels, 1].
    let rows = graph.dims(hidden);
    // The token count and the channel count.
    let [tokens, channels, _] = rows;
    // norm1 before attn1, norm2 before attn2.
    let norm = graph.op(format!("{block}.norm{number}"), &[hidden], rows);
    // For example "...transformer_blocks.0.attn1".
    let name = format!("{block}.attn{number}");
    // to_q: a linear layer without bias.
    let query = graph.op(format!("{name}.to_q"), &[norm], rows);
    // Self attention reads keys from the normed block. Cross attention reads them from the text.
    let source = text.unwrap_or(norm);
    // 4096 keys for self attention at 64 x 64, 77 for cross attention.
    let [key_tokens, _, _] = graph.dims(source);
    // to_k and to_v: linear layers without bias, to the channel count of the block.
    let key = graph.op(format!("{name}.to_k"), &[source], [key_tokens, channels, 1]);
    let value = graph.op(format!("{name}.to_v"), &[source], [key_tokens, channels, 1]);
    // A chunk is never larger than the query count.
    let chunk = CHUNK_ROWS.min(tokens);
    // The attention kernel, with its scores scratch of [heads][key_tokens][chunk].
    let mixed = graph.op_with_scratch(format!("{name}.attention"), &[query, key, value], rows, [heads, key_tokens, chunk]);
    // to_out.0: a linear layer with bias.
    let output = graph.op(format!("{name}.to_out.0"), &[mixed], rows);
    // Add it into the block, in place.
    graph.update(hidden, &[output]);
}

// LayerNorm, GEGLU, a linear layer, then add the result into `hidden` in place.
fn feed_forward(graph: &mut Graph, block: &str, hidden: TensorId) {
    // [tokens, channels, 1].
    let rows = graph.dims(hidden);
    // The token count and the channel count.
    let [tokens, channels, _] = rows;
    // The width of the inner layer.
    let inner = FEED_FORWARD_FACTOR * channels;
    // norm3.
    let norm = graph.op(format!("{block}.norm3"), &[hidden], rows);
    // ff.net.0.proj: a linear layer to 2 x inner values, the value half and the gate half.
    let projected = graph.op(format!("{block}.ff.net.0.proj"), &[norm], [tokens, 2 * inner, 1]);
    // value x GELU(gate), into a new tensor of inner values.
    let gated = graph.op(format!("{block}.ff.net.0.geglu"), &[projected], [tokens, inner, 1]);
    // ff.net.2: a linear layer back to the block width.
    let output = graph.op(format!("{block}.ff.net.2"), &[gated], rows);
    // Add it into the block, in place.
    graph.update(hidden, &[output]);
}
