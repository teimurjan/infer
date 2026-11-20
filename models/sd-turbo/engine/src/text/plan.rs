// The graph of a forward pass, the planner, and the plan type.
use crate::plan::{Graph, MemoryPlan, assign_slots};

// The numbers below come from text_encoder/config.json of stabilityai/sd-turbo.
// Tokens per prompt (max_position_embeddings).
pub(crate) const TOKENS: u32 = 77;
// Values per token (hidden_size).
pub(crate) const WIDTH: u32 = 1024;
// Encoder layers (num_hidden_layers).
pub(crate) const LAYERS: usize = 23;
// The width of the MLP (intermediate_size).
const MLP_WIDTH: u32 = 4096;
// The tensor that the CPU writes: token embedding plus position embedding. The layers change it in place.
pub(crate) const EMBEDDINGS: &str = "embeddings";
// The output: final_layer_norm, which is last_hidden_state.
pub(crate) const OUTPUT: &str = "text_model.final_layer_norm";

// The weight name prefix of encoder layer `layer`.
pub(crate) fn layer_prefix(layer: usize) -> String {
    // For example "text_model.encoder.layers.0".
    format!("text_model.encoder.layers.{layer}")
}

// Record the forward pass of CLIPTextModel, op by op, as the engine runs it.
fn text_graph() -> Graph {
    // An empty graph.
    let mut graph = Graph::default();
    // The size of every token tensor.
    let rows = [TOKENS, WIDTH, 1];
    // The embeddings. They are the stream that every layer adds into.
    let hidden = graph.op(EMBEDDINGS, &[], rows);
    // The 23 layers.
    for layer in 0..LAYERS {
        // For example "text_model.encoder.layers.0".
        let prefix = layer_prefix(layer);
        // layer_norm1, then q, k and v with bias.
        let norm1 = graph.op(format!("{prefix}.layer_norm1"), &[hidden], rows);
        let [query, key, value] = ["q_proj", "k_proj", "v_proj"].map(|part| graph.op(format!("{prefix}.self_attn.{part}"), &[norm1], rows));
        // Causal self attention. It keeps its scores in workgroup memory, so it needs no scratch tensor.
        let mixed = graph.op(format!("{prefix}.self_attn.attention"), &[query, key, value], rows);
        // out_proj, then add it into the stream.
        let attention = graph.op(format!("{prefix}.self_attn.out_proj"), &[mixed], rows);
        graph.update(hidden, &[attention]);
        // layer_norm2, fc1, GELU in place, fc2, then add it into the stream.
        let norm2 = graph.op(format!("{prefix}.layer_norm2"), &[hidden], rows);
        let wide = graph.op(format!("{prefix}.mlp.fc1"), &[norm2], [TOKENS, MLP_WIDTH, 1]);
        graph.update(wide, &[]);
        let narrow = graph.op(format!("{prefix}.mlp.fc2"), &[wide], rows);
        graph.update(hidden, &[narrow]);
    }
    // final_layer_norm. The UNet stage reads it after this stage, so it lives to the end.
    let output = graph.op(OUTPUT, &[hidden], rows);
    graph.keep(output);
    // The full pass.
    graph
}

// The forward pass and the slot of each of its tensors.
pub(crate) fn planned_text() -> (Graph, MemoryPlan) {
    // The ops and tensors.
    let graph = text_graph();
    // The lifetimes, then the slots.
    let plan = assign_slots(&graph.lifetimes());
    // Both.
    (graph, plan)
}
