// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// A kernel with its buffers.
use crate::dispatch::BoundKernel;
// Where the bias of a linear layer comes from.
use crate::conv::Bias;
// The linear layer on token rows.
use crate::linear::bind_rows_linear;
// The attention kernel on slots.
use crate::attention::{AttentionSlots, bind_attention_into};
// LayerNorm, the in-place add and GELU.
use crate::kernels::{bind_add, bind_gelu, bind_layer_norm};
// The buffers of the activation plan.
use crate::plan::ActivationPool;
// One submit of a stage.
use crate::step::{Layout, Step};
// The weight buffers of the slot kernels.
use crate::weights::WeightBuffers;

// The layout of the text encoder.
use super::plan::{EMBEDDINGS, LAYERS, OUTPUT, TOKENS, layer_prefix};
// The position add.
use super::embed::bind_positions;

// Bind every op of the text encoder to the slots of `pool`, in run order. One step per layer.
// The ops and the tensor names are the ones of text/plan.rs.
pub(super) fn bind_text(gpu: &Gpu, weights: &impl WeightBuffers, pool: &ActivationPool) -> Result<Vec<Step>, EngineError> {
    // The run copies the token rows in. This step adds the position embeddings and shows the result.
    let mut steps = vec![Step::new("text embeddings", vec![bind_positions(gpu, weights, pool)?]).checked("text_embeddings", EMBEDDINGS, Layout::Tokens)];
    // The 23 layers.
    for layer in 0..LAYERS {
        // One step per layer. Layer 0 shows its output, the stream after the layer.
        let step = Step::new(layer_prefix(layer), bind_layer(gpu, weights, pool, &layer_prefix(layer))?);
        steps.push(if layer == 0 { step.checked("text_layer_0", EMBEDDINGS, Layout::Tokens) } else { step });
    }
    // final_layer_norm gives the output.
    let end = bind_layer_norm(gpu, weights, OUTPUT, &pool.tensor(EMBEDDINGS)?, &pool.tensor(OUTPUT)?)?;
    steps.push(Step::new("text final_layer_norm", vec![end]).checked("text_encoder", OUTPUT, Layout::Tokens));
    // Every step.
    Ok(steps)
}

// The kernels of one CLIPEncoderLayer, in run order.
fn bind_layer(gpu: &Gpu, weights: &impl WeightBuffers, pool: &ActivationPool, prefix: &str) -> Result<Vec<BoundKernel>, EngineError> {
    // The tensor names of the plan.
    let [norm1, query, key, value, mixed, attention] = ["layer_norm1", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.attention", "self_attn.out_proj"].map(|part| format!("{prefix}.{part}"));
    let [norm2, wide, narrow] = ["layer_norm2", "mlp.fc1", "mlp.fc2"].map(|part| format!("{prefix}.{part}"));
    // The stream.
    let hidden = pool.tensor(EMBEDDINGS)?;
    // The kernels.
    Ok(vec![
        // layer_norm1.
        bind_layer_norm(gpu, weights, &norm1, &hidden, &pool.tensor(&norm1)?)?,
        // q, k and v. CLIP has a bias on each.
        bind_rows_linear(gpu, weights, &query, &pool.tensor(&norm1)?, Bias::FromWeights, &pool.tensor(&query)?)?,
        bind_rows_linear(gpu, weights, &key, &pool.tensor(&norm1)?, Bias::FromWeights, &pool.tensor(&key)?)?,
        bind_rows_linear(gpu, weights, &value, &pool.tensor(&norm1)?, Bias::FromWeights, &pool.tensor(&value)?)?,
        // Causal self attention: token i sees tokens 0 to i. All 77 rows fit one chunk.
        bind_attention_into(gpu, &AttentionSlots { query: pool.tensor(&query)?, key: pool.tensor(&key)?, value: pool.tensor(&value)?, output: pool.tensor(&mixed)?, chunk_rows: TOKENS, causal: true })?,
        // out_proj, then the residual add.
        bind_rows_linear(gpu, weights, &attention, &pool.tensor(&mixed)?, Bias::FromWeights, &pool.tensor(&attention)?)?,
        bind_add(gpu, &hidden, &pool.tensor(&attention)?)?,
        // layer_norm2, fc1, GELU, fc2, then the residual add.
        bind_layer_norm(gpu, weights, &norm2, &hidden, &pool.tensor(&norm2)?)?,
        bind_rows_linear(gpu, weights, &wide, &pool.tensor(&norm2)?, Bias::FromWeights, &pool.tensor(&wide)?)?,
        bind_gelu(gpu, &pool.tensor(&wide)?)?,
        bind_rows_linear(gpu, weights, &narrow, &pool.tensor(&wide)?, Bias::FromWeights, &pool.tensor(&narrow)?)?,
        bind_add(gpu, &hidden, &pool.tensor(&narrow)?)?,
    ])
}
