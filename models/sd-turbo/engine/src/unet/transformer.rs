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
use crate::attention::{AttentionSlots, bind_attention_groups, bind_attention_into};
// The buffers of the activation plan.
use crate::plan::ActivationPool;
// The weight buffers of the slot kernels.
use crate::weights::WeightBuffers;

// The norms, GEGLU, the add, and the layout change at the end.
use crate::kernels::{NormLayout, bind_add, bind_geglu, bind_group_norm_into, bind_layer_norm, bind_transpose_add};
// One submit of the stage.
use crate::step::{Layout, Step};

// The eps of the GroupNorm at the start of every transformer (Transformer2DModel uses 1e-6).
const NORM_EPS: f32 = 1e-6;
// Query rows of self attention per submit. At 4,096 tokens, all rows in one submit took 250 to 298 ms on the iPhone
// in ten benchmark runs in a row, past the 250 ms limit ("Benchmark page on the iPhone"). 1,024 rows make 4 submits there.
// Smaller images have 1,024 rows or fewer, so they keep one submit.
const SELF_ATTENTION_ROWS_PER_SUBMIT: u32 = 1024;

// The GPU, the weights and the pool, which every part of the transformer needs.
struct Context<'a, W: WeightBuffers> {
    // The GPU.
    gpu: &'a Gpu,
    // The weight reader.
    weights: &'a W,
    // The slots.
    pool: &'a ActivationPool,
}

// Read the weights of the transformer `prefix` and bind it from the `input` tensor. The output is the tensor "<prefix>.output".
// The ops and the tensor names are the ones of unet/plan/transformer.rs. The text is the tensor `text`.
// Seven steps, plus 3 more at 64 x 64. Self attention has steps of its own, because it is the longest kernel (145.0 ms per run on the iPhone at 64 x 64).
// It has one step per SELF_ATTENTION_ROWS_PER_SUBMIT query rows.
pub(crate) fn bind_transformer(gpu: &Gpu, weights: &impl WeightBuffers, pool: &ActivationPool, prefix: &str, input: &str, text: &str) -> Result<Vec<Step>, EngineError> {
    // The parts that every helper needs.
    let context = Context { gpu, weights, pool };
    // The one block of this transformer.
    let block = format!("{prefix}.transformer_blocks.0");
    // The stream of token rows that the block changes in place.
    let hidden = format!("{prefix}.proj_in");
    // GroupNorm (written as token rows), then proj_in.
    let norm = format!("{prefix}.norm");
    let start = vec![
        bind_group_norm_into(gpu, weights, &norm, &pool.tensor(input)?, false, NORM_EPS, NormLayout::Tokens, &pool.tensor(&norm)?)?,
        bind_rows_linear(gpu, weights, &hidden, &pool.tensor(&norm)?, Bias::FromWeights, &pool.tensor(&hidden)?)?,
    ];
    // The steps, in run order.
    let mut steps = vec![Step::new(format!("{prefix} proj_in"), start).checked(&hidden, &hidden, Layout::Tokens)];
    // Self attention: norm1 and q, k, v; then the attention; then to_out.0.
    let self_attention = format!("{block}.attn1");
    steps.push(Step::new(format!("{self_attention} q k v"), attention_inputs(&context, &block, 1, &hidden, None)?));
    let groups = self_attention_groups(&context, &self_attention)?;
    let parts = groups.len();
    // One step per group of query rows, for example "...attn1 attention 2 of 4".
    steps.extend(groups.into_iter().enumerate().map(|(index, group)| Step::new(format!("{self_attention} attention {} of {parts}", index + 1), vec![group])));
    steps.push(Step::new(format!("{self_attention} to_out"), vec![out_projection(&context, &self_attention)?]).checked(&self_attention, format!("{self_attention}.to_out.0"), Layout::Tokens));
    // Cross attention, after the self attention result is added in. It reads only 77 keys, so it fits one step.
    let cross_attention = format!("{block}.attn2");
    let mut cross = vec![bind_add(gpu, &pool.tensor(&hidden)?, &pool.tensor(&format!("{self_attention}.to_out.0"))?)?];
    cross.extend(attention_inputs(&context, &block, 2, &hidden, Some(text))?);
    cross.push(attention(&context, &cross_attention)?);
    cross.push(out_projection(&context, &cross_attention)?);
    steps.push(Step::new(cross_attention.clone(), cross).checked(&cross_attention, format!("{cross_attention}.to_out.0"), Layout::Tokens));
    // The feed-forward part, after the cross attention result is added in.
    let ff_output = format!("{block}.ff.net.2");
    let mut ff = vec![bind_add(gpu, &pool.tensor(&hidden)?, &pool.tensor(&format!("{cross_attention}.to_out.0"))?)?];
    ff.extend(feed_forward(&context, &block, &hidden)?);
    steps.push(Step::new(format!("{block}.ff"), ff).checked(format!("{block}.ff"), &ff_output, Layout::Tokens));
    // The end: add the feed-forward result, proj_out, then back to [channels][pixels] plus the transformer input.
    let [proj_out, output] = ["proj_out", "output"].map(|part| format!("{prefix}.{part}"));
    let end = vec![
        bind_add(gpu, &pool.tensor(&hidden)?, &pool.tensor(&ff_output)?)?,
        bind_rows_linear(gpu, weights, &proj_out, &pool.tensor(&hidden)?, Bias::FromWeights, &pool.tensor(&proj_out)?)?,
        bind_transpose_add(gpu, &pool.tensor(&proj_out)?, &pool.tensor(input)?, &pool.tensor(&output)?)?,
    ];
    steps.push(Step::new(format!("{prefix} output"), end).checked(prefix, output, Layout::Image));
    // All steps, in run order.
    Ok(steps)
}

// LayerNorm `number` of the block, then to_q, to_k and to_v of attention `number`, all without bias.
// Self attention reads keys from the normed rows. Cross attention reads them from `text`.
fn attention_inputs(context: &Context<impl WeightBuffers>, block: &str, number: u32, hidden: &str, text: Option<&str>) -> Result<Vec<BoundKernel>, EngineError> {
    // The pool, for short lines.
    let pool = context.pool;
    // For example "...transformer_blocks.0.norm1" and "...transformer_blocks.0.attn1".
    let (norm, name) = (format!("{block}.norm{number}"), format!("{block}.attn{number}"));
    // The source of the keys and values.
    let source = text.map_or(norm.clone(), str::to_string);
    // The tensor names of q, k and v.
    let [query, key, value] = ["to_q", "to_k", "to_v"].map(|part| format!("{name}.{part}"));
    // The four kernels.
    Ok(vec![
        bind_layer_norm(context.gpu, context.weights, &norm, &pool.tensor(hidden)?, &pool.tensor(&norm)?)?,
        bind_rows_linear(context.gpu, context.weights, &query, &pool.tensor(&norm)?, Bias::Zero, &pool.tensor(&query)?)?,
        bind_rows_linear(context.gpu, context.weights, &key, &pool.tensor(&source)?, Bias::Zero, &pool.tensor(&key)?)?,
        bind_rows_linear(context.gpu, context.weights, &value, &pool.tensor(&source)?, Bias::Zero, &pool.tensor(&value)?)?,
    ])
}

// Bind attention `name` with `bind`, on its q, k, v, output and scores scratch, as the plan names them.
// The slots borrow their names, so they live only inside this call.
fn bind_on_attention_slots<T>(context: &Context<impl WeightBuffers>, name: &str, bind: impl FnOnce(&AttentionSlots) -> Result<T, EngineError>) -> Result<T, EngineError> {
    // The pool, for short lines.
    let pool = context.pool;
    // The tensor names of the plan.
    let [query, key, value, output, scores] = ["to_q", "to_k", "to_v", "attention", "attention.scratch"].map(|part| format!("{name}.{part}"));
    // The UNet attentions are not causal.
    bind(&AttentionSlots { query: pool.tensor(&query)?, key: pool.tensor(&key)?, value: pool.tensor(&value)?, scores: pool.tensor(&scores)?, output: pool.tensor(&output)?, causal: false })
}

// The attention kernel of attention `name`, every chunk in one kernel.
fn attention(context: &Context<impl WeightBuffers>, name: &str) -> Result<BoundKernel, EngineError> {
    bind_on_attention_slots(context, name, |slots| bind_attention_into(context.gpu, slots))
}

// The self attention kernel of attention `name`, in groups of SELF_ATTENTION_ROWS_PER_SUBMIT query rows. Each group is one submit.
fn self_attention_groups(context: &Context<impl WeightBuffers>, name: &str) -> Result<Vec<BoundKernel>, EngineError> {
    bind_on_attention_slots(context, name, |slots| bind_attention_groups(context.gpu, slots, SELF_ATTENTION_ROWS_PER_SUBMIT))
}

// to_out.0 of attention `name`: a linear layer with bias.
fn out_projection(context: &Context<impl WeightBuffers>, name: &str) -> Result<BoundKernel, EngineError> {
    // The tensor names of the plan.
    let [mixed, output] = ["attention", "to_out.0"].map(|part| format!("{name}.{part}"));
    // The kernel.
    bind_rows_linear(context.gpu, context.weights, &output, &context.pool.tensor(&mixed)?, Bias::FromWeights, &context.pool.tensor(&output)?)
}

// norm3, ff.net.0.proj, GEGLU, ff.net.2.
fn feed_forward(context: &Context<impl WeightBuffers>, block: &str, hidden: &str) -> Result<Vec<BoundKernel>, EngineError> {
    // The pool, for short lines.
    let pool = context.pool;
    // The tensor names of the plan.
    let [norm, projected, gated, output] = ["norm3", "ff.net.0.proj", "ff.net.0.geglu", "ff.net.2"].map(|part| format!("{block}.{part}"));
    // The four kernels.
    Ok(vec![
        bind_layer_norm(context.gpu, context.weights, &norm, &pool.tensor(hidden)?, &pool.tensor(&norm)?)?,
        bind_rows_linear(context.gpu, context.weights, &projected, &pool.tensor(&norm)?, Bias::FromWeights, &pool.tensor(&projected)?)?,
        bind_geglu(context.gpu, &pool.tensor(&projected)?, &pool.tensor(&gated)?)?,
        bind_rows_linear(context.gpu, context.weights, &output, &pool.tensor(&gated)?, Bias::FromWeights, &pool.tensor(&output)?)?,
    ])
}
