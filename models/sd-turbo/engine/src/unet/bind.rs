// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// A 3x3 convolution between two slots.
use crate::conv::{Bias, bind_conv3x3_into};
// A one-row linear layer between two slots.
use crate::linear::bind_vector_linear;
// The buffers of the activation plan.
use crate::plan::ActivationPool;
// The weight buffers of the slot kernels.
use crate::weights::WeightBuffers;

// The level layout of the plan.
use super::plan::{DOWN_LAYERS, LEVEL_CHANNELS};
// GroupNorm.
use crate::kernels::{NormLayout, bind_group_norm_into};
// The level binders.
use super::down::bind_down_level;
use super::middle::bind_middle;
use super::up::bind_up_level;
// One submit of the stage.
use crate::step::{Layout, Step};

// The tensors of the plan that the CPU writes before a run.
pub(super) const SAMPLE: &str = "sample";
pub(super) const TEXT: &str = "encoder_hidden_states";
pub(super) const TIME_FEATURES: &str = "time_proj";
// The time embedding, which every ResNet block reads.
const TIME: &str = "time_embedding.linear_2";
// The UNet output, which the scheduler step reads.
pub(super) const OUTPUT: &str = "conv_out";
// The eps of conv_norm_out (norm_eps in unet/config.json).
const NORM_EPS: f32 = 1e-5;

// Bind every op of the UNet forward pass to the slots of `pool`, in run order, as a list of steps.
// The order is the one of diffusers' UNet2DConditionModel.forward and of unet/plan/forward.rs.
pub(super) fn bind_unet(gpu: &Gpu, weights: &impl WeightBuffers, pool: &ActivationPool) -> Result<Vec<Step>, EngineError> {
    // linear_1 reads the features as they are. linear_2 applies SiLU to linear_1's output as it reads it.
    let time = vec![
        bind_vector_linear(gpu, weights, "time_embedding.linear_1", &pool.tensor(TIME_FEATURES)?, false, &pool.tensor("time_embedding.linear_1")?)?,
        bind_vector_linear(gpu, weights, "time_embedding.linear_2", &pool.tensor("time_embedding.linear_1")?, true, &pool.tensor(TIME)?)?,
    ];
    // conv_in: 3x3 convolution with bias, stride 1, from the latent to 320 channels.
    let conv_in = vec![bind_conv3x3_into(gpu, weights, "conv_in", &pool.tensor(SAMPLE)?, 1, Bias::FromWeights, &pool.tensor("conv_in")?)?];
    // The time embedding, then conv_in.
    let mut steps = vec![Step::new("time_embedding", time).checked("time_embedding", TIME, Layout::Vector), Step::new("conv_in", conv_in).checked("conv_in", "conv_in", Layout::Image)];
    // The conv_in output is the first skip tensor.
    let mut skips = vec!["conv_in".to_string()];
    // The tensor that flows down.
    let mut hidden = "conv_in".to_string();
    // The four down levels.
    for level in 0..LEVEL_CHANNELS.len() {
        // Bind one level.
        let down = bind_down_level(gpu, weights, pool, level, &hidden, TIME, TEXT)?;
        // Keep its steps and skip tensors.
        steps.extend(down.steps);
        skips.extend(down.skips);
        // Its output goes on.
        hidden = down.output;
    }
    // The middle block.
    let (middle, output) = bind_middle(gpu, weights, pool, &hidden, TIME, TEXT)?;
    steps.extend(middle);
    hidden = output;
    // The up path reads the skip tensors in reverse order, DOWN_LAYERS + 1 per level.
    let reversed: Vec<String> = skips.into_iter().rev().collect();
    for (level, level_skips) in reversed.chunks(DOWN_LAYERS + 1).enumerate() {
        // Bind one level.
        let (up, output) = bind_up_level(gpu, weights, pool, level, &hidden, level_skips, TIME, TEXT)?;
        steps.extend(up);
        hidden = output;
    }
    // conv_norm_out (GroupNorm and SiLU) and conv_out (320 to 4 channels), in one step.
    let end = vec![
        bind_group_norm_into(gpu, weights, "conv_norm_out", &pool.tensor(&hidden)?, true, NORM_EPS, NormLayout::Image, &pool.tensor("conv_norm_out")?)?,
        bind_conv3x3_into(gpu, weights, OUTPUT, &pool.tensor("conv_norm_out")?, 1, Bias::FromWeights, &pool.tensor(OUTPUT)?)?,
    ];
    steps.push(Step::new("conv_out", end).checked("unet_output", OUTPUT, Layout::Image));
    // Every step of the forward pass.
    Ok(steps)
}
