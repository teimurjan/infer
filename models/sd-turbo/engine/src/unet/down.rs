// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// A 3x3 convolution between two slots.
use crate::conv::{Bias, bind_conv3x3_into};
// The buffers of the activation plan.
use crate::plan::ActivationPool;
// The weight buffers of the slot kernels.
use crate::weights::WeightBuffers;

// The level layout of the plan.
use super::plan::{DOWN_LAYERS, DOWN_NAMES, LEVEL_CHANNELS};
// The block binders.
use super::resnet::bind_resnet;
use super::transformer::bind_transformer;
// One submit of the stage.
use crate::step::{Layout, Step};

// The steps of one down level, the tensor that goes on, and the skip tensors that the up path reads.
pub(crate) struct DownLevel {
    // Every submit of the level, in run order.
    pub(crate) steps: Vec<Step>,
    // The name of the level output in the plan.
    pub(crate) output: String,
    // The names of the skip tensors of this level, in the order they were made.
    pub(crate) skips: Vec<String>,
}

// Bind down level `level` from the `input` tensor, with the time embedding `time` and the text `text`.
// Levels 0 to 2 (CrossAttnDownBlock2D): 2 pairs of ResNet block and transformer, then a 3x3 convolution of stride 2.
// Level 3 (DownBlock2D): 2 ResNet blocks only. The last step checks the golden of the level, for example "down_block_0".
pub(crate) fn bind_down_level(gpu: &Gpu, weights: &impl WeightBuffers, pool: &ActivationPool, level: usize, input: &str, time: &str, text: &str) -> Result<DownLevel, EngineError> {
    // For example "down_blocks.0".
    let prefix = format!("down_blocks.{level}");
    // The last level has no transformers and no downsample.
    let last = level == LEVEL_CHANNELS.len() - 1;
    // The steps so far.
    let mut steps = Vec::new();
    // The tensor that flows through the level.
    let mut hidden = input.to_string();
    // The skip tensors: the output of each layer, and the downsample output.
    let mut skips = Vec::new();
    // Each layer: a ResNet block, then a transformer on levels 0 to 2.
    for layer in 0..DOWN_LAYERS {
        // For example "down_blocks.0.resnets.1".
        let resnet = format!("{prefix}.resnets.{layer}");
        steps.extend(bind_resnet(gpu, weights, pool, &resnet, &hidden, time)?);
        // The ResNet output is the tensor "<resnet>.conv2".
        hidden = format!("{resnet}.conv2");
        // The transformer, except on the last level.
        if !last {
            // For example "down_blocks.0.attentions.1".
            let transformer = format!("{prefix}.attentions.{layer}");
            steps.extend(bind_transformer(gpu, weights, pool, &transformer, &hidden, text)?);
            // The transformer output is the tensor "<transformer>.output".
            hidden = format!("{transformer}.output");
        }
        // The up path reads the output of each layer.
        skips.push(hidden.clone());
    }
    // Levels 0 to 2 end with the downsample: a 3x3 convolution with stride 2 and padding 1.
    if !last {
        // For example "down_blocks.0.downsamplers.0.conv".
        let downsample = format!("{prefix}.downsamplers.0.conv");
        let kernel = bind_conv3x3_into(gpu, weights, &downsample, &pool.tensor(&hidden)?, 2, Bias::FromWeights, &pool.tensor(&downsample)?)?;
        steps.push(Step::new(downsample.clone(), vec![kernel]));
        // The downsample output goes on, and the up path reads it too.
        hidden = downsample;
        skips.push(hidden.clone());
    }
    // The last step of the level also ends the level, so it shows the level output.
    let end = steps.pop().map(|step| step.checked(DOWN_NAMES[level], hidden.clone(), Layout::Image));
    // Put it back.
    steps.extend(end);
    // The steps, the output and the skip tensors of the level.
    Ok(DownLevel { steps, output: hidden, skips })
}
