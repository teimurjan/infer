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
use super::plan::{DOWN_LAYERS, UP_NAMES};
// The concat and the upsample.
use crate::kernels::{bind_concat, bind_upsample};
// The block binders.
use super::resnet::bind_resnet;
use super::transformer::bind_transformer;
// One submit of the stage.
use crate::step::{Layout, Step};

// Bind up level `level` from the `input` tensor, with its 3 skip tensors (the newest first).
// Up level 0 (UpBlock2D) has no transformers. Levels 1 to 3 (CrossAttnUpBlock2D) have one after each ResNet block.
// Levels 0 to 2 end with a nearest 2x upsample and a 3x3 convolution. The last step checks the golden of the level, for example "up_block_0".
// Returns the steps and the name of the level output.
pub(crate) fn bind_up_level(gpu: &Gpu, weights: &impl WeightBuffers, pool: &ActivationPool, level: usize, input: &str, skips: &[String], time: &str, text: &str) -> Result<(Vec<Step>, String), EngineError> {
    // For example "up_blocks.1".
    let prefix = format!("up_blocks.{level}");
    // The steps so far.
    let mut steps = Vec::new();
    // The tensor that flows through the level.
    let mut hidden = input.to_string();
    // One ResNet block per skip tensor: DOWN_LAYERS + 1 of them.
    for (layer, skip) in skips.iter().enumerate().take(DOWN_LAYERS + 1) {
        // For example "up_blocks.1.resnets.2".
        let resnet = format!("{prefix}.resnets.{layer}");
        // torch.cat of the up tensor and the skip tensor, in its own step.
        let joined = format!("{resnet}.concat");
        steps.push(Step::new(joined.clone(), vec![bind_concat(gpu, &pool.tensor(&hidden)?, &pool.tensor(skip)?, &pool.tensor(&joined)?)?]));
        // The ResNet block reads the joined tensor.
        steps.extend(bind_resnet(gpu, weights, pool, &resnet, &joined, time)?);
        hidden = format!("{resnet}.conv2");
        // Levels 1 to 3 have a transformer after each ResNet block.
        if level > 0 {
            let transformer = format!("{prefix}.attentions.{layer}");
            steps.extend(bind_transformer(gpu, weights, pool, &transformer, &hidden, text)?);
            hidden = format!("{transformer}.output");
        }
    }
    // Levels 0 to 2 end with the upsample and a 3x3 convolution of stride 1.
    if level < UP_NAMES.len() - 1 {
        // The tensor names of the plan.
        let [interpolate, conv] = ["interpolate", "conv"].map(|part| format!("{prefix}.upsamplers.0.{part}"));
        // Both kernels in one step.
        let kernels = vec![
            bind_upsample(gpu, &pool.tensor(&hidden)?, &pool.tensor(&interpolate)?)?,
            bind_conv3x3_into(gpu, weights, &conv, &pool.tensor(&interpolate)?, 1, Bias::FromWeights, &pool.tensor(&conv)?)?,
        ];
        steps.push(Step::new(format!("{prefix}.upsamplers.0"), kernels));
        hidden = conv;
    }
    // The last step of the level also shows the level output.
    let end = steps.pop().map(|step| step.checked(UP_NAMES[level], hidden.clone(), Layout::Image));
    steps.extend(end);
    // The steps and the output.
    Ok((steps, hidden))
}
