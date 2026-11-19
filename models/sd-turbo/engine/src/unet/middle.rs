// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The buffers of the activation plan.
use crate::plan::ActivationPool;
// The weight buffers of the slot kernels.
use crate::weights::WeightBuffers;

// The block binders.
use super::resnet::bind_resnet;
use super::transformer::bind_transformer;
// One submit of the stage.
use crate::step::{Layout, Step};

// The name of the middle block, in diffusers and in the golden set.
const PREFIX: &str = "mid_block";

// UNetMidBlock2DCrossAttn: ResNet block, transformer, ResNet block. Returns the steps and the name of the output tensor.
// The last step checks the golden "mid_block".
pub(crate) fn bind_middle(gpu: &Gpu, weights: &impl WeightBuffers, pool: &ActivationPool, input: &str, time: &str, text: &str) -> Result<(Vec<Step>, String), EngineError> {
    // The steps of the first ResNet block.
    let mut steps = bind_resnet(gpu, weights, pool, &format!("{PREFIX}.resnets.0"), input, time)?;
    // The transformer reads its output.
    steps.extend(bind_transformer(gpu, weights, pool, &format!("{PREFIX}.attentions.0"), &format!("{PREFIX}.resnets.0.conv2"), text)?);
    // The second ResNet block reads the transformer output.
    steps.extend(bind_resnet(gpu, weights, pool, &format!("{PREFIX}.resnets.1"), &format!("{PREFIX}.attentions.0.output"), time)?);
    // The output of the middle block.
    let output = format!("{PREFIX}.resnets.1.conv2");
    // The last step also shows the golden of the whole block.
    let end = steps.pop().map(|step| step.checked(PREFIX, output.clone(), Layout::Image));
    steps.extend(end);
    // The steps and the output.
    Ok((steps, output))
}
