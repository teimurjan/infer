// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// A 3x3 convolution between two slots.
use crate::conv::{Bias, bind_conv1x1_into, bind_conv3x3_into};
// A one-row linear layer between two slots.
use crate::linear::bind_vector_linear;
// The buffers of the activation plan.
use crate::plan::ActivationPool;
// The weight buffers of the slot kernels.
use crate::weights::WeightBuffers;

// GroupNorm and the in-place add.
use crate::kernels::{NormLayout, bind_add, bind_group_norm_into};
// One submit of the stage.
use crate::step::{Layout, Step};

// The eps of every ResNet GroupNorm (norm_eps in unet/config.json).
const NORM_EPS: f32 = 1e-5;

// Read the weights of the ResNet block `prefix` and bind it from the `input` tensor, with the time embedding `time`.
// The ops and the tensor names are the ones of unet/plan/resnet.rs. The output is the tensor "<prefix>.conv2".
// `input` can be a String or a str, because the up path names its joined tensors at run time.
// Two steps, each with one 3x3 convolution, so a submit stays short also on the iPhone. The second step checks the golden `prefix`.
pub(crate) fn bind_resnet(gpu: &Gpu, weights: &impl WeightBuffers, pool: &ActivationPool, prefix: &str, input: &str, time: &str) -> Result<Vec<Step>, EngineError> {
    // The tensor names of the plan.
    let [norm1, time_proj, conv1, norm2, conv2, shortcut] = ["norm1", "time_emb_proj", "conv1", "norm2", "conv2", "conv_shortcut"].map(|part| format!("{prefix}.{part}"));
    // norm1 + SiLU, SiLU + time_emb_proj, conv1, and the time vector added to each channel.
    let first = vec![
        bind_group_norm_into(gpu, weights, &norm1, &pool.tensor(input)?, true, NORM_EPS, NormLayout::Image, &pool.tensor(&norm1)?)?,
        bind_vector_linear(gpu, weights, &time_proj, &pool.tensor(time)?, true, &pool.tensor(&time_proj)?)?,
        bind_conv3x3_into(gpu, weights, &conv1, &pool.tensor(&norm1)?, 1, Bias::FromWeights, &pool.tensor(&conv1)?)?,
        bind_add(gpu, &pool.tensor(&conv1)?, &pool.tensor(&time_proj)?)?,
    ];
    // norm2 + SiLU and conv2.
    let mut second = vec![
        bind_group_norm_into(gpu, weights, &norm2, &pool.tensor(&conv1)?, true, NORM_EPS, NormLayout::Image, &pool.tensor(&norm2)?)?,
        bind_conv3x3_into(gpu, weights, &conv2, &pool.tensor(&norm2)?, 1, Bias::FromWeights, &pool.tensor(&conv2)?)?,
    ];
    // The skip path: the block input, or a 1x1 convolution of it when the plan has one (the channel count changes).
    let skip = if pool.tensor(&shortcut).is_ok() {
        // conv_shortcut from the block input.
        second.push(bind_conv1x1_into(gpu, weights, &shortcut, &pool.tensor(input)?, &pool.tensor(&shortcut)?)?);
        // The add reads the shortcut output.
        shortcut
    } else {
        // The add reads the block input.
        input.to_string()
    };
    // Add the skip path to the conv2 output.
    second.push(bind_add(gpu, &pool.tensor(&conv2)?, &pool.tensor(&skip)?)?);
    // Both steps. The second one shows the block output.
    Ok(vec![Step::new(format!("{prefix} part 1"), first), Step::new(format!("{prefix} part 2"), second).checked(prefix, conv2, Layout::Image)])
}
