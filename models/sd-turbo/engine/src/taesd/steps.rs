// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The tensor type that flows between layers.
use crate::activation::Activation;
// A part of the network that ran.
use crate::block::Block;
// The 3x3 convolution with weights from the weight file.
use crate::conv::{Bias, bind_conv3x3};
// One submit, timed and checked against the 0.25 s limit.
use crate::submit::submit_within_limit;
// The weight reader.
use crate::weights::WeightSource;

// The small TAESD kernels.
use super::kernels;

// The weight name prefix of one decoder layer. The number is the position in diffusers' decoder.layers.
fn layer_name(layer: usize) -> String {
    // For example "decoder.layers.0".
    format!("decoder.layers.{layer}")
}

// Clamp, then decoder.layers.0 (3x3 convolution, 4 to 64 channels), then decoder.layers.1 (ReLU). One submit.
pub(super) async fn conv_in(gpu: &Gpu, weights: &impl WeightSource, latent: Activation) -> Result<Block, EngineError> {
    // tanh(x / 3) * 3 into a new buffer.
    let (clamp, clamped) = kernels::clamp(gpu, &latent)?;
    // The convolution reads the clamped latent.
    let conv = bind_conv3x3(gpu, weights, &layer_name(0), &clamped, 1, Bias::FromWeights)?;
    // ReLU on the convolution output, in place.
    let relu = kernels::relu(gpu, conv.output())?;
    // The three kernels, in order, in one submit.
    let seconds = submit_within_limit(gpu, "taesd conv_in", &[clamp.dispatch().clone(), conv.dispatch().clone(), relu.dispatch().clone()]).await?;
    // The latent and the clamped copy are freed here. Only the output stays.
    Ok(Block { output: conv.into_activation(), submit_seconds: vec![seconds] })
}

// The TAESD blocks at positions `layers` of decoder.layers, one after the other. Each block uses 3 submits.
pub(super) async fn tiny_blocks(gpu: &Gpu, weights: &impl WeightSource, layers: std::ops::RangeInclusive<usize>, input: Activation) -> Result<Block, EngineError> {
    // The activation that goes into the next block.
    let mut hidden = input;
    // The seconds of every submit, in order.
    let mut submit_seconds = Vec::new();
    // One block at a time. Each block frees its input when it is done.
    for layer in layers {
        // Run one block.
        let block = tiny_block(gpu, weights, layer, hidden).await?;
        // Keep its submit times.
        submit_seconds.extend(block.submit_seconds);
        // Its output feeds the next block.
        hidden = block.output;
    }
    // The output of the last block.
    Ok(Block { output: hidden, submit_seconds })
}

// One AutoencoderTinyBlock: ReLU(conv(ReLU(conv(ReLU(conv(x))))) + x). Its convolutions are conv.0, conv.2 and conv.4.
// Each convolution has its own submit, so one submit stays short also at 512 x 512.
async fn tiny_block(gpu: &Gpu, weights: &impl WeightSource, layer: usize, input: Activation) -> Result<Block, EngineError> {
    // For example "decoder.layers.2.conv".
    let prefix = format!("{}.conv", layer_name(layer));
    // conv.0 and ReLU. The input stays, because the block adds it at the end.
    let (first, first_seconds) = conv_relu(gpu, weights, &format!("{prefix}.0"), &input).await?;
    // conv.2 and ReLU.
    let (second, second_seconds) = conv_relu(gpu, weights, &format!("{prefix}.2"), &first).await?;
    // Free the first output before the next convolution. At 512 x 512 it is 67 MB.
    drop(first);
    // conv.4, then add the block input, then ReLU.
    let conv = bind_conv3x3(gpu, weights, &format!("{prefix}.4"), &second, 1, Bias::FromWeights)?;
    // The skip connection and the ReLU, in place on the convolution output.
    let add_relu = kernels::add_relu(gpu, conv.output(), &input.buffer)?;
    // Both kernels in one submit.
    let third_seconds = submit_within_limit(gpu, &format!("{prefix}.4"), &[conv.dispatch().clone(), add_relu.dispatch().clone()]).await?;
    // The input and the second output are freed here. Only the block output stays.
    Ok(Block { output: conv.into_activation(), submit_seconds: vec![first_seconds, second_seconds, third_seconds] })
}

// One 3x3 convolution with a bias, then ReLU in place, in one submit. Returns the output and the seconds of the submit.
async fn conv_relu(gpu: &Gpu, weights: &impl WeightSource, prefix: &str, input: &Activation) -> Result<(Activation, f64), EngineError> {
    // Read the weights and connect the convolution.
    let conv = bind_conv3x3(gpu, weights, prefix, input, 1, Bias::FromWeights)?;
    // ReLU on its output.
    let relu = kernels::relu(gpu, conv.output())?;
    // Both kernels in one submit.
    let seconds = submit_within_limit(gpu, prefix, &[conv.dispatch().clone(), relu.dispatch().clone()]).await?;
    // The weights are freed here.
    Ok((conv.into_activation(), seconds))
}

// nn.Upsample (2x, nearest) at decoder.layers[conv_layer - 1], then the 3x3 convolution without bias at decoder.layers[conv_layer]. One submit.
pub(super) async fn upsample_conv(gpu: &Gpu, weights: &impl WeightSource, conv_layer: usize, input: Activation) -> Result<Block, EngineError> {
    // Twice as high and twice as wide, into a new buffer.
    let (upsample, upsampled) = kernels::upsample(gpu, &input)?;
    // These convolutions have bias=False in diffusers.
    let conv = bind_conv3x3(gpu, weights, &layer_name(conv_layer), &upsampled, 1, Bias::Zero)?;
    // Both kernels in one submit.
    let seconds = submit_within_limit(gpu, &layer_name(conv_layer), &[upsample.dispatch().clone(), conv.dispatch().clone()]).await?;
    // The input and the upsampled copy are freed here.
    Ok(Block { output: conv.into_activation(), submit_seconds: vec![seconds] })
}

// The last 3x3 convolution (64 to 3 channels) at decoder.layers[layer], then x * 2 - 1. One submit.
pub(super) async fn conv_out(gpu: &Gpu, weights: &impl WeightSource, layer: usize, input: Activation) -> Result<Block, EngineError> {
    // bind_conv3x3 pads the 3 output channels to 4 with zeros.
    let conv = bind_conv3x3(gpu, weights, &layer_name(layer), &input, 1, Bias::FromWeights)?;
    // The output size of the convolution.
    let [_, height, width] = conv.out_size();
    // The first 3 channels, mapped to [-1, 1], into a new buffer.
    let (image, rgb) = kernels::image(gpu, conv.output(), height, width)?;
    // Both kernels in one submit.
    let seconds = submit_within_limit(gpu, &layer_name(layer), &[conv.dispatch().clone(), image.dispatch().clone()]).await?;
    // The convolution output is freed here. Only the image stays.
    Ok(Block { output: rgb, submit_seconds: vec![seconds] })
}
