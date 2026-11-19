// Our GPU connection.
use crate::gpu::Gpu;
// The tensor type that flows between layers.
use crate::activation::Activation;

// One part of a network that ran: its output, and the time of each of its submits.
pub(crate) struct Block {
    // The output.
    pub(crate) output: Activation,
    // Seconds per submit, in order.
    pub(crate) submit_seconds: Vec<f64>,
}

// What a network shows the caller after each part that has a golden tensor.
pub struct BlockOutput<'a> {
    // The golden name of the part, for example "conv_in".
    pub name: &'a str,
    // The shape as in PyTorch, with the batch of 1 in front. For example [1, 320, 64, 64], or [1, 1280] for a vector.
    pub shape: Vec<usize>,
    // The output, copied back from the GPU.
    pub values: &'a [f32],
    // How many submits the part used.
    pub submits: usize,
    // The longest of those submits, in seconds.
    pub longest_submit_seconds: f64,
    // The most bytes our GPU buffers held at one time so far.
    pub peak_bytes: u64,
}

// Copy a block output back to the CPU and give it to `check`. Return the values, for a caller that needs them after the check.
pub(crate) async fn check_block<E>(gpu: &Gpu, name: &str, block: &Block, check: &mut impl FnMut(&BlockOutput) -> Result<(), E>) -> Result<Vec<f32>, E>
where
    // A GPU error must be able to become the caller's error type.
    E: From<crate::error::EngineError>,
{
    // [1, channels, height, width], as in PyTorch.
    let shape = [1, block.output.channels as usize, block.output.height as usize, block.output.width as usize];
    // The output buffer holds exactly this tensor.
    check_tensor(gpu, name, &block.output.buffer, &shape, &block.submit_seconds, check).await
}

// Copy the values of a tensor of PyTorch shape `shape` from the start of `buffer` back to the CPU, and give them to `check`.
// The buffer can be larger than the tensor, as a slot of the activation plan is. Return the values after the check.
pub(crate) async fn check_tensor<E>(gpu: &Gpu, name: &str, buffer: &wgpu::Buffer, shape: &[usize], submit_seconds: &[f64], check: &mut impl FnMut(&BlockOutput) -> Result<(), E>) -> Result<Vec<f32>, E>
where
    // A GPU error must be able to become the caller's error type.
    E: From<crate::error::EngineError>,
{
    // Copy only the values of the tensor back.
    let values = gpu.read_f32_prefix(buffer, shape.iter().product()).await?;
    // The longest submit of the block. f64::max ignores NaN, and a time is never NaN.
    let longest_submit_seconds = submit_seconds.iter().copied().fold(0.0, f64::max);
    // Show the block to the caller.
    check(&BlockOutput { name, shape: shape.to_vec(), values: &values, submits: submit_seconds.len(), longest_submit_seconds, peak_bytes: gpu.peak_bytes() })?;
    // The values, after the check passed.
    Ok(values)
}
