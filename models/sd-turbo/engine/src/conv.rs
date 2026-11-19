// The kinds of GPU buffer that the memory counter keeps apart.
use crate::memory::MemoryCategory;
// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// Buffers that count their bytes.
use crate::memory::GpuBuffer;
// The kernel ready to record, and a kernel with its buffers.
use crate::dispatch::{BoundKernel, Dispatch};
// The tensor type that flows between layers.
use crate::activation::Activation;
// The 3x3 convolution kernel, connected to GPU buffers.
use crate::layers::{BoundLayer, Conv1x1Shape, Conv3x3Shape, LayerShape, bind_layer, bind_matrix_into, conv3x3_output_size};
// The weight reader, and the weight buffers of the slot kernels.
use crate::weights::{WeightBuffer, WeightBuffers, WeightSource};
// A tensor in a slot of the activation plan.
use crate::plan::SlotTensor;

// Size of one f32 value in bytes.
const F32_BYTES: usize = 4;
// The kernel computes 4 output channels per invocation, so the output channel count must be a multiple of 4.
const CHANNEL_BLOCK: usize = 4;

// Where the bias of a convolution comes from.
#[derive(Clone, Copy)]
pub(crate) enum Bias {
    // Read `<prefix>.bias` from the weight file.
    FromWeights,
    // The layer has no bias (bias=False in PyTorch). The kernel adds zeros.
    Zero,
}

// A 3x3 convolution with its weights on the GPU, ready to record.
pub(crate) struct Conv {
    // The kernel with its buffers.
    layer: BoundLayer,
    // The weights and the bias. We keep them until the submit is done, because the browser frees a dropped buffer.
    _weights: [GpuBuffer; 2],
    // Output channels, height and width.
    out: [u32; 3],
}

impl Conv {
    // The kernel, ready to record.
    pub(crate) fn dispatch(&self) -> &Dispatch {
        self.layer.dispatch()
    }

    // The buffer the kernel writes, so that another kernel in the same submit can read or change it.
    pub(crate) fn output(&self) -> &GpuBuffer {
        self.layer.output()
    }

    // Output channels (padded to a multiple of 4), height and width.
    pub(crate) fn out_size(&self) -> [u32; 3] {
        self.out
    }

    // The output as an activation. Call this after the submit. It drops the weights.
    pub(crate) fn into_activation(self) -> Activation {
        // Unpack the output size.
        let [channels, height, width] = self.out;
        // The output buffer and its shape.
        Activation { buffer: self.layer.into_output(), channels, height, width }
    }
}

// The weights and the bias of one 3x3 convolution, on the GPU.
pub(crate) struct Conv3x3Weights {
    // [out][in][3][3], with zero rows for the padded output channels.
    weight: GpuBuffer,
    // One bias per output channel, padded the same way.
    bias: GpuBuffer,
    // The output channel count, padded to a multiple of 4.
    out_channels: u32,
}

// Read `<prefix>.weight` and the bias, check them against the input channel count, and copy them to the GPU.
// The output channel count comes from the weight shape, so the caller never repeats it.
// If that count is not a multiple of 4, it is padded with zero channels. TAESD's last convolution has 3 channels, so its output gets a 4th channel of zeros.
pub(crate) fn load_conv3x3(gpu: &Gpu, weights: &impl WeightSource, prefix: &str, in_channels: u32, bias: Bias) -> Result<Conv3x3Weights, EngineError> {
    // For example "conv_in.weight".
    let weight_name = format!("{prefix}.weight");
    // Read the shape and the bytes.
    let weight = weights.read(&weight_name)?;
    // PyTorch's layout is [out_channels, in_channels, 3, 3], and in_channels must be the input's channel count.
    let out_channels = match weight.shape[..] {
        // The right rank, kernel size and input channels.
        [out_channels, weight_in, 3, 3] if weight_in == in_channels as usize => out_channels,
        // Anything else cannot run on this input.
        _ => return Err(EngineError::WeightShape { name: weight_name, shape: weight.shape, expected: format!("[out, {in_channels}, 3, 3]") }),
    };
    // The channel count that the kernel computes, for example 3 -> 4.
    let padded_channels = out_channels.next_multiple_of(CHANNEL_BLOCK);
    // The extra output channels go at the end of [out][in][3][3], so zero bytes at the end give them zero weights.
    let weight_bytes = zero_padded(weight.bytes, padded_channels * in_channels as usize * 9);
    // The bias bytes, padded the same way.
    let bias_bytes = bias_bytes(weights, prefix, bias, out_channels, padded_channels)?;
    // A channel count that does not fit in u32 cannot come from a real model.
    let padded_u32 = u32::try_from(padded_channels).map_err(|_| EngineError::WeightShape { name: weight_name.clone(), shape: vec![out_channels], expected: "fewer than 2^32 output channels".to_string() })?;
    // Copy the weights to the GPU. The buffer name is the tensor name, for GPU error messages.
    let weight_buffer = gpu.storage_buffer_from_bytes(&weight_name, &weight_bytes, MemoryCategory::Weights);
    // Copy the bias to the GPU.
    let bias_buffer = gpu.storage_buffer_from_bytes(&format!("{prefix}.bias"), &bias_bytes, MemoryCategory::Weights);
    // Both buffers and the padded channel count.
    Ok(Conv3x3Weights { weight: weight_buffer, bias: bias_buffer, out_channels: padded_u32 })
}

// Read the weights of a 3x3 convolution and connect it to `input`. It makes its own output buffer.
pub(crate) fn bind_conv3x3(gpu: &Gpu, weights: &impl WeightSource, prefix: &str, input: &Activation, stride: u32, bias: Bias) -> Result<Conv, EngineError> {
    // The weights on the GPU.
    let loaded = load_conv3x3(gpu, weights, prefix, input.channels, bias)?;
    // The sizes of this convolution.
    let shape = Conv3x3Shape { in_channels: input.channels, out_channels: loaded.out_channels, height: input.height, width: input.width, stride };
    // Check the sizes and connect the buffers.
    let layer = bind_layer(gpu, &LayerShape::Conv3x3(shape), &input.buffer, &loaded.weight, &loaded.bias)?;
    // bind_layer checked that height and width are 1 or more, so the size formula is safe here.
    let out = [loaded.out_channels, conv3x3_output_size(input.height, stride), conv3x3_output_size(input.width, stride)];
    // Keep the weights with the kernel.
    Ok(Conv { layer, _weights: [loaded.weight, loaded.bias], out })
}

// Read the weights of a 3x3 convolution and connect it from the `input` slot to the `output` slot.
// The weight can be float or packed. The output size that the weights give must be the size that the plan has for `output`.
pub(crate) fn bind_conv3x3_into(gpu: &Gpu, weights: &impl WeightBuffers, prefix: &str, input: &SlotTensor, stride: u32, bias: Bias, output: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // The input size.
    let [in_channels, height, width] = input.dims;
    // The weight, as the weights give it.
    let weight = weights.matrix(gpu, &format!("{prefix}.weight"))?;
    // PyTorch's layout is [out_channels, in_channels, 3, 3], and in_channels must be the input's channel count.
    let out_channels = match weight.shape[..] {
        // The right rank, kernel size and input channels.
        [out_channels, weight_in, 3, 3] if weight_in == in_channels as usize => out_channels,
        // Anything else cannot run on this input.
        _ => return Err(EngineError::WeightShape { name: format!("{prefix}.weight"), shape: weight.shape, expected: format!("[out, {in_channels}, 3, 3]") }),
    };
    // The output size that the kernel will write. A count that does not fit in u32 cannot match the plan.
    let actual = [u32::try_from(out_channels).unwrap_or(u32::MAX), conv3x3_output_size(height, stride), conv3x3_output_size(width, stride)];
    // A plan with other sizes would put the tensor into a slot that is too small, or read it with the wrong shape.
    if actual != output.dims {
        return Err(EngineError::PlanMismatch { tensor: output.name.to_string(), planned: output.dims, actual });
    }
    // The bias.
    let bias = slot_bias(gpu, weights, prefix, bias, out_channels)?;
    // The sizes of this convolution.
    let shape = LayerShape::Conv3x3(Conv3x3Shape { in_channels, out_channels: actual[0], height, width, stride });
    // Check the sizes and connect the slots, with the float or the packed kernel. This is the only bind group of the kernel.
    Ok(bind_matrix_into(gpu, &shape, input.buffer, weight.matrix, bias.buffer(), output.buffer)?.keeping(bias.into_owned()))
}

// The bias of a slot layer with `out_channels` outputs: `<prefix>.bias` from the weights, or zeros made for this kernel.
pub(crate) fn slot_bias<'a>(gpu: &Gpu, weights: &'a impl WeightBuffers, prefix: &str, bias: Bias, out_channels: usize) -> Result<WeightBuffer<'a>, EngineError> {
    // The two kinds of bias.
    match bias {
        // Read it and check its shape.
        Bias::FromWeights => {
            // For example "conv_in.bias".
            let name = format!("{prefix}.bias");
            // The f32 values.
            let tensor = weights.float(gpu, &name)?;
            // One bias per output channel.
            if tensor.shape != [out_channels] {
                return Err(EngineError::WeightShape { name, shape: tensor.shape, expected: format!("[{out_channels}]") });
            }
            // The buffer.
            Ok(tensor.buffer)
        }
        // bias=False in PyTorch. The kernel adds zeros from a buffer of its own.
        Bias::Zero => Ok(WeightBuffer::Owned(gpu.storage_buffer_from_bytes(&format!("{prefix}.bias zeros"), &vec![0; out_channels * F32_BYTES], MemoryCategory::Weights))),
    }
}

// Read `<prefix>.weight` [out, in, 1, 1] and `<prefix>.bias`, and connect a 1x1 convolution from the `input` slot to the `output` slot.
// The UNet uses it for conv_shortcut, the skip path of a ResNet block that changes the channel count. The weight can be float or packed.
pub(crate) fn bind_conv1x1_into(gpu: &Gpu, weights: &impl WeightBuffers, prefix: &str, input: &SlotTensor, output: &SlotTensor) -> Result<BoundKernel, EngineError> {
    // The input size. A 1x1 kernel keeps the height and the width.
    let [in_channels, height, width] = input.dims;
    // The weight, as the weights give it.
    let weight = weights.matrix(gpu, &format!("{prefix}.weight"))?;
    // PyTorch's layout is [out_channels, in_channels, 1, 1].
    let out_channels = match weight.shape[..] {
        // The right rank, kernel size and input channels.
        [out_channels, weight_in, 1, 1] if weight_in == in_channels as usize => out_channels,
        // Anything else cannot run on this input.
        _ => return Err(EngineError::WeightShape { name: format!("{prefix}.weight"), shape: weight.shape, expected: format!("[out, {in_channels}, 1, 1]") }),
    };
    // The output size must be the planned size. A count that does not fit in u32 cannot match.
    let actual = [u32::try_from(out_channels).unwrap_or(u32::MAX), height, width];
    if output.dims != actual {
        return Err(EngineError::PlanMismatch { tensor: output.name.to_string(), planned: output.dims, actual });
    }
    // The bias.
    let bias = slot_bias(gpu, weights, prefix, Bias::FromWeights, out_channels)?;
    // The sizes of this convolution.
    let shape = LayerShape::Conv1x1(Conv1x1Shape { in_channels, out_channels: actual[0], height, width });
    // Check the sizes and connect the slots, with the float or the packed kernel.
    Ok(bind_matrix_into(gpu, &shape, input.buffer, weight.matrix, bias.buffer(), output.buffer)?.keeping(bias.into_owned()))
}

// The bias as bytes, `padded_channels` values long.
fn bias_bytes(weights: &impl WeightSource, prefix: &str, bias: Bias, out_channels: usize, padded_channels: usize) -> Result<Vec<u8>, EngineError> {
    // The two kinds of bias.
    match bias {
        // All zeros.
        Bias::Zero => Ok(vec![0; padded_channels * F32_BYTES]),
        // Read it from the file.
        Bias::FromWeights => {
            // For example "conv_in.bias".
            let name = format!("{prefix}.bias");
            // Read the shape and the bytes.
            let tensor = weights.read(&name)?;
            // One bias per output channel.
            if tensor.shape != [out_channels] {
                return Err(EngineError::WeightShape { name, shape: tensor.shape, expected: format!("[{out_channels}]") });
            }
            // Zero biases for the padded channels.
            Ok(zero_padded(tensor.bytes, padded_channels))
        }
    }
}

// The bytes of `values` f32 values: the given bytes, then zero bytes up to that length.
fn zero_padded(mut bytes: Vec<u8>, values: usize) -> Vec<u8> {
    // resize adds zeros at the end. The length is never shorter, because the padded count is never smaller.
    bytes.resize(values * F32_BYTES, 0);
    // The longer list.
    bytes
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the helper into the tests.
    use super::zero_padded;

    // Padding must keep the real values in front and add whole zero values at the end.
    #[test]
    fn zero_padding_keeps_values_in_front() {
        // One f32 (4 bytes) padded to 2 values.
        assert_eq!(zero_padded(vec![1, 2, 3, 4], 2), vec![1, 2, 3, 4, 0, 0, 0, 0]);
        // Already long enough: nothing changes.
        assert_eq!(zero_padded(vec![1, 2, 3, 4], 1), vec![1, 2, 3, 4]);
    }
}
