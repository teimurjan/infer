// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// Buffers that count their bytes.
use crate::memory::GpuBuffer;
// A kernel ready to record.
use crate::dispatch::{Dispatch, Work};
// The tensor type that flows between layers.
use crate::activation::Activation;
// The shared buffer length check.
use crate::checks::check_buffer;
// The check for sizes that must be positive multiples of 4.
use crate::layers::block_dimension;

// The WGSL sources, copied into the program when it compiles.
const CLAMP: &str = include_str!("clamp.wgsl");
const RELU: &str = include_str!("relu.wgsl");
const ADD_RELU: &str = include_str!("add_relu.wgsl");
// The UNet upsample reuses this shader on its slots.
pub(crate) const UPSAMPLE: &str = include_str!("upsample.wgsl");
const IMAGE: &str = include_str!("image.wgsl");
// Invocations per workgroup. It must match @workgroup_size(256) in every shader of this folder.
const WORKGROUP_SIZE: u32 = 256;
// Values in one vec4.
const VEC4: usize = 4;
// Bytes in one vec4 of f32.
const VEC4_BYTES: u64 = 16;
// The image has red, green and blue.
const RGB: u32 = 3;
// The last convolution pads 3 output channels to 4, because its kernel computes 4 channels at a time.
const PADDED_RGB: u32 = 4;

// A small kernel connected to its buffers, ready to record.
pub(super) struct Small {
    // The kernel with its buffers.
    dispatch: Dispatch,
    // The sizes buffer. We keep it, because the browser frees a buffer when Rust drops it.
    _uniform: GpuBuffer,
}

impl Small {
    // The kernel, ready to record.
    pub(super) fn dispatch(&self) -> &Dispatch {
        &self.dispatch
    }
}

// clamp.wgsl: a new activation with tanh(x / 3) * 3.
pub(super) fn clamp(gpu: &Gpu, input: &Activation) -> Result<(Small, Activation), EngineError> {
    // The output has the same shape as the input.
    let output = gpu.empty_storage_buffer("taesd clamp", input.value_count());
    // One invocation per vec4.
    let count = vec4_count("clamp input", input.value_count())?;
    // Connect the sizes, the input and the output.
    let kernel = bind(gpu, "taesd clamp", CLAMP, &[count], &[&input.buffer, &output], 2 * u64::from(count))?;
    // The kernel and the activation it fills.
    Ok((kernel, Activation { buffer: output, channels: input.channels, height: input.height, width: input.width }))
}

// relu.wgsl: ReLU on `values`, in place.
pub(super) fn relu(gpu: &Gpu, values: &wgpu::Buffer) -> Result<Small, EngineError> {
    // One invocation per vec4 of the buffer.
    let count = buffer_vec4_count("relu values", values)?;
    // Connect the sizes and the buffer.
    bind(gpu, "taesd relu", RELU, &[count], &[values], 2 * u64::from(count))
}

// add_relu.wgsl: values = ReLU(values + skip), in place.
pub(super) fn add_relu(gpu: &Gpu, values: &wgpu::Buffer, skip: &wgpu::Buffer) -> Result<Small, EngineError> {
    // One invocation per vec4 of the buffer.
    let count = buffer_vec4_count("add_relu values", values)?;
    // The skip tensor must have exactly as many values. A shorter one would be read past its end.
    check_buffer("add_relu skip", skip, count as usize * VEC4)?;
    // Connect the sizes, the skip and the values, in binding order.
    bind(gpu, "taesd add_relu", ADD_RELU, &[count], &[skip, values], 3 * u64::from(count))
}

// upsample.wgsl: a new activation twice as high and twice as wide.
pub(super) fn upsample(gpu: &Gpu, input: &Activation) -> Result<(Small, Activation), EngineError> {
    // The shader reads 2 input pixels per output vec4, so the input width must be even. A multiple of 4 is what every activation has.
    let in_width = block_dimension("upsample input width", input.width)?;
    // The output size.
    let output = Activation { buffer: gpu.empty_storage_buffer("taesd upsample", input.value_count() * 4), channels: input.channels, height: input.height * 2, width: in_width * 2 };
    // One invocation per output vec4.
    let count = vec4_count("upsample output", output.value_count())?;
    // Connect the sizes, the input and the output.
    // It reads the input, a quarter of the output, and writes the output.
    let kernel = bind(gpu, "taesd upsample", UPSAMPLE, &[count, in_width], &[&input.buffer, &output.buffer], u64::from(count) + u64::from(count) / 4)?;
    // The kernel and the activation it fills.
    Ok((kernel, output))
}

// image.wgsl: the first 3 channels of the last convolution, mapped from [0, 1] to [-1, 1].
pub(super) fn image(gpu: &Gpu, input: &wgpu::Buffer, height: u32, width: u32) -> Result<(Small, Activation), EngineError> {
    // The last convolution writes 4 channels: red, green, blue and a channel of zeros.
    check_buffer("image input", input, (PADDED_RGB * height * width) as usize)?;
    // The output has 3 channels.
    let output = Activation { buffer: gpu.empty_storage_buffer("taesd image", (RGB * height * width) as usize), channels: RGB, height, width };
    // One invocation per output vec4. The input is longer, and the kernel never reads its last channel.
    let count = vec4_count("image output", output.value_count())?;
    // Connect the sizes, the input and the output.
    // It reads 3 of the 4 channels, as many values as it writes.
    let kernel = bind(gpu, "taesd image", IMAGE, &[count], &[input, &output.buffer], 2 * u64::from(count))?;
    // The kernel and the activation it fills.
    Ok((kernel, output))
}

// Make the sizes buffer, and connect it to @binding(0) and `buffers` to the next bindings, in order.
// `vec4s` is the number of vec4 values that one run reads and writes in total, for the timing table.
fn bind(gpu: &Gpu, label: &'static str, shader: &str, sizes: &[u32], buffers: &[&wgpu::Buffer], vec4s: u64) -> Result<Small, EngineError> {
    // The first size is always the number of vec4s, one per invocation.
    let workgroups = sizes[0].div_ceil(WORKGROUP_SIZE);
    // The sizes go to @binding(0).
    let uniform = gpu.uniform_buffer(label, sizes);
    // The sizes first, then the buffers.
    let bindings: Vec<&wgpu::Buffer> = std::iter::once(&*uniform).chain(buffers.iter().copied()).collect();
    // Compile the shader and connect everything. Fails if the workgroups do not fit in one dispatch.
    let dispatch = Dispatch::new(gpu, label, shader, &bindings, [workgroups, 1, 1], Work { kernel: label, bytes: vec4s * VEC4_BYTES })?;
    // Keep the sizes buffer with the kernel.
    Ok(Small { dispatch, _uniform: uniform })
}

// The number of vec4s in `values` values. The count must be a multiple of 4 and fit in u32.
fn vec4_count(tensor: &'static str, values: usize) -> Result<u32, EngineError> {
    // The number of whole vec4s.
    let count = values / VEC4;
    // A remainder would be lost. Every activation has a width that is a multiple of 4, so this does not happen.
    if count * VEC4 != values {
        return Err(EngineError::TensorLength { tensor, actual: values, expected: count * VEC4 });
    }
    // A count above u32 does not fit in the sizes buffer. In WASM, usize is 32 bits, so the limit saturates instead of overflowing.
    u32::try_from(count).map_err(|_| EngineError::TooManyValues { count: values, max: (u32::MAX as usize).saturating_mul(VEC4) })
}

// The number of vec4s in a GPU buffer of f32 values.
fn buffer_vec4_count(tensor: &'static str, buffer: &wgpu::Buffer) -> Result<u32, EngineError> {
    // A buffer is never larger than memory, so the value count fits in usize.
    vec4_count(tensor, (buffer.size() / (VEC4_BYTES / VEC4 as u64)) as usize)
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function and the error into the tests.
    use super::{EngineError, vec4_count};

    // A count that is not a multiple of 4 would leave the last values unprocessed, so it must fail.
    #[test]
    fn vec4_count_needs_whole_vec4s() {
        // 64 x 64 x 64 values are 65536 vec4s.
        assert_eq!(vec4_count("values", 262_144).unwrap(), 65_536);
        // 6 values are one vec4 and 2 left over.
        assert!(matches!(vec4_count("values", 6), Err(EngineError::TensorLength { actual: 6, expected: 4, .. })));
    }
}
