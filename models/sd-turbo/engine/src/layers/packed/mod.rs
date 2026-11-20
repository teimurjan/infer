// The packed weight format, written by pack_weights.py in the project folder:
// one row of signed integers per output channel, packed into u32 words lowest bits first,
// each row padded with zero bits to a whole word, and one float16 scale per output channel.

// The kinds of GPU buffer that the memory counter keeps apart.
use crate::memory::MemoryCategory;
// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;

// conv3x3.rs holds the workgroup grid of the packed 3x3 convolution, which differs from the float one.
mod conv3x3;
// linear.rs holds the workgroup grid of the packed linear layer, which differs from the float one.
mod linear;

// The workgroup grid of the packed 3x3 convolution.
use conv3x3::conv3x3_workgroups;
// The workgroup grid of the packed linear layer.
use linear::linear_workgroups;
// The float layer plans, the binding helpers, and the prepared layer from the parent module.
use super::{Binding, LayerPlan, LayerShape, PreparedLayer, bind_plan, connect, plan_layer};
// The one-row layer plan.
use super::vector::{VectorShape, plan_vector};
// A kernel with its buffers.
use crate::dispatch::BoundKernel;
// The shared buffer length checks.
use crate::checks::{check_buffer, check_buffer_holds};

// The WGSL source of each packed kernel, copied into the program when it compiles.
const LINEAR_SHADER: &str = include_str!("linear.wgsl");
const CONV1X1_SHADER: &str = include_str!("conv1x1.wgsl");
const CONV3X3_SHADER: &str = include_str!("conv3x3.wgsl");
const VECTOR_SHADER: &str = include_str!("vector.wgsl");
// Bits in one packed word.
const WORD_BITS: usize = 32;
// Two float16 scales share one u32 word.
const SCALES_PER_WORD: usize = 2;

// How many bits each packed weight has. The kernels need 4 * bits to fill a word exactly, so only 4 and 8 work.
#[derive(Clone, Copy)]
pub enum WeightBits {
    // Integers from -8 to 7.
    Four,
    // Integers from -128 to 127.
    Eight,
}

impl WeightBits {
    // The number of bits, as the kernels read it.
    pub fn value(self) -> u32 {
        // One number per variant.
        match self {
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

// The tensors of one packed layer, as raw file bytes or f32 values.
pub struct PackedTensors<'a> {
    // The layer input, in PyTorch's layout.
    pub input: &'a [f32],
    // The packed words, 4 bytes each, lowest byte first.
    pub weight: &'a [u8],
    // One float16 scale per output channel, 2 bytes each, lowest byte first.
    pub scale: &'a [u8],
    // One f32 bias per output channel.
    pub bias: &'a [f32],
}

// The plan of a packed layer: the float plan with the packed kernel, plus the word counts of the packed buffers.
struct PackedPlan {
    // Sizes, workgroups and kernel.
    layer: LayerPlan,
    // u32 words in the weight buffer.
    weight_words: usize,
    // u32 words in the scale buffer.
    scale_words: usize,
}

// Plan a packed layer from the float plan of the same shape. Only the weight and scale buffers differ.
fn plan_packed(shape: &LayerShape, bits: WeightBits) -> Result<PackedPlan, EngineError> {
    // The float plan checks every size, and gives the lengths and the workgroups.
    let float_plan = plan_layer(shape)?;
    // One bias per output channel.
    let out_channels = float_plan.lengths.bias;
    // Weights of one output channel: in_features, or in_channels x 3 x 3.
    let row_values = float_plan.lengths.weight / out_channels;
    // Words per row, rounded up, because each row starts on a word.
    let row_words = (row_values * bits.value() as usize).div_ceil(WORD_BITS);
    // The packed kernel, its label and its workgroups, for each layer kind.
    let (label, shader, workgroups) = match shape {
        // Matrix multiply plus bias. Its own grid, because each invocation takes 8 rows, not 4.
        LayerShape::Linear(linear) => ("linear_packed", LINEAR_SHADER, linear_workgroups(linear)),
        // 1x1 convolution. Same grid as the float kernel.
        LayerShape::Conv1x1(_) => ("conv1x1_packed", CONV1X1_SHADER, float_plan.workgroups),
        // 3x3 convolution. Its own grid, because its workgroup has another shape.
        LayerShape::Conv3x3(conv) => ("conv3x3_packed", CONV3X3_SHADER, conv3x3_workgroups(conv)),
    };
    // The packed Shape structs add bits and row_words after the float fields.
    let sizes = float_plan.sizes.iter().copied().chain([bits.value(), row_words as u32]).collect();
    // Every output channel count is a multiple of 4, so the scales fill whole words.
    let scale_words = out_channels / SCALES_PER_WORD;
    // The packed plan.
    Ok(PackedPlan { layer: LayerPlan { label, shader, sizes, workgroups, ..float_plan }, weight_words: out_channels * row_words, scale_words })
}

// Plan the packed one-row layer from the float one-row plan.
fn plan_packed_vector(shape: &VectorShape, bits: WeightBits) -> Result<PackedPlan, EngineError> {
    // The float plan checks the sizes, and gives the lengths and the workgroups.
    let float_plan = plan_vector(shape)?;
    // Two scales share a word, so the output count must be even.
    if !shape.out_features.is_multiple_of(SCALES_PER_WORD as u32) {
        return Err(EngineError::BadDimension { dimension: "packed out_features", value: shape.out_features });
    }
    // Words per row, rounded up.
    let row_words = (shape.in_features as usize * bits.value() as usize).div_ceil(WORD_BITS);
    // The packed Shape struct adds bits and row_words after the float fields.
    let sizes = float_plan.sizes.iter().copied().chain([bits.value(), row_words as u32]).collect();
    // The packed plan.
    let out_features = shape.out_features as usize;
    Ok(PackedPlan { layer: LayerPlan { label: "vector_packed", shader: VECTOR_SHADER, sizes, ..float_plan }, weight_words: out_features * row_words, scale_words: out_features / SCALES_PER_WORD })
}

// Connect a packed layer from the `input` slot to the `output` slot, with the packed words, scales and bias that the caller keeps.
pub(crate) fn bind_packed_into(gpu: &Gpu, shape: &LayerShape, bits: WeightBits, input: &wgpu::Buffer, words: &wgpu::Buffer, scale: &wgpu::Buffer, bias: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // Check the sizes and pick the kernel.
    bind_packed_plan(gpu, &plan_packed(shape, bits)?, input, words, scale, bias, output)
}

// The same for the one-row layer of the time embedding.
pub(crate) fn bind_packed_vector_into(gpu: &Gpu, shape: &VectorShape, bits: WeightBits, input: &wgpu::Buffer, words: &wgpu::Buffer, scale: &wgpu::Buffer, bias: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // Check the sizes and pick the kernel.
    bind_packed_plan(gpu, &plan_packed_vector(shape, bits)?, input, words, scale, bias, output)
}

// Check the buffers of a packed plan and connect its kernel. Bindings: 0 sizes, 1 input, 2 words, 3 scales, 4 bias, 5 output.
fn bind_packed_plan(gpu: &Gpu, plan: &PackedPlan, input: &wgpu::Buffer, words: &wgpu::Buffer, scale: &wgpu::Buffer, bias: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // A slot can be larger than its tensor.
    check_buffer_holds("input", input, plan.layer.lengths.input)?;
    check_buffer_holds("output", output, plan.layer.lengths.output)?;
    // The weight buffers hold one tensor each, so their length must be exact.
    check_buffer("packed weight", words, plan.weight_words)?;
    check_buffer("scale", scale, plan.scale_words)?;
    check_buffer("bias", bias, plan.layer.lengths.bias)?;
    // The four tensors, in binding order 1 to 4.
    let inputs = [
        Binding { name: "input", buffer: input, values: plan.layer.lengths.input },
        Binding { name: "packed weight", buffer: words, values: plan.weight_words },
        Binding { name: "scale", buffer: scale, values: plan.scale_words },
        Binding { name: "bias", buffer: bias, values: plan.layer.lengths.bias },
    ];
    // Connect the kernel.
    let (dispatch, uniform) = connect(gpu, &plan.layer, &inputs, output)?;
    // The kernel with its uniform.
    Ok(BoundKernel::new(vec![dispatch], vec![uniform]))
}

// Copy the tensors to the GPU, check the sizes, and compile the packed kernel for this layer kind.
pub(crate) fn prepare_packed_layer(gpu: &Gpu, shape: &LayerShape, bits: WeightBits, tensors: &PackedTensors) -> Result<PreparedLayer, EngineError> {
    // Check the sizes and pick the kernel.
    let plan = plan_packed(shape, bits)?;
    // The input goes to the GPU as it is.
    let input = gpu.storage_buffer_from("packed layer input", tensors.input, MemoryCategory::Activations);
    // The packed words, straight from the file bytes.
    let weight = gpu.storage_buffer_from_bytes("packed layer weight", tensors.weight, MemoryCategory::Weights);
    // The float16 scales, straight from the file bytes.
    let scale = gpu.storage_buffer_from_bytes("packed layer scale", tensors.scale, MemoryCategory::Weights);
    // The f32 bias.
    let bias = gpu.storage_buffer_from("packed layer bias", tensors.bias, MemoryCategory::Weights);
    // The four tensors, in binding order 1 to 4. The output is binding 5.
    let inputs = [
        Binding { name: "input", buffer: &input, values: plan.layer.lengths.input },
        Binding { name: "packed weight", buffer: &weight, values: plan.weight_words },
        Binding { name: "scale", buffer: &scale, values: plan.scale_words },
        Binding { name: "bias", buffer: &bias, values: plan.layer.lengths.bias },
    ];
    // Check the lengths and connect the kernel.
    let layer = bind_plan(gpu, &plan.layer, &inputs)?;
    // Keep the layer with its buffers.
    Ok(PreparedLayer::new(layer, vec![input, weight, scale, bias]))
}
