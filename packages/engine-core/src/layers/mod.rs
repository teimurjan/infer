// conv1x1.rs prepares a 1x1 convolution.
mod conv1x1;
// conv3x3.rs prepares a 3x3 convolution with stride 1 or 2.
mod conv3x3;
// linear.rs prepares a matrix multiply plus bias.
mod linear;
// vector.rs prepares nn.Linear on one row, with an optional SiLU on the input.
mod vector;
// packed/ holds the versions of the three kernels that read 4-bit or 8-bit packed weights.
mod packed;

// The kinds of GPU buffer that the memory counter keeps apart.
use crate::memory::MemoryCategory;
// A kernel ready to run, a kernel with its buffers, and the function that records runs of it.
use crate::dispatch::{BoundKernel, Dispatch, Work, encode_runs};
// Buffers that count their bytes.
use crate::memory::GpuBuffer;
// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The shared buffer length checks: exact for buffers of one tensor, "at least" for slots.
use crate::checks::{check_buffer, check_buffer_holds};

// The shape types that callers fill in.
pub use conv1x1::Conv1x1Shape;
pub use conv3x3::Conv3x3Shape;
// The output size formula, which the UNet needs to know the shape of each activation.
pub use conv3x3::conv3x3_output_size;
pub use linear::LinearShape;
// The packed weight types and the function that prepares a packed layer.
pub use packed::{PackedTensors, WeightBits};
pub use packed::prepare_packed_layer;
// The packed kernels on slots.
use packed::{bind_packed_into, bind_packed_vector_into};
// The weight matrix that a binder got: float or packed.
use crate::weights::Matrix;
// The one-row linear layer.
pub use vector::{VectorShape, bind_vector_into};

// Every invocation computes a block of 4 x 4 outputs, so block dimensions must be multiples of 4.
const BLOCK: u32 = 4;
// Workgroups are 8 x 8 invocations. It must match @workgroup_size(8, 8, 1) in every layer shader.
const WORKGROUP_SIDE: u32 = 8;

// One layer kind and its sizes.
pub enum LayerShape {
    // nn.Linear.
    Linear(LinearShape),
    // nn.Conv2d with a 1x1 kernel.
    Conv1x1(Conv1x1Shape),
    // nn.Conv2d with a 3x3 kernel and padding 1.
    Conv3x3(Conv3x3Shape),
}

// The three tensors a layer reads, in PyTorch's layout.
pub struct Tensors<'a> {
    // The layer input.
    pub input: &'a [f32],
    // The weights.
    pub weight: &'a [f32],
    // One bias per output channel or column.
    pub bias: &'a [f32],
}

// How many values each tensor must have for one layer shape.
struct Lengths {
    // Values in the input.
    input: usize,
    // Values in the weights.
    weight: usize,
    // Values in the bias.
    bias: usize,
    // Values in the output.
    output: usize,
}

// The checked sizes of one layer: everything its kernel needs except the data.
struct LayerPlan {
    // The kernel name, for GPU error messages.
    label: &'static str,
    // The WGSL source.
    shader: &'static str,
    // The values of the shader's Shape struct, in field order.
    sizes: Vec<u32>,
    // The length of each tensor.
    lengths: Lengths,
    // How many workgroups to start along x, y and z.
    workgroups: [u32; 3],
}

// Check the sizes of one layer and plan its kernel.
fn plan_layer(shape: &LayerShape) -> Result<LayerPlan, EngineError> {
    // Each layer kind has its own sizes and shader.
    match shape {
        // Matrix multiply plus bias.
        LayerShape::Linear(linear) => linear::plan_linear(linear),
        // 1x1 convolution.
        LayerShape::Conv1x1(conv) => conv1x1::plan_conv1x1(conv),
        // 3x3 convolution.
        LayerShape::Conv3x3(conv) => conv3x3::plan_conv3x3(conv),
    }
}

// A layer whose kernel is connected to buffers that are already on the GPU.
// The UNet chains these: the output of one layer is the input of the next, and nothing goes back to the CPU.
pub struct BoundLayer {
    // The kernel with its buffers.
    dispatch: Dispatch,
    // The buffer the kernel writes.
    output: GpuBuffer,
    // The sizes buffer. We keep it, because the browser frees a buffer when Rust drops it.
    _uniform: GpuBuffer,
}

impl BoundLayer {
    // The kernel, ready to record.
    pub fn dispatch(&self) -> &Dispatch {
        &self.dispatch
    }

    // The buffer the kernel writes, so that another kernel in the same submit can read or change it.
    pub fn output(&self) -> &GpuBuffer {
        &self.output
    }

    // The output buffer. Take it only after the submit that runs the layer.
    pub fn into_output(self) -> GpuBuffer {
        self.output
    }
}

// One buffer that a kernel reads, with its name and the number of 4-byte values it must hold.
struct Binding<'a> {
    // The tensor name, for the error message.
    name: &'static str,
    // The GPU buffer.
    buffer: &'a wgpu::Buffer,
    // The number of 4-byte values: f32 values, or u32 words for packed data.
    values: usize,
}

// Check the buffer lengths, make the output buffer, and connect the kernel.
// The bindings are: 0 the sizes, then the inputs in order, then the output.
fn bind_plan(gpu: &Gpu, plan: &LayerPlan, inputs: &[Binding]) -> Result<BoundLayer, EngineError> {
    // A buffer of the wrong length would make the kernel read the wrong values, so stop first.
    for input in inputs {
        // Compare the buffer size with the length the plan needs.
        check_buffer(input.name, input.buffer, input.values)?;
    }
    // The output goes to the last binding.
    let output = gpu.empty_storage_buffer(plan.label, plan.lengths.output);
    // Connect the kernel.
    let (dispatch, uniform) = connect(gpu, plan, inputs, &output)?;
    // Keep everything together.
    Ok(BoundLayer { dispatch, output, _uniform: uniform })
}

// Make the sizes buffer, compile the shader, and connect the sizes, the inputs and the output, in binding order.
fn connect(gpu: &Gpu, plan: &LayerPlan, inputs: &[Binding], output: &wgpu::Buffer) -> Result<(Dispatch, GpuBuffer), EngineError> {
    // The sizes go to @binding(0).
    let uniform = gpu.uniform_buffer(plan.label, &plan.sizes);
    // The sizes, the inputs and the output, in binding order.
    let buffers: Vec<&wgpu::Buffer> = std::iter::once(&*uniform).chain(inputs.iter().map(|input| input.buffer)).chain(std::iter::once(output)).collect();
    // Each input once and the output once. Every value is 4 bytes: an f32, or a u32 word of packed weights.
    let values = inputs.iter().map(|input| input.values as u64).sum::<u64>() + plan.lengths.output as u64;
    // Compile the shader and connect the buffers. This makes the one bind group of the kernel.
    let dispatch = Dispatch::new(gpu, plan.label, plan.shader, &buffers, plan.workgroups, Work::f32_values(plan.label, values))?;
    // The kernel and its sizes buffer.
    Ok((dispatch, uniform))
}

// Check the sizes, make the output buffer, and connect the kernel to the input, weight and bias buffers.
// The caller must keep input, weight and bias alive until the GPU has run the dispatch.
pub fn bind_layer(gpu: &Gpu, shape: &LayerShape, input: &wgpu::Buffer, weight: &wgpu::Buffer, bias: &wgpu::Buffer) -> Result<BoundLayer, EngineError> {
    // Check the sizes and pick the kernel.
    let plan = plan_layer(shape)?;
    // The three float tensors, in binding order 1, 2, 3. The output is binding 4.
    let inputs = [
        Binding { name: "input", buffer: input, values: plan.lengths.input },
        Binding { name: "weight", buffer: weight, values: plan.lengths.weight },
        Binding { name: "bias", buffer: bias, values: plan.lengths.bias },
    ];
    // Check the lengths and connect the kernel.
    bind_plan(gpu, &plan, &inputs)
}

// Like bind_layer, but the kernel writes into `output` and makes no buffer of its own.
// The result keeps only the uniform. The caller adds the weights with `keeping`.
pub fn bind_layer_into(gpu: &Gpu, shape: &LayerShape, input: &wgpu::Buffer, weight: &wgpu::Buffer, bias: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // Check the sizes, pick the kernel, and connect the slots.
    bind_plan_into(gpu, &plan_layer(shape)?, input, weight, bias, output)
}

// Check the buffers of a planned layer and connect its kernel. Bindings: 0 the sizes, 1 the input, 2 the weight, 3 the bias, 4 the output.
// The input and the output can be larger than the tensors, because a slot is as large as its largest tensor.
fn bind_plan_into(gpu: &Gpu, plan: &LayerPlan, input: &wgpu::Buffer, weight: &wgpu::Buffer, bias: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // A slot that is too small would make the kernel read or write past the tensor.
    check_buffer_holds("input", input, plan.lengths.input)?;
    check_buffer_holds("output", output, plan.lengths.output)?;
    // The weights hold one tensor each, so their length must be exact.
    check_buffer("weight", weight, plan.lengths.weight)?;
    check_buffer("bias", bias, plan.lengths.bias)?;
    // The input, weight and bias, in binding order 1, 2, 3. The lengths are checked above.
    let inputs = [
        Binding { name: "input", buffer: input, values: plan.lengths.input },
        Binding { name: "weight", buffer: weight, values: plan.lengths.weight },
        Binding { name: "bias", buffer: bias, values: plan.lengths.bias },
    ];
    // Connect the kernel.
    let (dispatch, uniform) = connect(gpu, plan, &inputs, output)?;
    // The layer with its uniform, ready to run many times.
    Ok(BoundKernel::new(vec![dispatch], vec![uniform]))
}

// Connect the float or the packed kernel of `shape`, whichever `matrix` needs. The kernel keeps a float matrix that was made for it.
pub fn bind_matrix_into(gpu: &Gpu, shape: &LayerShape, input: &wgpu::Buffer, matrix: Matrix, bias: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // Pick the kernel.
    match matrix {
        // The float kernel, and the matrix buffer if it is its own.
        Matrix::Float(weight) => Ok(bind_layer_into(gpu, shape, input, weight.buffer(), bias, output)?.keeping(weight.into_owned())),
        // The packed kernel. The packed buffers belong to the loaded weights.
        Matrix::Packed { words, scale, bits } => bind_packed_into(gpu, shape, bits, input, words, scale, bias, output),
    }
}

// The same for the one-row layer.
pub fn bind_vector_matrix_into(gpu: &Gpu, shape: &VectorShape, input: &wgpu::Buffer, matrix: Matrix, bias: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // Pick the kernel.
    match matrix {
        // The float kernel, and the matrix buffer if it is its own.
        Matrix::Float(weight) => Ok(bind_vector_into(gpu, shape, input, weight.buffer(), bias, output)?.keeping(weight.into_owned())),
        // The packed kernel.
        Matrix::Packed { words, scale, bits } => bind_packed_vector_into(gpu, shape, bits, input, words, scale, bias, output),
    }
}

// A layer with its data copied from the CPU, ready to run many times. The layer tests and benchmarks use it.
pub struct PreparedLayer {
    // The kernel with its buffers.
    layer: BoundLayer,
    // The buffers the kernel reads. We keep them, because the browser frees a buffer when Rust drops it.
    _inputs: Vec<GpuBuffer>,
    // Bytes that one run must read and write at least: every tensor once.
    bytes_moved: u64,
}

impl PreparedLayer {
    // Keep a bound layer with the buffers it reads. Every tensor is read or written once, so the bytes moved are the buffer sizes.
    fn new(layer: BoundLayer, inputs: Vec<GpuBuffer>) -> Self {
        // The input buffers plus the output buffer.
        let bytes_moved = inputs.iter().map(|input| input.size()).sum::<u64>() + layer.output.size();
        // Keep everything together.
        Self { layer, _inputs: inputs, bytes_moved }
    }

    // The kernel, for timing.
    pub fn dispatch(&self) -> &Dispatch {
        self.layer.dispatch()
    }

    // Bytes that one run must read and write at least.
    pub fn bytes_moved(&self) -> u64 {
        self.bytes_moved
    }

    // Run the layer once and copy the output back to the CPU.
    pub async fn run(&self, gpu: &Gpu) -> Result<Vec<f32>, EngineError> {
        // Record one dispatch and send it to the GPU.
        gpu.queue.submit([encode_runs(gpu, std::slice::from_ref(self.dispatch()), 1)]);
        // Copy the output back.
        gpu.read_f32(&self.layer.output).await
    }
}

// Copy the tensors to the GPU, check the sizes, and compile the right kernel for this layer kind.
pub fn prepare_layer(gpu: &Gpu, shape: &LayerShape, tensors: &Tensors) -> Result<PreparedLayer, EngineError> {
    // The input goes to the GPU as it is.
    let input = gpu.storage_buffer_from("layer input", tensors.input, MemoryCategory::Activations);
    // The weights, in PyTorch's layout.
    let weight = gpu.storage_buffer_from("layer weight", tensors.weight, MemoryCategory::Weights);
    // The bias.
    let bias = gpu.storage_buffer_from("layer bias", tensors.bias, MemoryCategory::Weights);
    // bind_layer checks the lengths, so a wrong tensor stops here.
    let layer = bind_layer(gpu, shape, &input, &weight, &bias)?;
    // Keep the layer with its buffers.
    Ok(PreparedLayer::new(layer, vec![input, weight, bias]))
}

// A dimension that the kernel splits into blocks of 4. It must be a positive multiple of 4.
pub fn block_dimension(dimension: &'static str, value: u32) -> Result<u32, EngineError> {
    // 0 and values like 6 are not allowed.
    if value == 0 || !value.is_multiple_of(BLOCK) {
        return Err(EngineError::BadDimension { dimension, value });
    }
    // The value is fine.
    Ok(value)
}

// Workgroups that cover `invocations` invocations along one axis, 8 per workgroup, rounded up.
fn workgroups_for(invocations: u32) -> u32 {
    // Round up, so the last few invocations also get a workgroup.
    invocations.div_ceil(WORKGROUP_SIDE)
}

// The number of values in a tensor with these dimensions.
fn product(dimensions: &[u32]) -> usize {
    // Multiply as usize, so that large tensors cannot overflow u32.
    dimensions.iter().map(|&dimension| dimension as usize).product()
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function and the error into the tests.
    use super::{EngineError, block_dimension};

    // A size that is not a multiple of 4 would make the kernel read past its rows, so it must be refused.
    #[test]
    fn block_dimensions_must_be_positive_multiples_of_4() {
        // 320 is fine.
        assert_eq!(block_dimension("rows", 320).unwrap(), 320);
        // 6 is not a multiple of 4.
        assert!(matches!(block_dimension("rows", 6), Err(EngineError::BadDimension { value: 6, .. })));
        // 0 is not positive.
        assert!(matches!(block_dimension("rows", 0), Err(EngineError::BadDimension { value: 0, .. })));
    }
}
