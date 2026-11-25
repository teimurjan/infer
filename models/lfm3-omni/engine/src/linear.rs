// nn.Linear without bias on every row, with the linear kernel of engine-core (float or packed weights).

// Our GPU connection, a kernel with its buffers, our error type, buffer categories and the weight buffers.
use engine_core::dispatch::BoundKernel;
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::layers::{LayerShape, LinearShape, bind_matrix_into};
use engine_core::memory::MemoryCategory;
use engine_core::weights::{GpuWeights, Matrix, WeightBuffers};

// Bind `output` [rows][out] = `input` [rows][in] x weight `name` [out][in] transposed. LFM3 has no biases.
pub fn bind_linear(gpu: &Gpu, weights: &GpuWeights, name: &str, rows: u32, input: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // The weight and its PyTorch shape [out, in].
    let weight = weights.matrix(gpu, name)?;
    let [out_features, in_features] = weight.shape[..] else {
        return Err(EngineError::WeightShape { name: name.to_string(), shape: weight.shape.clone(), expected: "[out, in]".to_string() });
    };
    bind_linear_matrix(gpu, name, weight.matrix, [out_features as u32, in_features as u32], rows, input, output)
}

// The same with a matrix of known shape [out, in] that the caller holds, for example one chunk of the output layer.
pub fn bind_linear_matrix(gpu: &Gpu, label: &str, matrix: Matrix, [out_features, in_features]: [u32; 2], rows: u32, input: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // The engine-core kernel adds a bias, so LFM3 gives it zeros.
    let zeros = gpu.storage_buffer_from(&format!("{label} zero bias"), &vec![0.0; out_features as usize], MemoryCategory::Weights);
    // The sizes of this layer.
    let shape = LayerShape::Linear(LinearShape { rows, in_features, out_features });
    // Connect the buffers, and keep the zero bias with the kernel.
    Ok(bind_matrix_into(gpu, &shape, input, matrix, &zeros, output)?.keeping(Some(zeros)))
}
