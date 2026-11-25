// nn.Linear without bias on every row, with the linear kernel of engine-core (float or packed weights).

// Our GPU connection, a kernel with its buffers, our error type, buffer categories and the weight buffers.
use engine_core::dispatch::BoundKernel;
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::layers::{LayerShape, LinearShape, bind_matrix_into};
use engine_core::memory::MemoryCategory;
use engine_core::weights::{GpuWeights, WeightBuffers};

// Bind `output` [rows][out] = `input` [rows][in] x weight `name` [out][in] transposed. LFM3 has no biases.
pub fn bind_linear(gpu: &Gpu, weights: &GpuWeights, name: &str, rows: u32, input: &wgpu::Buffer, output: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // The weight and its PyTorch shape [out, in].
    let weight = weights.matrix(gpu, name)?;
    let [out_features, in_features] = weight.shape[..] else {
        return Err(EngineError::WeightShape { name: name.to_string(), shape: weight.shape.clone(), expected: "[out, in]".to_string() });
    };
    // The engine-core kernel adds a bias, so LFM3 gives it zeros.
    let zeros = gpu.storage_buffer_from(&format!("{name} zero bias"), &vec![0.0; out_features], MemoryCategory::Weights);
    // The sizes of this layer. A count that does not fit in u32 cannot come from this model.
    let shape = LayerShape::Linear(LinearShape { rows, in_features: in_features as u32, out_features: out_features as u32 });
    // Connect the buffers, and keep the zero bias with the kernel.
    Ok(bind_matrix_into(gpu, &shape, input, weight.matrix, &zeros, output)?.keeping(Some(zeros)))
}
