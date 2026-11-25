// The kinds of GPU buffer that the memory counter keeps apart.
use crate::memory::MemoryCategory;
// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// Buffers that count their bytes.
use crate::memory::GpuBuffer;
// A kernel ready to record, and the function that records it.
use crate::dispatch::{Dispatch, Work, encode_runs};
// The bit width of a packed weight.
use super::bits::WeightBits;

// The two places that weights come from.
use super::{GpuWeights, WeightSource};

// The WGSL source of the float16 to f32 kernel.
const WIDEN_SHADER: &str = include_str!("widen.wgsl");
// Invocations per workgroup. It must match @workgroup_size(256) in widen.wgsl.
const WORKGROUP_SIZE: u32 = 256;
// Bytes in one f32 value.
const F32_BYTES: u64 = 4;

// A weight buffer that a slot kernel binds: one that a loaded weight file owns, or one made for this kernel.
pub enum WeightBuffer<'a> {
    // A buffer of `GpuWeights`. The weights must live as long as the kernel, so the UNet stage keeps them.
    Borrowed(&'a wgpu::Buffer),
    // A buffer made for this kernel. The kernel must keep it.
    Owned(GpuBuffer),
}

impl WeightBuffer<'_> {
    // The buffer, for a bind group.
    pub fn buffer(&self) -> &wgpu::Buffer {
        // Both kinds hold a wgpu buffer.
        match self {
            Self::Borrowed(buffer) => buffer,
            Self::Owned(buffer) => buffer,
        }
    }

    // The buffer that the kernel must keep, if this one is its own.
    pub fn into_owned(self) -> Option<GpuBuffer> {
        // A borrowed buffer stays with its owner.
        match self {
            Self::Borrowed(_) => None,
            Self::Owned(buffer) => Some(buffer),
        }
    }
}

// A float32 tensor on the GPU, with its shape.
pub struct FloatWeight<'a> {
    // The size of each dimension, in PyTorch's order.
    pub shape: Vec<usize>,
    // The f32 values.
    pub buffer: WeightBuffer<'a>,
}

// The two kinds of weight matrix that the layer kernels read.
pub enum Matrix<'a> {
    // f32 values in PyTorch's layout.
    Float(WeightBuffer<'a>),
    // Packed integers and float16 row scales, in the format of pack_weights.py.
    Packed {
        // The packed words, one row per output channel.
        words: &'a wgpu::Buffer,
        // One float16 scale per output channel, two per word.
        scale: &'a wgpu::Buffer,
        // 4 or 8 bits per integer.
        bits: WeightBits,
    },
}

// A weight matrix (of a linear layer or a convolution) with its PyTorch shape.
pub struct MatrixWeight<'a> {
    // The size of each dimension, in PyTorch's order, for example [out, in, 3, 3].
    pub shape: Vec<usize>,
    // The values.
    pub matrix: Matrix<'a>,
}

// Where the slot kernels get their weights.
pub trait WeightBuffers {
    // A tensor as f32 values, such as a bias or a norm scale.
    fn float(&self, gpu: &Gpu, name: &str) -> Result<FloatWeight<'_>, EngineError>;
    // A weight matrix, float or packed.
    fn matrix(&self, gpu: &Gpu, name: &str) -> Result<MatrixWeight<'_>, EngineError>;
}

// Weights from a CPU reader. Each tensor is read and copied to a new GPU buffer when a kernel asks for it.
pub struct Uploading<'a, W: WeightSource>(pub &'a W);

impl<W: WeightSource> WeightBuffers for Uploading<'_, W> {
    // Read the f32 bytes and copy them to the GPU.
    fn float(&self, gpu: &Gpu, name: &str) -> Result<FloatWeight<'_>, EngineError> {
        // The reader accepts float32 only.
        let tensor = self.0.read(name)?;
        // A new buffer for the kernel. The buffer name is the tensor name, for GPU error messages.
        Ok(FloatWeight { shape: tensor.shape, buffer: WeightBuffer::Owned(gpu.storage_buffer_from_bytes(name, &tensor.bytes, MemoryCategory::Weights)) })
    }

    // A CPU reader has float32 matrices only.
    fn matrix(&self, gpu: &Gpu, name: &str) -> Result<MatrixWeight<'_>, EngineError> {
        // The same read as a float tensor.
        let weight = self.float(gpu, name)?;
        // As a float matrix.
        Ok(MatrixWeight { shape: weight.shape, matrix: Matrix::Float(weight.buffer) })
    }
}

impl WeightBuffers for GpuWeights {
    // Lend a float32 buffer, or widen a float16 one into a new f32 buffer.
    fn float(&self, gpu: &Gpu, name: &str) -> Result<FloatWeight<'_>, EngineError> {
        // The tensor as the file stores it.
        let tensor = self.tensor(name)?;
        // Pick by the dtype of the index.
        let buffer = match tensor.dtype.as_str() {
            // The kernels read it as it is.
            "float32" => WeightBuffer::Borrowed(&tensor.buffer),
            // The small float16 tensors of the packed file: biases and norm scales.
            "float16" => WeightBuffer::Owned(widen(gpu, name, &tensor.buffer, tensor.shape.iter().product())?),
            // A packed tensor is not a float tensor.
            other => return Err(EngineError::WeightDtype { name: name.to_string(), dtype: other.to_string() }),
        };
        // The values and the shape.
        Ok(FloatWeight { shape: tensor.shape.clone(), buffer })
    }

    // Lend the buffers of a packed matrix, or give a float one.
    fn matrix(&self, gpu: &Gpu, name: &str) -> Result<MatrixWeight<'_>, EngineError> {
        // The tensor as the file stores it.
        let tensor = self.tensor(name)?;
        // The bit width of a packed dtype.
        let bits = match tensor.dtype.as_str() {
            "int4" => WeightBits::Four,
            "int8" => WeightBits::Eight,
            // Any other dtype is a float tensor.
            _ => {
                let weight = self.float(gpu, name)?;
                return Ok(MatrixWeight { shape: weight.shape, matrix: Matrix::Float(weight.buffer) });
            }
        };
        // The float16 scales follow as "<name>_scale", as pack_weights.py writes them.
        let scale = &self.tensor(&format!("{name}_scale"))?.buffer;
        // The packed words and the scales, as the file stores them.
        Ok(MatrixWeight { shape: tensor.shape.clone(), matrix: Matrix::Packed { words: &tensor.buffer, scale, bits } })
    }
}

// Make a new f32 buffer with the `count` float16 values of `input`, with one submit of widen.wgsl.
// The submit runs before every later submit, so the kernels that read the new buffer see its values.
fn widen(gpu: &Gpu, name: &str, input: &wgpu::Buffer, count: usize) -> Result<GpuBuffer, EngineError> {
    // Exactly one f32 per value, because the shader takes the count from the buffer length.
    let output = gpu.empty_weight_buffer(&format!("{name} as f32"), count as u64 * F32_BYTES);
    // One invocation per value. A count that does not fit u32 cannot be a bias or a norm scale.
    let workgroups = u32::try_from(count.div_ceil(WORKGROUP_SIZE as usize)).map_err(|_| EngineError::TooManyValues { count, max: u32::MAX as usize })?;
    // Bindings: the float16 words, the f32 values.
    // It reads the float16 words and writes the f32 values.
    let work = Work { kernel: "widen", bytes: input.size() + output.size() };
    let dispatch = Dispatch::new(gpu, "widen", WIDEN_SHADER, &[input, &output], [workgroups, 1, 1], work)?;
    // Send it now. Nothing waits for it here: the GPU runs submits in order.
    gpu.queue.submit([encode_runs(gpu, std::slice::from_ref(&dispatch), 1)]);
    // The new buffer.
    Ok(output)
}
