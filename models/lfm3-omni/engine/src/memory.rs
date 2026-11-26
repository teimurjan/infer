// The memory of earlier tokens: what each layer must keep so that the next word needs only one new row.
// An attention layer keeps the keys and values of every position. A convolution layer keeps h = B * x of every
// position, of which the convolution reads the last 7. One small step buffer says where the next run starts.

// Our GPU connection, our error type, and buffers that count their bytes.
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::memory::{GpuBuffer, MemoryCategory};

// The layer kinds and sizes, the head size, and the rotary tables.
use crate::config::{KV_HEADS, LAYER_KINDS, LayerKind, WIDTH, rope_settings};
use crate::kernels::{HEAD_DIM, RopeTables};
use crate::rope_table::rope_table;

// What one layer keeps.
pub enum LayerMemory {
    // h of every position: [capacity][width].
    Conv { history: GpuBuffer },
    // The keys and values of every position: [capacity][kv_heads][128] each.
    Attention { keys: GpuBuffer, values: GpuBuffer },
}

// The memory of all 24 layers, the step buffer and the rotary tables, for up to `capacity` positions.
pub struct Memory {
    capacity: u32,
    // [start, keys]: the position of the first new token, and the positions that exist after the run.
    step: GpuBuffer,
    layers: Vec<LayerMemory>,
    // cos and sin of every position, for the full and the sliding layers.
    full: [GpuBuffer; 2],
    sliding: [GpuBuffer; 2],
}

impl Memory {
    // Make the memory for up to `capacity` positions, the prompt and the answer together.
    pub fn new(gpu: &Gpu, capacity: u32) -> Self {
        let layers = LAYER_KINDS.iter().map(|kind| match kind {
            LayerKind::Conv => LayerMemory::Conv { history: gpu.empty_storage_buffer("conv history", (capacity * WIDTH) as usize) },
            LayerKind::Full | LayerKind::Sliding => {
                let values = (capacity * KV_HEADS * HEAD_DIM) as usize;
                LayerMemory::Attention { keys: gpu.empty_storage_buffer("memory keys", values), values: gpu.empty_storage_buffer("memory values", values) }
            }
        }).collect();
        // The cos and sin tables of one kind of attention layer.
        let tables = |kind: LayerKind| {
            let settings = rope_settings(kind).expect("an attention kind has rotary settings");
            let table = rope_table(capacity, settings.angles, HEAD_DIM, settings.theta);
            [gpu.storage_buffer_from("rope cos", &table.cos, MemoryCategory::Weights), gpu.storage_buffer_from("rope sin", &table.sin, MemoryCategory::Weights)]
        };
        let step = gpu.writable_uniform_buffer("step", &[0, 0]);
        Self { capacity, step, layers, full: tables(LayerKind::Full), sliding: tables(LayerKind::Sliding) }
    }

    // The most positions that the memory holds.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    // The step buffer that the kernels read.
    pub fn step(&self) -> &wgpu::Buffer {
        &self.step
    }

    // What layer `index` keeps.
    pub fn layer(&self, index: usize) -> &LayerMemory {
        &self.layers[index]
    }

    // The rotary tables of one kind of attention layer.
    pub fn rope(&self, kind: LayerKind) -> RopeTables<'_> {
        let [cos, sin] = if kind == LayerKind::Full { &self.full } else { &self.sliding };
        RopeTables { cos, sin }
    }

    // Before a run of `rows` new tokens from position `start`: tell the kernels where it starts.
    pub fn set_step(&self, gpu: &Gpu, start: u32, rows: u32) -> Result<(), EngineError> {
        // A position past the memory would write outside its buffers.
        if start + rows > self.capacity {
            return Err(EngineError::TooManyValues { count: (start + rows) as usize, max: self.capacity as usize });
        }
        gpu.write_u32(&self.step, &[start, start + rows]);
        Ok(())
    }
}
