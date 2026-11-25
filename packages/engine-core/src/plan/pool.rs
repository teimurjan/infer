// HashMap finds a tensor by its name.
use std::collections::HashMap;

// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// Buffers that count their bytes.
use crate::memory::GpuBuffer;

// The graph that names the tensors, and the plan that puts them into slots.
use super::{Graph, MemoryPlan};

// Bytes in one f32 value.
const F32_BYTES: u64 = 4;

// A tensor of the plan: the slot buffer that holds it, and its size.
pub struct SlotTensor<'a> {
    // The tensor name in the plan, for error messages.
    pub name: &'a str,
    // The slot buffer. It can be larger than the tensor.
    pub buffer: &'a wgpu::Buffer,
    // The size of the tensor, as in the plan.
    pub dims: [u32; 3],
}

// One GPU buffer per slot of the plan. The buffers are made once, at the start of the stage, and live until the stage ends.
pub struct ActivationPool {
    // One buffer per slot, in the order of the plan.
    slots: Vec<GpuBuffer>,
    // Tensor name to (slot, size).
    tensors: HashMap<String, (usize, [u32; 3])>,
}

impl ActivationPool {
    // Make the buffer of every slot, and remember where each tensor of `graph` goes.
    pub fn new(gpu: &Gpu, graph: &Graph, plan: &MemoryPlan) -> Self {
        // Every slot holds f32 values, so its size is a whole number of values.
        let slots = plan.slot_bytes.iter().enumerate().map(|(slot, &bytes)| gpu.empty_storage_buffer(&format!("slot {slot}"), (bytes / F32_BYTES) as usize)).collect();
        // The slot and the size of each tensor, by name.
        let tensors = (0..graph.tensor_count()).map(|tensor| (graph.name(tensor).to_string(), (plan.slot_of[tensor], graph.dims_at(tensor)))).collect();
        // The pool.
        Self { slots, tensors }
    }

    // The slot buffer and the size of the tensor called `name`.
    pub fn tensor<'a>(&'a self, name: &'a str) -> Result<SlotTensor<'a>, EngineError> {
        // A name that the plan does not have is a typo or a plan that is out of date.
        let &(slot, dims) = self.tensors.get(name).ok_or_else(|| EngineError::UnplannedTensor(name.to_string()))?;
        // The buffer of its slot.
        Ok(SlotTensor { name, buffer: &self.slots[slot], dims })
    }
}
