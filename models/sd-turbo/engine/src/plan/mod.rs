// graph.rs records a forward pass as a list of ops and the tensors that each op reads and writes.
mod graph;
// live.rs finds the most bytes that are live at one op. No plan can be smaller than that.
mod live;
// pool.rs makes one GPU buffer per slot and finds the buffer of a tensor by its name.
mod pool;
// slots.rs puts tensors whose lifetimes do not overlap into the same reusable buffer.
mod slots;

// The parts that the UNet plan uses.
pub(crate) use graph::{Graph, TensorId};
pub(crate) use live::live_peak;
pub(crate) use pool::{ActivationPool, SlotTensor};
pub(crate) use slots::assign_slots;

// One tensor of a forward pass: its size, and the first and the last op that use it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TensorLife {
    // The bytes that the tensor needs.
    pub(crate) bytes: u64,
    // The position of the op that writes it.
    pub(crate) first: usize,
    // The position of the last op that reads it. The tensor is live from `first` to `last`, both included.
    pub(crate) last: usize,
}

impl TensorLife {
    // Two tensors that are live at the same op cannot share a buffer.
    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        // Two closed ranges overlap when each one starts before the other one ends.
        self.first <= other.last && other.first <= self.last
    }
}

// The result of the planner: the buffer of each tensor, and the size of each buffer.
#[derive(Debug)]
pub(crate) struct MemoryPlan {
    // Entry i is the buffer (slot) of tensor i.
    pub(crate) slot_of: Vec<usize>,
    // Entry s is the size in bytes of slot s: the size of its largest tensor.
    pub(crate) slot_bytes: Vec<u64>,
}

impl MemoryPlan {
    // The bytes of all slots together. The slots exist for the whole stage, so this is the planned peak of the activations.
    pub(crate) fn total_bytes(&self) -> u64 {
        self.slot_bytes.iter().sum()
    }

    // The largest slot. It must fit in one GPU buffer and in one binding.
    pub(crate) fn largest_slot_bytes(&self) -> u64 {
        // 0 for a plan without tensors.
        self.slot_bytes.iter().copied().max().unwrap_or(0)
    }
}
