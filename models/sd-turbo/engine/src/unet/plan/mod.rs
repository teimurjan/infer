// forward.rs records the UNet forward pass as ops.
mod forward;
// resnet.rs records one ResNet block.
mod resnet;
// transformer.rs records one transformer block with its two attentions.
mod transformer;

// The planner, its lower bound, the graph and the plan type.
use crate::plan::{Graph, MemoryPlan, assign_slots, live_peak};

// The forward pass, for the tests.
use forward::unet_graph;
// The level layout, which the binders of the stage follow.
pub(crate) use forward::{DOWN_LAYERS, DOWN_NAMES, LEVEL_CHANNELS, UP_NAMES};

// The activation plan of one UNet run, for the report.
pub struct ActivationPlan {
    // Every tensor of the forward pass.
    pub tensors: usize,
    // Every op of the forward pass.
    pub ops: usize,
    // The reusable buffers (slots) that hold all the tensors.
    pub buffers: usize,
    // The bytes of all buffers together. The buffers live for the whole UNet stage, so this is the planned peak.
    pub planned_bytes: u64,
    // The largest buffer. It must fit the buffer and binding limits of the device.
    pub largest_buffer_bytes: u64,
    // The most bytes that are live at one op. No plan can be smaller.
    pub live_peak_bytes: u64,
    // The name of that op.
    pub live_peak_op: String,
}

// The forward pass and the slot of each of its tensors. The UNet stage makes its buffers from this.
pub(crate) fn planned_unet() -> (Graph, MemoryPlan) {
    // The ops and tensors.
    let unet = unet_graph();
    // The lifetimes, then the slots.
    let plan = assign_slots(&unet.graph.lifetimes());
    // Both.
    (unet.graph, plan)
}

// Record the forward pass, work out the lifetime of each tensor, and put the tensors into reusable buffers.
pub fn plan_unet_activations() -> ActivationPlan {
    // The ops and tensors.
    let unet = unet_graph();
    // The size, first op and last op of each tensor.
    let lives = unet.graph.lifetimes();
    // The buffer of each tensor.
    let plan = assign_slots(&lives);
    // The lower bound.
    let peak = live_peak(&lives);
    // The numbers for the report.
    ActivationPlan {
        tensors: lives.len(),
        ops: unet.graph.op_count(),
        buffers: plan.slot_bytes.len(),
        planned_bytes: plan.total_bytes(),
        largest_buffer_bytes: plan.largest_slot_bytes(),
        live_peak_bytes: peak.bytes,
        live_peak_op: unet.graph.op_name(peak.op).to_string(),
    }
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the planner and the forward pass into the tests.
    use super::{assign_slots, plan_unet_activations, unet_graph};

    // The default WebGPU limit for one storage binding (maxStorageBufferBindingSize), 128 MiB. A kernel binds a whole slot.
    const BINDING_LIMIT: u64 = 134_217_728;

    // A wrong channel count or a skip tensor in the wrong order would change a block shape. PyTorch wrote the true shapes.
    #[test]
    fn block_shapes_match_the_golden_files() {
        // The forward pass.
        let unet = unet_graph();
        // The golden folder is in the project folder, one level above engine/.
        let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../golden/float32");
        // Every block with a golden file.
        for &(name, tensor) in &unet.blocks {
            // Read the golden file of this block.
            let golden = compare::safetensors::read_tensors(&folder.join(format!("{name}.safetensors"))).unwrap_or_else(|error| panic!("{name}: {error}"));
            // PyTorch's shape without the batch of 1, as u32.
            let shape: Vec<u32> = golden[0].shape[1..].iter().map(|&size| size as u32).collect();
            // Our size without the trailing 1s.
            let ours: Vec<u32> = unet.graph.dims(tensor).into_iter().take(shape.len()).collect();
            // They must be equal.
            assert_eq!(ours, shape, "{name}");
        }
        // conv_in, time_embedding, 4 down, mid, 4 up and the output: 12 blocks.
        assert_eq!(unet.blocks.len(), 12);
    }

    // Two tensors that are live at the same op must never share a buffer, or one would overwrite the other.
    #[test]
    fn tensors_that_are_live_together_never_share_a_buffer() {
        // The forward pass and its lifetimes.
        let unet = unet_graph();
        let lives = unet.graph.lifetimes();
        // The plan.
        let plan = assign_slots(&lives);
        // Each pair of tensors once.
        for first in 0..lives.len() {
            for second in first + 1..lives.len() {
                // A shared buffer is allowed only for tensors that never meet.
                let shared = plan.slot_of[first] == plan.slot_of[second];
                assert!(!(shared && lives[first].overlaps(&lives[second])), "{} and {} share a buffer", unet.graph.name(first), unet.graph.name(second));
            }
            // Each tensor fits its buffer.
            assert!(lives[first].bytes <= plan.slot_bytes[plan.slot_of[first]], "{} does not fit", unet.graph.name(first));
        }
        // The same check from the side of the ops: the tensors of one op are in different buffers.
        for op in 0..unet.graph.op_count() {
            // The slots of the tensors of this op. One tensor can appear twice in one op, so remove repeated tensors first.
            let mut tensors: Vec<usize> = unet.graph.tensors_of_op(op).collect();
            tensors.sort_unstable();
            tensors.dedup();
            // Their slots.
            let mut slots: Vec<usize> = tensors.iter().map(|&tensor| plan.slot_of[tensor]).collect();
            slots.sort_unstable();
            // No slot repeats.
            assert!(slots.windows(2).all(|pair| pair[0] != pair[1]), "op {} has two tensors in one buffer", unet.graph.op_name(op));
        }
    }

    // The pool finds a slot by tensor name. Two tensors with one name would make it give one of them the wrong slot.
    #[test]
    fn every_tensor_name_is_unique() {
        // The forward pass.
        let unet = unet_graph();
        // Every name.
        let mut names: Vec<&str> = (0..unet.graph.tensor_count()).map(|tensor| unet.graph.name(tensor)).collect();
        // Sort, so equal names sit side by side.
        names.sort_unstable();
        // No name repeats.
        let repeated: Vec<&[&str]> = names.windows(2).filter(|pair| pair[0] == pair[1]).collect();
        assert!(repeated.is_empty(), "{repeated:?}");
    }

    // The plan can never be below the live bytes, and each buffer must fit one binding.
    #[test]
    fn plan_is_above_the_lower_bound_and_fits_one_binding() {
        // The numbers.
        let plan = plan_unet_activations();
        // A plan below the lower bound would mean that two live tensors share bytes.
        assert!(plan.planned_bytes >= plan.live_peak_bytes);
        // A kernel binds a whole buffer.
        assert!(plan.largest_buffer_bytes <= BINDING_LIMIT, "{}", plan.largest_buffer_bytes);
    }
}
