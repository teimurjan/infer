// Reverse sorts the largest tensors first.
use std::cmp::Reverse;

// The lifetime of one tensor and the result type.
use super::{MemoryPlan, TensorLife};

// Give each tensor a slot. Two tensors share a slot only if their lifetimes do not overlap.
// The method is "greedy by size": take the tensors from the largest to the smallest.
// Put each one into the smallest slot that is free for its whole lifetime, or into a new slot.
// Because the largest tensors come first, a slot rarely has to grow. It still grows to its largest tensor, so the plan stays correct in any order.
pub(crate) fn assign_slots(tensors: &[TensorLife]) -> MemoryPlan {
    // The tensor positions, largest first. Ties go to the earlier tensor, so the result is the same on every run.
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    // Sort by size (descending), then by first op, then by position.
    order.sort_by_key(|&tensor| (Reverse(tensors[tensor].bytes), tensors[tensor].first, tensor));
    // The tensors in each slot so far.
    let mut members: Vec<Vec<usize>> = Vec::new();
    // The size of each slot so far.
    let mut slot_bytes: Vec<u64> = Vec::new();
    // The slot of each tensor. Every entry is set in the loop.
    let mut slot_of = vec![0; tensors.len()];
    // One tensor at a time, largest first.
    for tensor in order {
        // The slots where no member is live at the same time as this tensor.
        let free = (0..members.len()).filter(|&slot| members[slot].iter().all(|&other| !tensors[other].overlaps(&tensors[tensor])));
        // The smallest free slot wastes the fewest bytes. Ties go to the older slot.
        let best = free.min_by_key(|&slot| (slot_bytes[slot], slot));
        // No free slot: open a new one with the size of this tensor.
        let slot = best.unwrap_or_else(|| {
            // The new slot starts at the size of this tensor.
            slot_bytes.push(tensors[tensor].bytes);
            // It starts empty.
            members.push(Vec::new());
            // Its position.
            members.len() - 1
        });
        // The slot must hold its largest tensor.
        slot_bytes[slot] = slot_bytes[slot].max(tensors[tensor].bytes);
        // The tensor joins the slot.
        members[slot].push(tensor);
        // Remember the choice.
        slot_of[tensor] = slot;
    }
    // Every tensor has a slot.
    MemoryPlan { slot_of, slot_bytes }
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the planner and the lifetime into the tests.
    use super::{TensorLife, assign_slots};

    // A short way to write one lifetime.
    fn life(bytes: u64, first: usize, last: usize) -> TensorLife {
        TensorLife { bytes, first, last }
    }

    // A chain a -> b -> c: a and c are never live together, so they share a slot.
    #[test]
    fn tensors_that_never_meet_share_a_slot() {
        // a lives at ops 0 and 1, b at 1 and 2, c at 2 and 3.
        let plan = assign_slots(&[life(100, 0, 1), life(100, 1, 2), life(100, 2, 3)]);
        // a and c share. b needs its own slot.
        assert_eq!(plan.slot_of[0], plan.slot_of[2]);
        assert_ne!(plan.slot_of[0], plan.slot_of[1]);
        // Two slots of 100 bytes.
        assert_eq!(plan.total_bytes(), 200);
    }

    // A tensor that is read at the op where another one is written is live at that op, so they must not share.
    #[test]
    fn the_last_reader_and_a_new_writer_do_not_share() {
        // a ends at op 2 and b starts at op 2.
        let plan = assign_slots(&[life(10, 0, 2), life(10, 2, 3)]);
        // Two slots.
        assert_ne!(plan.slot_of[0], plan.slot_of[1]);
    }

    // A small tensor goes into the smallest slot that is free, not into the first one.
    #[test]
    fn a_small_tensor_takes_the_smallest_free_slot() {
        // A 1000-byte tensor and a 50-byte tensor end at op 0. A 40-byte tensor comes later.
        let plan = assign_slots(&[life(1000, 0, 0), life(50, 0, 0), life(40, 1, 1)]);
        // The 40-byte tensor goes into the 50-byte slot.
        assert_eq!(plan.slot_of[2], plan.slot_of[1]);
        // The slots stay at 1000 and 50 bytes.
        assert_eq!(plan.total_bytes(), 1050);
    }
}
