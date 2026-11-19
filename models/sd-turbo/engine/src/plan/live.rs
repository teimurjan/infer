// The lifetime of one tensor.
use super::TensorLife;

// The most bytes that are live at one op, and the first op where that happens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LivePeak {
    // The bytes of every tensor that is live at `op`.
    pub(crate) bytes: u64,
    // The position of the op.
    pub(crate) op: usize,
}

// Add up the live bytes at each op and return the largest sum.
// Every live tensor needs its own bytes, so no slot plan can be smaller than this. The planner is compared with it.
pub(crate) fn live_peak(tensors: &[TensorLife]) -> LivePeak {
    // The number of ops: one more than the last op that any tensor reaches.
    let ops = tensors.iter().map(|tensor| tensor.last + 1).max().unwrap_or(0);
    // The live bytes at each op.
    let at_op = |op: usize| tensors.iter().filter(|tensor| tensor.first <= op && op <= tensor.last).map(|tensor| tensor.bytes).sum::<u64>();
    // Walk every op and keep the largest sum. On a tie, the first op wins, because the fold replaces only on a larger sum.
    (0..ops).fold(LivePeak { bytes: 0, op: 0 }, |best, op| {
        // The bytes live at this op.
        let bytes = at_op(op);
        // Keep the larger one.
        if bytes > best.bytes { LivePeak { bytes, op } } else { best }
    })
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function and the types into the tests.
    use super::{LivePeak, TensorLife, live_peak};

    // At op 1, the first two tensors are live: 100 + 30 bytes. At op 2, only 30 + 5.
    #[test]
    fn finds_the_op_with_the_most_live_bytes() {
        // Three tensors.
        let tensors = [TensorLife { bytes: 100, first: 0, last: 1 }, TensorLife { bytes: 30, first: 1, last: 2 }, TensorLife { bytes: 5, first: 2, last: 2 }];
        // The peak is at op 1.
        assert_eq!(live_peak(&tensors), LivePeak { bytes: 130, op: 1 });
    }
}
