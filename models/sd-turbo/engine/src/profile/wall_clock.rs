// A kernel ready to record, and the function that records it.
use crate::dispatch::{Dispatch, encode_runs};
// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;
// One timed submit, and the check against the project limit.
use crate::submit::{check_submit, timed_submit};

// Run each dispatch as its own submit and wait for it, in order. The CPU clock around each wait is the time of that dispatch.
// The time includes the submit and the wait for onSubmittedWorkDone, so a small kernel looks slower than it is.
// Returns the seconds of the whole step and the seconds of each dispatch. Here both are the submit times.
pub(super) async fn time_dispatches(gpu: &Gpu, label: &str, dispatches: &[Dispatch]) -> Result<(f64, Vec<f64>), EngineError> {
    // The seconds of each dispatch, in order.
    let mut seconds = Vec::with_capacity(dispatches.len());
    for dispatch in dispatches {
        // One dispatch in one submit. Stop at once if it took too long, before the next submit.
        seconds.push(check_submit(gpu, label, timed_submit(gpu, encode_runs(gpu, std::slice::from_ref(dispatch), 1)).await?)?);
    }
    // The step took the sum of its submits.
    Ok((seconds.iter().sum(), seconds))
}
