// A kernel ready to run, and the function that records runs of it.
use crate::dispatch::{Dispatch, encode_runs};
// Our GPU connection.
use crate::gpu::Gpu;
// Our error type, and the names of the submits for its message.
use crate::error::{EngineError, SubmitPhase};
// The project limit for one submit, and the clock around one submit.
use crate::submit::{SUBMIT_LIMIT_SECONDS, timed_submit};

// Keep doubling the runs per submit while one submit takes less than this.
// The last doubling then lands between 0.05 s and about 0.1 s, far below the limit.
const TARGET_SECONDS: f64 = 0.05;
// Never put more runs than this into one submit, even on a very fast GPU.
const MAX_RUNS_PER_SUBMIT: u32 = 4096;
// How many timed submits we keep.
const TIMED_SUBMITS: usize = 30;

// One kernel to time, and its name for error messages. One run is every dispatch in `dispatches`, in order.
pub struct Kernel<'a> {
    // The benchmark name, for error messages.
    pub name: &'static str,
    // The dispatches of one run, with their buffers.
    pub dispatches: &'a [Dispatch],
}

// The times we measured for one kernel.
pub struct Timing {
    // How many runs each timed submit held.
    pub runs_per_submit: u32,
    // The wall time of each timed submit, in seconds.
    pub seconds: Vec<f64>,
}

// Measure one kernel: one skipped submit, then calibration, then TIMED_SUBMITS timed submits.
pub async fn measure(gpu: &Gpu, kernel: &Kernel<'_>) -> Result<Timing, EngineError> {
    // Finish the buffer uploads first, so they are not part of any timed submit.
    gpu.finish_uploads().await?;
    // The first submit pays one-time costs, such as finishing the shader compile. We throw its time away.
    time_submit(gpu, kernel, 1, SubmitPhase::WarmUp).await?;
    // Find how many runs fit into one submit of about 0.05 to 0.1 s.
    let runs = calibrate(gpu, kernel).await?;
    // Room for every timed submit.
    let mut seconds = Vec::with_capacity(TIMED_SUBMITS);
    // Each await must finish before the next submit starts, so this is a plain loop.
    for submit in 1..=TIMED_SUBMITS {
        // Time one submit and keep the time.
        seconds.push(time_submit(gpu, kernel, runs, SubmitPhase::Timed(submit)).await?);
    }
    // All timed submits.
    Ok(Timing { runs_per_submit: runs, seconds })
}

// Start with 1 run per submit and double while a submit is shorter than TARGET_SECONDS.
async fn calibrate(gpu: &Gpu, kernel: &Kernel<'_>) -> Result<u32, EngineError> {
    // Start small. One run of each kernel is sized to be far below the limit.
    let mut runs = 1;
    // Each step times one submit and decides the next size.
    loop {
        // How long the current size takes.
        let seconds = time_submit(gpu, kernel, runs, SubmitPhase::Calibration).await?;
        // Double, or stop at the current size.
        match next_runs(runs, seconds) {
            // Try the larger size next.
            Some(next) => runs = next,
            // The current size is long enough.
            None => return Ok(runs),
        }
    }
}

// The next size to try: double, if the current submit was short and the double is allowed. None means stop.
fn next_runs(current: u32, seconds: f64) -> Option<u32> {
    // Double the count. None if it does not fit into u32.
    let doubled = current.checked_mul(2)?;
    // Only grow when the current submit is short and the new count is within the cap.
    (seconds < TARGET_SECONDS && doubled <= MAX_RUNS_PER_SUBMIT).then_some(doubled)
}

// Submit `runs` runs as one submit, and return how many seconds it took from submit to onSubmittedWorkDone.
async fn time_submit(gpu: &Gpu, kernel: &Kernel<'_>, runs: u32, phase: SubmitPhase) -> Result<f64, EngineError> {
    // Record the commands before the clock starts, so recording is not part of the time.
    let commands = encode_runs(gpu, kernel.dispatches, runs);
    // Submit, wait for the GPU, and read the clock.
    let seconds = timed_submit(gpu, commands).await?;
    // Stop everything if this submit broke the project limit.
    if seconds >= SUBMIT_LIMIT_SECONDS {
        // Say which submit it was and how long it took.
        return Err(EngineError::SubmitTooLong { benchmark: kernel.name, phase, runs, seconds });
    }
    // The time of this submit.
    Ok(seconds)
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function and the constants into the tests.
    use super::{MAX_RUNS_PER_SUBMIT, TARGET_SECONDS, next_runs};

    // This rule is what keeps every submit far below 0.25 s, so a mistake here can lose the device on the phone.
    #[test]
    fn doubles_only_while_short_and_under_the_cap() {
        // A short submit doubles.
        assert_eq!(next_runs(4, TARGET_SECONDS / 2.0), Some(8));
        // A submit at the target stops.
        assert_eq!(next_runs(4, TARGET_SECONDS), None);
        // A short submit at the cap stops too.
        assert_eq!(next_runs(MAX_RUNS_PER_SUBMIT, 0.0), None);
    }
}
