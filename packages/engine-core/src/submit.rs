// A clock that works natively and in the browser.
use web_time::Instant;

// A kernel ready to run, and the function that records runs of it.
use crate::dispatch::{Dispatch, encode_runs};
// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;
// The profiled submit, which times each dispatch.
use crate::profile::profiled_submit;

// The project limit for one submit. iOS loses the device near 0.5 s.
pub const SUBMIT_LIMIT_SECONDS: f64 = 0.25;

// Send one command list to the GPU, wait until it is done, and return the seconds from submit to done.
pub async fn timed_submit(gpu: &Gpu, commands: wgpu::CommandBuffer) -> Result<f64, EngineError> {
    // Start the clock after the commands are recorded, so recording is not part of the time.
    let start = Instant::now();
    // Send the commands to the GPU.
    gpu.queue.submit([commands]);
    // Wait until the GPU says that all submitted work is done.
    gpu.wait_for_submitted_work().await?;
    // Stop the clock.
    Ok(start.elapsed().as_secs_f64())
}

// Run `dispatches` once, in order, as one submit. Fail if the submit broke the project limit.
// While a profile runs, the profiler times each dispatch, and every submit it makes is checked against the limit too.
pub async fn submit_within_limit(gpu: &Gpu, label: &str, dispatches: &[Dispatch]) -> Result<f64, EngineError> {
    // In the browser, the GPU process compiles new shaders and copies new buffers in the order it gets them.
    // Wait for that work first, so the timed submit holds only the kernels. In Chromium, the first TAESD submit took 590.9 ms without this wait.
    gpu.finish_uploads().await?;
    // Pick the plain submit or the profiled one.
    match gpu.profile_timer() {
        // One run of every dispatch, in one command list.
        None => check_submit(gpu, label, timed_submit(gpu, encode_runs(gpu, dispatches, 1)).await?),
        // Time each dispatch and add it to the profile.
        Some(timer) => profiled_submit(gpu, &timer, label, dispatches).await,
    }
}

// Keep the time of one submit for the page, and fail if it broke the project limit. Returns the same seconds.
pub fn check_submit(gpu: &Gpu, label: &str, seconds: f64) -> Result<f64, EngineError> {
    // Keep the longest time, so the page can show it.
    gpu.record_submit(seconds);
    // A submit at or past the limit could lose the device on the iPhone.
    if seconds >= SUBMIT_LIMIT_SECONDS {
        // Say which submit and how long it took.
        return Err(EngineError::SubmitOverLimit { label: label.to_string(), seconds });
    }
    // The time of this submit.
    Ok(seconds)
}
