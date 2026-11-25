// scope.rs turns GPU errors inside a profiled submit into engine errors.
mod scope;
// table.rs turns the totals into the printed table.
mod table;
// timestamps.rs times each dispatch with GPU timestamps, inside one submit.
mod timestamps;
// wall_clock.rs times each dispatch as its own awaited submit, for devices without timestamps.
mod wall_clock;

// A sorted map keeps one total per kernel name.
use std::collections::BTreeMap;
// Rc lets a submit hold the timestamp buffers while the profile stays in the Gpu.
use std::rc::Rc;

// The copy benchmark of module 3, which measures the door number.
use crate::bench::{BenchReport, copy_benchmark};
// A kernel ready to record, and what it does.
use crate::dispatch::{Dispatch, Work};
// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;

// The table rows, for the tests and the page.
pub use table::{ProfileRow, ProfileTable};
// The GPU error check.
use scope::caught;
// The query set and its buffer.
use timestamps::Timestamps;

// How the profiler times one dispatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProfileMode {
    // GPU timestamps at the start and the end of each dispatch. Needs the "timestamp-query" feature.
    Timestamps,
    // The CPU clock around one awaited submit per dispatch. Every device can do this, but each time includes the submit and the wait.
    WallClock,
}

// What times the dispatches of a running profile. A clone is a second handle to the same query set.
#[derive(Clone)]
pub enum Timer {
    // GPU timestamps, written into this query set.
    Timestamps(Rc<Timestamps>),
    // The CPU clock.
    WallClock,
}

impl Timer {
    // The mode that this timer uses, for the table.
    fn mode(&self) -> ProfileMode {
        match self {
            Self::Timestamps(_) => ProfileMode::Timestamps,
            Self::WallClock => ProfileMode::WallClock,
        }
    }
}

// The sums of one kernel name over a run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct KernelTotals {
    // Dispatches of this kernel.
    calls: u64,
    // Seconds of all its dispatches.
    seconds: f64,
    // Bytes that all its dispatches must read and write at least.
    bytes: u64,
}

// The time of every kernel since the profile started, and the door number to compare with.
pub struct Profile {
    // What times each dispatch.
    timer: Timer,
    // The copy speed of this device in GB/s, from the module 3 benchmark.
    door_gbps: f64,
    // One total per kernel name.
    kernels: BTreeMap<&'static str, KernelTotals>,
}

impl Profile {
    // An empty profile.
    pub fn new(timer: Timer, door_gbps: f64) -> Self {
        Self { timer, door_gbps, kernels: BTreeMap::new() }
    }

    // What times each dispatch.
    pub fn timer(&self) -> Timer {
        self.timer.clone()
    }

    // Add one dispatch that took `seconds`.
    pub fn add(&mut self, work: Work, seconds: f64) {
        // The total of this kernel name, made at its first dispatch.
        let totals = self.kernels.entry(work.kernel).or_default();
        // One more call, its time and its bytes.
        totals.calls += 1;
        totals.seconds += seconds;
        totals.bytes += work.bytes;
    }

    // The table, slowest kernel first.
    pub fn table(&self) -> ProfileTable {
        ProfileTable::new(self.timer.mode(), self.door_gbps, self.kernels.iter().map(|(&kernel, totals)| (kernel, *totals)))
    }
}

// Measure the door number with the copy benchmark of module 3, then start a profile on `gpu` that compares with it.
// Returns the copy report, so the caller can print the measured line.
pub async fn start_profile(gpu: &Gpu, mode: ProfileMode) -> Result<BenchReport, EngineError> {
    // Timestamps need the device feature. Fail before the benchmark if it is missing.
    if mode == ProfileMode::Timestamps && !gpu.has_timestamps() {
        return Err(EngineError::NoTimestamps);
    }
    // The same benchmark as module 3. Its buffers are freed when it returns.
    let door = copy_benchmark(gpu).await?;
    // One query set for the whole profile, or the CPU clock.
    let timer = match mode {
        ProfileMode::Timestamps => Timer::Timestamps(Rc::new(Timestamps::new(gpu).await?)),
        ProfileMode::WallClock => Timer::WallClock,
    };
    // Every later submit_within_limit adds its dispatches to this profile.
    gpu.set_profile(Some(Profile::new(timer, door.median_rate())));
    // The copy line.
    Ok(door)
}

// The best mode of this device: timestamps when it has them.
pub fn best_mode(gpu: &Gpu) -> ProfileMode {
    if gpu.has_timestamps() { ProfileMode::Timestamps } else { ProfileMode::WallClock }
}

// Run `dispatches` with timing per dispatch, add each time to the profile, and return the seconds of the step.
// Each submit is checked against the project limit. A GPU error in the step stops the run, so a skipped submit cannot give a wrong table or image.
pub async fn profiled_submit(gpu: &Gpu, timer: &Timer, label: &str, dispatches: &[Dispatch]) -> Result<f64, EngineError> {
    // Each timer gives the seconds of the whole step and the seconds of every dispatch.
    let timed = async {
        match timer {
            // One submit, timed by the GPU per dispatch.
            Timer::Timestamps(timestamps) => timestamps::time_dispatches(gpu, timestamps, label, dispatches).await,
            // One submit per dispatch, timed by the CPU.
            Timer::WallClock => wall_clock::time_dispatches(gpu, label, dispatches).await,
        }
    };
    // Catch GPU errors of the step.
    let (step_seconds, kernel_seconds) = caught(gpu, label, timed).await?;
    // Each dispatch with its time.
    gpu.add_to_profile(dispatches.iter().map(Dispatch::work).zip(kernel_seconds));
    // The step time.
    Ok(step_seconds)
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the profile into the tests.
    use super::{KernelTotals, Profile, Timer, Work};

    // Two dispatches of one kernel must become one total, or the table would show the same kernel twice.
    #[test]
    fn dispatches_of_one_kernel_add_up() {
        // An empty profile.
        let mut profile = Profile::new(Timer::WallClock, 50.0);
        // Two conv dispatches and one add.
        // The times are powers of 2, so their sum is exact in f64.
        profile.add(Work { kernel: "conv3x3", bytes: 100 }, 0.25);
        profile.add(Work { kernel: "add", bytes: 10 }, 0.125);
        profile.add(Work { kernel: "conv3x3", bytes: 300 }, 0.5);
        // The conv totals.
        assert_eq!(profile.kernels["conv3x3"], KernelTotals { calls: 2, seconds: 0.75, bytes: 400 });
        // Two names in total.
        assert_eq!(profile.kernels.len(), 2);
    }
}
