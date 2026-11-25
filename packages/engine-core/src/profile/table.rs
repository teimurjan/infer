// Display prints the table.
use std::fmt;

// The timing mode and the totals of one kernel.
use super::{KernelTotals, ProfileMode};

// Bytes per second in one GB/s, as in PROGRESS.md.
const BILLION: f64 = 1e9;
// Milliseconds in one second.
const MILLISECONDS: f64 = 1000.0;

// One line of the table.
#[derive(Debug, PartialEq)]
pub struct ProfileRow {
    // The kernel name.
    pub kernel: &'static str,
    // Its dispatches.
    pub calls: u64,
    // Its time, in milliseconds.
    pub total_ms: f64,
    // Its part of the time of all kernels, from 0 to 1.
    pub time_share: f64,
    // Its bytes over its time, in GB/s. None when its time is 0, which a coarse clock can give.
    pub gbps: Option<f64>,
    // Its speed over the door number, from 0 to 1 (and above 1 if a kernel beats the copy benchmark). None as above.
    pub door_share: Option<f64>,
}

// The whole table: the rows, slowest first, and what they were measured with.
pub struct ProfileTable {
    // How each dispatch was timed.
    mode: ProfileMode,
    // The door number, GB/s.
    door_gbps: f64,
    // The rows, slowest first.
    rows: Vec<ProfileRow>,
}

impl ProfileTable {
    // Turn the totals of each kernel into rows, slowest first.
    pub(super) fn new(mode: ProfileMode, door_gbps: f64, kernels: impl Iterator<Item = (&'static str, KernelTotals)>) -> Self {
        // Keep the totals, to sum the time before the shares.
        let kernels: Vec<(&'static str, KernelTotals)> = kernels.collect();
        // The time of all kernels.
        let all_seconds: f64 = kernels.iter().map(|(_, totals)| totals.seconds).sum();
        // One row per kernel.
        let mut rows: Vec<ProfileRow> = kernels.into_iter().map(|(kernel, totals)| row(kernel, totals, all_seconds, door_gbps)).collect();
        // The slowest kernel first. A tie keeps the name order of the map.
        rows.sort_by(|first, second| second.total_ms.total_cmp(&first.total_ms));
        // The table.
        Self { mode, door_gbps, rows }
    }

    // The rows, slowest first.
    pub fn rows(&self) -> &[ProfileRow] {
        &self.rows
    }
}

// The row of one kernel. `all_seconds` is the time of all kernels.
fn row(kernel: &'static str, totals: KernelTotals, all_seconds: f64, door_gbps: f64) -> ProfileRow {
    // Bytes over seconds, if the clock saw any time.
    let gbps = (totals.seconds > 0.0).then(|| totals.bytes as f64 / totals.seconds / BILLION);
    // The share of time. All kernels at 0 seconds give 0, not a division by 0.
    let time_share = if all_seconds > 0.0 { totals.seconds / all_seconds } else { 0.0 };
    // The row.
    ProfileRow { kernel, calls: totals.calls, total_ms: totals.seconds * MILLISECONDS, time_share, gbps, door_share: gbps.map(|speed| speed / door_gbps) }
}

impl fmt::Display for ProfileTable {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // What the numbers come from.
        let timing = match self.mode {
            ProfileMode::Timestamps => "GPU timestamps, one compute pass per dispatch",
            ProfileMode::WallClock => "CPU clock around one awaited submit per dispatch",
        };
        writeln!(formatter, "timing: {timing}; door: {:.2} GB/s", self.door_gbps)?;
        // The header. Each column has a fixed width, so the numbers line up.
        writeln!(formatter, "{:<16} {:>6} {:>10} {:>7} {:>8} {:>7}", "kernel", "calls", "total ms", "time", "GB/s", "door")?;
        // One line per kernel.
        for row in &self.rows {
            // A kernel with no time on the clock has no speed.
            let gbps = row.gbps.map_or("-".to_string(), |speed| format!("{speed:.2}"));
            let door = row.door_share.map_or("-".to_string(), |share| format!("{:.1}%", share * 100.0));
            writeln!(formatter, "{:<16} {:>6} {:>10.2} {:>6.1}% {:>8} {:>7}", row.kernel, row.calls, row.total_ms, row.time_share * 100.0, gbps, door)?;
        }
        // The sums.
        let calls: u64 = self.rows.iter().map(|row| row.calls).sum();
        let total_ms: f64 = self.rows.iter().map(|row| row.total_ms).sum();
        write!(formatter, "{:<16} {:>6} {:>10.2} {:>6.1}%", "all kernels", calls, total_ms, 100.0)
    }
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the table into the tests.
    use super::{KernelTotals, ProfileMode, ProfileRow, ProfileTable};

    // The rows must be slowest first, and each share must use the right total, or the user would fix the wrong kernel.
    #[test]
    fn rows_are_slowest_first_with_shares_of_time_and_door() {
        // conv: 3 s for 120 GB. add: 1 s for 20 GB. idle: no time on the clock.
        let kernels = [
            ("add", KernelTotals { calls: 4, seconds: 1.0, bytes: 20_000_000_000 }),
            ("conv", KernelTotals { calls: 2, seconds: 3.0, bytes: 120_000_000_000 }),
            ("idle", KernelTotals { calls: 1, seconds: 0.0, bytes: 8 }),
        ];
        // The door is 80 GB/s.
        let table = ProfileTable::new(ProfileMode::Timestamps, 80.0, kernels.into_iter());
        // conv: 40 GB/s is half the door, and 3 of 4 seconds.
        assert_eq!(table.rows()[0], ProfileRow { kernel: "conv", calls: 2, total_ms: 3000.0, time_share: 0.75, gbps: Some(40.0), door_share: Some(0.5) });
        // add: 20 GB/s is a quarter of the door, and 1 of 4 seconds.
        assert_eq!(table.rows()[1], ProfileRow { kernel: "add", calls: 4, total_ms: 1000.0, time_share: 0.25, gbps: Some(20.0), door_share: Some(0.25) });
        // idle: no speed.
        assert_eq!(table.rows()[2], ProfileRow { kernel: "idle", calls: 1, total_ms: 0.0, time_share: 0.0, gbps: None, door_share: None });
    }
}
