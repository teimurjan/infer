// Display lets us print a report as one line.
use std::fmt;

// The times from measure.rs.
use super::measure::Timing;

// One billion. Speeds are shown in billions per second (GB/s and G ops/s).
const BILLION: f64 = 1e9;

// The speeds of one benchmark.
pub struct BenchReport {
    // The benchmark name, for example "copy".
    name: &'static str,
    // The unit of the speed, for example "GB/s".
    unit: &'static str,
    // The speed of the middle submit, in billions per second. It ignores a few slow or fast outliers.
    median_rate: f64,
    // The speed of the fastest submit, in billions per second.
    best_rate: f64,
    // How many timed submits.
    submits: usize,
    // How many runs each submit held.
    runs_per_submit: u32,
    // The middle submit time, in milliseconds.
    median_ms: f64,
    // The longest submit time, in milliseconds. It must stay under 250.
    longest_ms: f64,
}

impl BenchReport {
    // The median rate, in the unit of the report: GB/s for copy, G ops/s for math.
    pub fn median_rate(&self) -> f64 {
        self.median_rate
    }

    // Turn the times into speeds. `work_per_run` is bytes for copy and operations for math.
    pub(crate) fn new(name: &'static str, unit: &'static str, work_per_run: f64, timing: &Timing) -> Self {
        // The work in one submit.
        let work_per_submit = work_per_run * f64::from(timing.runs_per_submit);
        // The middle time.
        let median_seconds = median(&timing.seconds);
        // The shortest time gives the best speed.
        let shortest_seconds = timing.seconds.iter().copied().fold(f64::INFINITY, f64::min);
        // The longest time shows how close we came to the limit.
        let longest_seconds = timing.seconds.iter().copied().fold(0.0, f64::max);
        // Work divided by time, in billions per second.
        Self {
            name,
            unit,
            median_rate: work_per_submit / median_seconds / BILLION,
            best_rate: work_per_submit / shortest_seconds / BILLION,
            submits: timing.seconds.len(),
            runs_per_submit: timing.runs_per_submit,
            median_ms: median_seconds * 1000.0,
            longest_ms: longest_seconds * 1000.0,
        }
    }
}

// One line, for example "copy: 90.12 GB/s median, 92.30 GB/s best (30 submits, 32 runs per submit, 60.1 ms median submit, 62.0 ms longest)".
impl fmt::Display for BenchReport {
    // Write the line.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // Speeds first, then the facts that show how they were measured.
        write!(
            formatter,
            "{}: {:.2} {} median, {:.2} {} best ({} submits, {} runs per submit, {:.1} ms median submit, {:.1} ms longest)",
            self.name, self.median_rate, self.unit, self.best_rate, self.unit, self.submits, self.runs_per_submit, self.median_ms, self.longest_ms
        )
    }
}

// The middle value of a list. For an even count, the average of the two middle values.
fn median(values: &[f64]) -> f64 {
    // Sort a copy, so the caller's list does not change. total_cmp sorts f64 values safely.
    let sorted = {
        let mut copy = values.to_vec();
        copy.sort_by(f64::total_cmp);
        copy
    };
    // The position just past the middle.
    let middle = sorted.len() / 2;
    // Odd count: one middle value. Even count: average the two middle values.
    if sorted.len() % 2 == 1 { sorted[middle] } else { (sorted[middle - 1] + sorted[middle]) / 2.0 }
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function into the tests.
    use super::median;

    // The even case is the easy one to get wrong, and TIMED_SUBMITS is even.
    #[test]
    fn median_of_an_even_count_averages_the_two_middle_values() {
        // Unsorted on purpose. Sorted it is 1, 2, 4, 9, so the middle values are 2 and 4.
        assert_eq!(median(&[9.0, 1.0, 4.0, 2.0]), 3.0);
    }
}
