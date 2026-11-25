// Display lets the page and the tests print the report.
use std::fmt;

// The kinds of buffer.
use super::MemoryCategory;

// Bytes in one MB. PROGRESS.md uses MB = 10^6 bytes.
const MEGABYTE: f64 = 1e6;

// The bytes of the engine's GPU buffers at one moment, per category.
pub struct MemoryReport {
    // Each category and its live bytes, in the order of MemoryCategory::ALL.
    pub(super) live: [(MemoryCategory, u64); MemoryCategory::ALL.len()],
    // The most bytes held at one time since the last reset, over all categories.
    pub(super) peak: u64,
}

impl MemoryReport {
    // The live bytes of one category.
    pub fn bytes(&self, category: MemoryCategory) -> u64 {
        // The entry at the slot of this category.
        self.live[category.slot()].1
    }

    // The most bytes held at one time since the last reset.
    pub fn peak(&self) -> u64 {
        self.peak
    }

    // The live bytes of all categories together.
    pub fn total(&self) -> u64 {
        self.live.iter().map(|(_, bytes)| bytes).sum()
    }
}

// One line per category, then the total and the peak, for example "weights: 9780736 bytes (9.8 MB)".
impl fmt::Display for MemoryReport {
    // Write the lines.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // One line per category.
        for (category, bytes) in self.live {
            writeln!(formatter, "{}: {}", category.name(), with_megabytes(bytes))?;
        }
        // The sum of the lines above.
        writeln!(formatter, "total: {}", with_megabytes(self.total()))?;
        // The high point. No newline after the last line.
        write!(formatter, "peak: {}", with_megabytes(self.peak))
    }
}

// The exact byte count, and the same number in MB with one decimal.
fn with_megabytes(bytes: u64) -> String {
    format!("{bytes} bytes ({:.1} MB)", bytes as f64 / MEGABYTE)
}
