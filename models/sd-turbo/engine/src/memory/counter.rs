// Cell holds a number that we change through a shared reference.
use std::cell::Cell;

// The kinds of buffer, and the snapshot that the counter gives out.
use super::{MemoryCategory, MemoryReport};

// The number of categories, so the counter has one number per category.
const CATEGORIES: usize = MemoryCategory::ALL.len();

// Counts the bytes of every GPU buffer that the engine holds right now, per category, and the most it ever held in total.
#[derive(Default)]
pub(crate) struct MemoryCounter {
    // Bytes in buffers that exist now, one number per category, in the order of MemoryCategory::ALL.
    live: [Cell<u64>; CATEGORIES],
    // The largest total that `live` reached since the last reset.
    peak: Cell<u64>,
    // The number of buffers made so far. A test uses it to prove that a run makes only its read-back copies.
    made: Cell<u64>,
}

impl MemoryCounter {
    // A buffer of this category was made.
    pub(super) fn add(&self, category: MemoryCategory, bytes: u64) {
        // The number of this category.
        let live = &self.live[category.slot()];
        // More bytes are live now.
        live.set(live.get() + bytes);
        // One more buffer.
        self.made.set(self.made.get() + 1);
        // Remember the new high point of the total.
        self.peak.set(self.peak.get().max(self.total()));
    }

    // A buffer of this category was freed.
    pub(super) fn remove(&self, category: MemoryCategory, bytes: u64) {
        // The number of this category.
        let live = &self.live[category.slot()];
        // Fewer bytes are live now. The peak stays.
        live.set(live.get() - bytes);
    }

    // The bytes of all categories together.
    fn total(&self) -> u64 {
        self.live.iter().map(Cell::get).sum()
    }

    // The number of buffers made so far, of every category.
    pub(crate) fn buffers_made(&self) -> u64 {
        self.made.get()
    }

    // The most bytes held at one time since the last reset.
    pub(crate) fn peak(&self) -> u64 {
        self.peak.get()
    }

    // Start a new measurement from the bytes held now.
    pub(crate) fn reset_peak(&self) {
        self.peak.set(self.total());
    }

    // A copy of the numbers right now, for a report.
    pub(crate) fn report(&self) -> MemoryReport {
        // Pair each category with its number.
        MemoryReport { live: MemoryCategory::ALL.map(|category| (category, self.live[category.slot()].get())), peak: self.peak() }
    }
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the counter and the categories into the tests.
    use super::{MemoryCategory, MemoryCounter};

    // The peak must keep the high point after buffers are freed, or the attention test would print a too-small number.
    #[test]
    fn peak_keeps_the_high_point_until_reset() {
        // A fresh counter.
        let counter = MemoryCounter::default();
        // Two buffers of two categories at the same time: 100 + 50 bytes.
        counter.add(MemoryCategory::Weights, 100);
        counter.add(MemoryCategory::Activations, 50);
        // Both are freed.
        counter.remove(MemoryCategory::Weights, 100);
        counter.remove(MemoryCategory::Activations, 50);
        // The high point was 150, over both categories.
        assert_eq!(counter.peak(), 150);
        // A reset starts again from the live bytes, which are 0 now.
        counter.reset_peak();
        assert_eq!(counter.peak(), 0);
    }

    // Each category must count only its own buffers.
    #[test]
    fn categories_count_apart() {
        // A fresh counter.
        let counter = MemoryCounter::default();
        // Two weight buffers and one uniform buffer.
        counter.add(MemoryCategory::Weights, 1000);
        counter.add(MemoryCategory::Weights, 24);
        counter.add(MemoryCategory::Uniforms, 16);
        // One weight buffer is freed.
        counter.remove(MemoryCategory::Weights, 24);
        // The snapshot.
        let report = counter.report();
        // Each category has its own number.
        assert_eq!(report.bytes(MemoryCategory::Weights), 1000);
        assert_eq!(report.bytes(MemoryCategory::Uniforms), 16);
        assert_eq!(report.bytes(MemoryCategory::Activations), 0);
        // The total is the sum.
        assert_eq!(report.total(), 1016);
    }
}
