// category.rs lists the kinds of buffer that the counter keeps apart.
mod category;
// counter.rs counts the live bytes per category and the peak.
mod counter;
// report.rs is a snapshot of the counter that the page and the tests print.
mod report;

// Deref lets a GpuBuffer be used wherever a &wgpu::Buffer is needed.
use std::ops::Deref;
// Rc shares one counter between the Gpu and every buffer.
use std::rc::Rc;

// The parts that other modules use.
pub use category::MemoryCategory;
pub(crate) use counter::MemoryCounter;
pub use report::MemoryReport;

// A GPU buffer that adds its size to its category when it is made, and removes it when Rust drops it.
// Dropping a wgpu buffer frees its GPU memory (in the browser, wgpu calls destroy()).
pub(crate) struct GpuBuffer {
    // The real buffer.
    buffer: wgpu::Buffer,
    // The category that counts this buffer.
    category: MemoryCategory,
    // The counter to update when this buffer is dropped.
    counter: Rc<MemoryCounter>,
}

impl GpuBuffer {
    // Start counting a buffer that was just made.
    pub(crate) fn new(buffer: wgpu::Buffer, category: MemoryCategory, counter: &Rc<MemoryCounter>) -> Self {
        // Its bytes are live from now on.
        counter.add(category, buffer.size());
        // Keep the buffer, its category and a handle to the counter.
        Self { buffer, category, counter: Rc::clone(counter) }
    }
}

// `&gpu_buffer` works wherever wgpu expects `&wgpu::Buffer`.
impl Deref for GpuBuffer {
    // The type we point to.
    type Target = wgpu::Buffer;

    // Give the real buffer.
    fn deref(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

// Runs when the buffer goes out of scope.
impl Drop for GpuBuffer {
    // The buffer is freed right after this, so stop counting its bytes.
    fn drop(&mut self) {
        self.counter.remove(self.category, self.buffer.size());
    }
}
