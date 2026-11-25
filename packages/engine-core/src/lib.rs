// The GPU base that every model engine in models/ shares. A model engine adds its own kernels and stages on top.

// Little-endian bytes to f32 values.
pub mod bytes;
// Size checks that many kernels share.
pub mod checks;
// The error type of every engine.
pub mod error;
// The GPU connection: device, queue, buffers, memory counter and profile hooks.
pub mod gpu;
// A kernel with its buffers, ready to record.
pub mod dispatch;
// One submit, timed and checked against the 0.25 s limit.
pub mod submit;
// The largest difference between two lists of values.
pub mod max_error;
// GPU memory by category, and the peak.
pub mod memory;
// The time of each kernel, with GPU timestamps or the CPU clock.
pub mod profile;
// Copy and math speed of the GPU.
pub mod bench;
// The activation planner: tensor lifetimes, slots and the buffer pool.
pub mod plan;
// Weight files: the index, streaming upload, and the buffers that kernels bind.
pub mod weights;
// Linear, 1x1 and 3x3 convolution layers, float and packed, with their kernels.
pub mod layers;
