// Our helper that turns f32 values into bytes.
use crate::bytes::f32_to_le_bytes;
// A kernel ready to run.
use crate::dispatch::{Dispatch, Work};
// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;

// The shared timing loop.
use super::measure::{Kernel, measure};
// The report type.
use super::report::BenchReport;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("copy.wgsl");
// Size of each buffer: 64 MiB. It must be much larger than the GPU caches, so we measure main memory.
const BUFFER_BYTES: u64 = 64 << 20;
// One vec4<f32> is 16 bytes. Each invocation copies one.
const VEC4_BYTES: u64 = 16;
// Invocations per workgroup. It must match @workgroup_size(256) in copy.wgsl.
const WORKGROUP_SIZE: u64 = 256;
// We fill the source 1 MiB at a time, so the CPU never holds a 64 MiB copy.
const FILL_CHUNK_VALUES: usize = (1 << 20) / 4;

// Measure memory speed: each dispatch reads 64 MiB from one buffer and writes 64 MiB to another.
pub async fn copy_benchmark(gpu: &Gpu) -> Result<BenchReport, EngineError> {
    // f32 values in one buffer.
    let values = (BUFFER_BYTES / 4) as usize;
    // The buffer we read.
    let source = gpu.empty_storage_buffer("copy source", values);
    // Put non-zero values in it, so the GPU cannot treat it as a special all-zero buffer.
    fill_with_pattern(gpu, &source);
    // The buffer we write.
    let destination = gpu.empty_storage_buffer("copy destination", values);
    // 64 MiB / 16 bytes / 256 = 16384 workgroups, under the 65535 limit.
    let workgroups = (BUFFER_BYTES / VEC4_BYTES / WORKGROUP_SIZE) as u32;
    // Compile the shader. `source` goes to @binding(0) and `destination` to @binding(1), as in copy.wgsl.
    let dispatch = Dispatch::new(gpu, "copy", SHADER, &[&source, &destination], [workgroups, 1, 1], Work { kernel: "copy", bytes: 2 * BUFFER_BYTES })?;
    // Time it.
    let timing = measure(gpu, &Kernel { name: "copy", dispatches: std::slice::from_ref(&dispatch) }).await?;
    // Every dispatch reads the whole source and writes the whole destination, so it moves 2 x 64 MiB.
    let bytes_per_dispatch = 2.0 * BUFFER_BYTES as f64;
    // Turn the times into GB/s.
    Ok(BenchReport::new("copy", "GB/s", bytes_per_dispatch, &timing))
}

// Write a repeating pattern of 0, 1, 2, ... into the whole buffer, one chunk at a time.
fn fill_with_pattern(gpu: &Gpu, buffer: &wgpu::Buffer) {
    // One chunk of the pattern, as bytes.
    let chunk = f32_to_le_bytes(&(0..FILL_CHUNK_VALUES).map(|index| index as f32).collect::<Vec<_>>());
    // Write the chunk at offset 0, 1 MiB, 2 MiB, and so on. 64 MiB is a whole number of chunks.
    // The queue holds these writes and does them before the next submit.
    (0..buffer.size()).step_by(chunk.len()).for_each(|offset| gpu.queue.write_buffer(buffer, offset, &chunk));
}
