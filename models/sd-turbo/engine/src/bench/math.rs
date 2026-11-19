// The kinds of GPU buffer that the memory counter keeps apart.
use crate::memory::MemoryCategory;
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
const SHADER: &str = include_str!("math.wgsl");
// Invocations per dispatch: about one million, enough to keep every GPU core busy.
const INVOCATIONS: u32 = 1 << 20;
// Invocations per workgroup. It must match @workgroup_size(256) in math.wgsl.
const WORKGROUP_SIZE: u32 = 256;
// Operations per invocation: 256 steps x 4 fma x 2 operations. It must match math.wgsl.
const OPS_PER_INVOCATION: f64 = 2048.0;
// Start values are spread over 0.5 to 0.9. Below 1, so the loop in math.wgsl stays between 1 and 5.
const LOWEST_START: f32 = 0.5;
// The width of the spread of start values.
const START_SPREAD: f32 = 0.4;
// How many different start values we use before they repeat.
const DISTINCT_STARTS: u32 = 1000;

// Measure arithmetic speed: each invocation runs 1024 fma on its one value.
pub async fn math_benchmark(gpu: &Gpu) -> Result<BenchReport, EngineError> {
    // One start value per invocation, between 0.5 and 0.9.
    let starts: Vec<f32> = (0..INVOCATIONS)
        // Each index maps to one of DISTINCT_STARTS evenly spaced values.
        .map(|index| LOWEST_START + START_SPREAD * (index % DISTINCT_STARTS) as f32 / DISTINCT_STARTS as f32)
        // Put them into one list.
        .collect();
    // Copy the start values to the GPU (4 MiB).
    let input = gpu.storage_buffer_from("math input", &starts, MemoryCategory::Activations);
    // Room for the results (4 MiB).
    let output = gpu.empty_storage_buffer("math output", INVOCATIONS as usize);
    // 2^20 / 256 = 4096 workgroups.
    let workgroups = INVOCATIONS / WORKGROUP_SIZE;
    // Compile the shader. `input` goes to @binding(0) and `output` to @binding(1), as in math.wgsl.
    let dispatch = Dispatch::new(gpu, "math", SHADER, &[&input, &output], [workgroups, 1, 1], Work::f32_values("math", 2 * u64::from(INVOCATIONS)))?;
    // Time it.
    let timing = measure(gpu, &Kernel { name: "math", dispatches: std::slice::from_ref(&dispatch) }).await?;
    // 2^20 invocations x 2048 operations is about 2.1 billion operations per dispatch.
    let ops_per_dispatch = f64::from(INVOCATIONS) * OPS_PER_INVOCATION;
    // Turn the times into billions of operations per second.
    Ok(BenchReport::new("math", "G ops/s", ops_per_dispatch, &timing))
}
