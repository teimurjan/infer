// copy.rs measures memory speed: read one large buffer, write another.
mod copy;
// math.rs measures arithmetic speed: a long loop of multiply-adds on one value per invocation.
mod math;
// measure.rs times submits: skip the first, find a safe size, repeat.
mod measure;
// report.rs turns the times into speeds and one printable line.
mod report;

// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;

// The result type that the native example and the web page print.
pub use report::BenchReport;
// The timing loop, which the layer tests use too.
pub(crate) use measure::{Kernel, measure};
// The copy benchmark alone. The profiler uses its median as the door number.
pub use copy::copy_benchmark;

// Run the benchmarks, one after the other, and return one report each.
pub async fn run_benchmarks(gpu: &Gpu) -> Result<Vec<BenchReport>, EngineError> {
    // Memory speed first.
    let copy = copy::copy_benchmark(gpu).await?;
    // Then float32 arithmetic speed.
    let math = math::math_benchmark(gpu).await?;
    // Float16 arithmetic speed, only where the device has shader-f16.
    if !gpu.has_shader_f16() {
        return Ok(vec![copy, math]);
    }
    let math_f16 = math::math_f16_benchmark(gpu).await?;
    // The reports, in the order they ran.
    Ok(vec![copy, math, math_f16])
}
