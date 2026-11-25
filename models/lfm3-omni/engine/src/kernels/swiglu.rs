// Our GPU connection, a kernel with its buffers, and our error type.
use engine_core::dispatch::{BoundKernel, Dispatch, Work};
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;

// The shared workgroup size.
use super::WORKGROUP_SIZE;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("swiglu.wgsl");

// Bind gate = silu(gate) * up over `values` values.
pub fn bind_swiglu(gpu: &Gpu, values: u32, gate: &wgpu::Buffer, up: &wgpu::Buffer) -> Result<BoundKernel, EngineError> {
    // It reads both and writes the gate.
    let work = Work::f32_values("swiglu", 3 * u64::from(values));
    // One invocation per value.
    let workgroups = [values.div_ceil(WORKGROUP_SIZE), 1, 1];
    let dispatch = Dispatch::new(gpu, "swiglu", SHADER, &[gate, up], workgroups, work)?;
    // No sizes buffer: the kernel takes the count from the buffer length.
    Ok(BoundKernel::new(vec![dispatch], Vec::new()))
}
