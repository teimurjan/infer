// The kinds of GPU buffer that the memory counter keeps apart.
use crate::memory::MemoryCategory;
// A kernel ready to run, and the function that records runs of it.
use crate::dispatch::{Dispatch, Work, encode_runs};
// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The function that compares two lists.
use crate::max_error::max_abs_error;

// The WGSL source, copied into the program when it compiles.
const SHADER: &str = include_str!("silu.wgsl");
// Invocations per workgroup. It must match @workgroup_size(64) in silu.wgsl.
const WORKGROUP_SIZE: u32 = 64;

// Run SiLU on the GPU for every value in `input`, and return the results.
pub async fn run_silu(gpu: &Gpu, input: &[f32]) -> Result<Vec<f32>, EngineError> {
    // How many workgroups to start. Fails fast if the input is empty or too big.
    let workgroups = workgroup_count(input.len(), gpu.max_workgroups_per_dispatch())?;
    // Copy the input to the GPU.
    let input_buffer = gpu.storage_buffer_from("silu input", input, MemoryCategory::Activations);
    // Make room for the output on the GPU.
    let output_buffer = gpu.empty_storage_buffer("silu output", input.len());
    // Compile the shader. `input` goes to @binding(0) and `output` to @binding(1), as in silu.wgsl.
    let dispatch = Dispatch::new(gpu, "silu", SHADER, &[&input_buffer, &output_buffer], [workgroups, 1, 1], Work::f32_values("silu", 2 * input.len() as u64))?;
    // Record one dispatch and send it to the GPU.
    gpu.queue.submit([encode_runs(gpu, std::slice::from_ref(&dispatch), 1)]);
    // Copy the output back and return it.
    gpu.read_f32(&output_buffer).await
}

// Open the GPU, run SiLU on `input`, and return the largest difference from `expected`.
// The native test and the web page both call this, so they measure the same thing.
pub async fn silu_max_error(input: &[f32], expected: &[f32]) -> Result<f32, EngineError> {
    // Open the GPU.
    let gpu = Gpu::new().await?;
    // Run the kernel.
    let output = run_silu(&gpu, input).await?;
    // Compare with the PyTorch result.
    max_abs_error(&output, expected)
}

// The number of workgroups that covers `count` values, WORKGROUP_SIZE values per workgroup.
fn workgroup_count(count: usize, max_workgroups: u32) -> Result<u32, EngineError> {
    // The GPU cannot read back a buffer of size 0.
    if count == 0 {
        // Stop before we touch the GPU.
        return Err(EngineError::EmptyInput);
    }
    // Divide and round up, so the last few values also get a workgroup.
    let needed = count.div_ceil(WORKGROUP_SIZE as usize);
    // The most values that one dispatch can cover.
    let max_values = max_workgroups as usize * WORKGROUP_SIZE as usize;
    // Turn the count into u32. It fails only if it is bigger than the limit too.
    u32::try_from(needed)
        // Keep it only if it is within the limit.
        .ok()
        .filter(|workgroups| *workgroups <= max_workgroups)
        // Otherwise say how many values one dispatch can do.
        .ok_or(EngineError::TooManyValues { count, max: max_values })
}

// Tests run with `cargo test`.
#[cfg(test)]
mod tests {
    // Bring the function and the error into the tests.
    use super::{EngineError, workgroup_count};

    // Rounding up is easy to get wrong by one, and a missing workgroup leaves output values at 0.
    #[test]
    fn rounds_up_to_whole_workgroups() {
        // 64 values fill exactly one workgroup.
        assert_eq!(workgroup_count(64, 65535).unwrap(), 1);
        // One more value needs a second workgroup.
        assert_eq!(workgroup_count(65, 65535).unwrap(), 2);
    }

    // Asking for too many workgroups would fail inside the GPU with a less clear message.
    #[test]
    fn rejects_more_than_one_dispatch_can_do() {
        // 2 workgroups of 64 is 128 values. 129 values need 3 workgroups.
        assert!(matches!(workgroup_count(129, 2), Err(EngineError::TooManyValues { count: 129, max: 128 })));
    }
}
