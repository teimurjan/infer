// The packed linear kernel of engine-core with one float16 scale per 32 values, the 4-bit format of LFM3, on two real
// weights of layer 0. PyTorch computed the expected outputs in float64 with the same rounded weights, so the test measures
// only the error of the kernel. It needs `uv run engine/testdata/make_grouped_linear.py` in the model folder.

// The shared helpers of the tests.
mod common;

use common::model_folder;
use compare::safetensors::read_tensors;
use engine_core::dispatch::encode_runs;
use engine_core::gpu::Gpu;
use engine_core::memory::MemoryCategory;
use engine_core::weights::{GpuWeights, load_to_gpu};
use lfm3_engine::bind_linear;

// The two weights and the two row counts of make_grouped_linear.py.
const WEIGHTS: [&str; 2] = ["conv.in_proj", "feed_forward.w2"];
const ROW_COUNTS: [u32; 2] = [1, 9];
// The largest difference we accept. The outputs reach about 16, and one output adds up to 7680 products in float32.
// A scale read from the wrong group or row would change the outputs by far more.
const TOLERANCE: f32 = 1e-4;

// Run one case: bind the weight, run it on the input, and give back the largest difference from PyTorch.
async fn run_case(gpu: &Gpu, weights: &GpuWeights, weight: &str, rows: u32) -> f32 {
    let path = model_folder().join(format!("engine/testdata/grouped_linear/{weight}_{rows}.safetensors"));
    let tensors = read_tensors(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let find = |name: &str| tensors.iter().find(|tensor| tensor.name == name).unwrap_or_else(|| panic!("no {name} in {}", path.display()));
    let (input, expected) = (find("input"), find("expected"));
    let input_buffer = gpu.storage_buffer_from("input", &input.values, MemoryCategory::Activations);
    let output = gpu.empty_storage_buffer("output", expected.values.len());
    let kernel = bind_linear(gpu, weights, &format!("layers.0.{weight}.weight"), rows, &input_buffer, &output).unwrap_or_else(|error| panic!("{error}"));
    gpu.queue.submit([encode_runs(gpu, kernel.dispatches(), 1)]);
    let values = gpu.read_f32(&output).await.unwrap_or_else(|error| panic!("{error}"));
    let largest = values.iter().zip(&expected.values).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let biggest = expected.values.iter().map(|value| value.abs()).fold(0.0, f32::max);
    println!("{weight}, {rows} rows: largest difference from PyTorch {largest:e}, largest value {biggest}");
    largest
}

#[test]
fn grouped_4_bit_linear_matches_pytorch() {
    pollster::block_on(async {
        let gpu = Gpu::new().await.unwrap_or_else(|error| panic!("{error}"));
        let path = model_folder().join("engine/testdata/grouped_linear/weights.bin");
        let weights = load_to_gpu(&gpu, &path).await.unwrap_or_else(|error| panic!("{error}. Run uv run engine/testdata/make_grouped_linear.py first."));
        for weight in WEIGHTS {
            for rows in ROW_COUNTS {
                let largest = run_case(&gpu, &weights, weight, rows).await;
                // `<=` is false for NaN, so a NaN also fails.
                assert!(largest <= TOLERANCE, "{weight}, {rows} rows: largest difference {largest:e} is above {TOLERANCE:e}");
            }
        }
    });
}
