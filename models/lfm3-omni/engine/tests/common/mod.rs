// Helpers that the layer tests share: the model folder, the golden tensors, the weights, and one layer run.
// Each test file uses only some of them, so the unused ones are not a warning.
#![allow(dead_code)]
// It needs `uv run python/dump_text_golden.py` and the weight files of `python/export_weights.py` in the model folder.

// The paths of the model folder.
use std::path::{Path, PathBuf};

// The golden file reader of packages/compare, and the GPU base.
use compare::safetensors::read_tensors;
use engine_core::dispatch::{Dispatch, encode_runs};
use engine_core::gpu::Gpu;
use engine_core::memory::{GpuBuffer, MemoryCategory};
use engine_core::weights::{GpuWeights, load_to_gpu};
use lfm3_engine::Memory;

// The model folder, one level above engine/.
pub fn model_folder() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("engine/ has a parent folder").to_path_buf()
}

// The values and the shape of golden/text/<name>.safetensors.
pub fn golden(name: &str) -> (Vec<f32>, Vec<usize>) {
    let path = model_folder().join("golden/text").join(format!("{name}.safetensors"));
    let tensor = read_tensors(&path).unwrap_or_else(|error| panic!("{}: {error}. Run python/dump_text_golden.py first.", path.display())).remove(0);
    (tensor.values, tensor.shape)
}

// The weights of layer `layer`, from weights/layer<layer>.bin, streamed into GPU buffers.
pub async fn layer_weights(gpu: &Gpu, layer: usize) -> GpuWeights {
    let path = model_folder().join(format!("weights/layer{layer}.bin"));
    load_to_gpu(gpu, &path).await.unwrap_or_else(|error| panic!("{error}. Run python/export_weights.py layer{layer} layers.{layer}. first."))
}

// The input of layer `layer` on the GPU, the number of its rows, and the PyTorch output of the layer.
pub fn layer_golden(gpu: &Gpu, layer: usize) -> (GpuBuffer, u32, Vec<f32>) {
    // hidden_<i> is the input of layer i, and hidden_<i + 1> its output.
    let (input, shape) = golden(&format!("hidden_{layer:02}"));
    let (expected, _) = golden(&format!("hidden_{:02}", layer + 1));
    (gpu.storage_buffer_from("hidden", &input, MemoryCategory::Activations), shape[0] as u32, expected)
}

// A memory of earlier tokens for a prompt of `rows` tokens, with the step set to positions 0 to rows - 1.
pub fn prompt_memory(gpu: &Gpu, rows: u32) -> Memory {
    let memory = Memory::new(gpu, rows);
    memory.set_step(gpu, 0, rows).unwrap_or_else(|error| panic!("{error}"));
    memory
}

// Run `dispatches` in one submit, read `hidden` back, print the largest difference from `expected`, and return it.
pub async fn run_and_compare(gpu: &Gpu, label: &str, dispatches: &[Dispatch], hidden: &GpuBuffer, expected: &[f32]) -> f32 {
    gpu.queue.submit([encode_runs(gpu, dispatches, 1)]);
    let output = gpu.read_f32(hidden).await.unwrap_or_else(|error| panic!("{error}"));
    // The largest difference, and the largest value, for the log.
    let largest = output.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let biggest = expected.iter().map(|value| value.abs()).fold(0.0, f32::max);
    println!("{label}: largest difference from PyTorch {largest:e}, largest value {biggest}");
    largest
}
