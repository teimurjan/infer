// Layer 0 of the LFM3 language model, a short convolution layer, against the official PyTorch model.
// It needs `uv run python/dump_text_golden.py` and `uv run python/export_weights.py layer0 layers.0.` in the model folder.

// The paths of the model folder.
use std::path::{Path, PathBuf};

// The golden file reader of packages/compare.
use compare::safetensors::read_tensors;
// The GPU base, and the layer under test.
use engine_core::dispatch::encode_runs;
use engine_core::gpu::Gpu;
use engine_core::memory::MemoryCategory;
use engine_core::weights::load_to_gpu;
use lfm3_engine::ConvLayer;

// The largest difference we accept. The output values of layer 0 reach about 0.9, and the first run differed from
// PyTorch by 2.3e-6, from float32 sums in another order. 1e-4 leaves room and still catches a wrong index.
const TOLERANCE: f32 = 1e-4;

// The model folder, one level above engine/.
fn model_folder() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("engine/ has a parent folder").to_path_buf()
}

// The values and the shape of golden/text/<name>.safetensors.
fn golden(name: &str) -> (Vec<f32>, Vec<usize>) {
    let path = model_folder().join("golden/text").join(format!("{name}.safetensors"));
    let tensor = read_tensors(&path).unwrap_or_else(|error| panic!("{}: {error}. Run python/dump_text_golden.py first.", path.display())).remove(0);
    (tensor.values, tensor.shape)
}

#[test]
fn layer_0_matches_pytorch() {
    pollster::block_on(async {
        let gpu = Gpu::new().await.unwrap_or_else(|error| panic!("{error}"));
        // The weights of layer 0, streamed into GPU buffers.
        let weights = load_to_gpu(&gpu, &model_folder().join("weights/layer0.bin")).await.unwrap_or_else(|error| panic!("{error}. Run python/export_weights.py layer0 layers.0. first."));
        // The input of layer 0 and its PyTorch output, [32 tokens][2048].
        let (input, shape) = golden("hidden_00");
        let (expected, _) = golden("hidden_01");
        let rows = shape[0] as u32;
        // The hidden rows on the GPU. The layer changes them in place.
        let hidden = gpu.storage_buffer_from("hidden", &input, MemoryCategory::Activations);
        let layer = ConvLayer::bind(&gpu, &weights, "layers.0.", rows, &hidden).unwrap_or_else(|error| panic!("{error}"));
        // One submit with the 11 kernels, then read the hidden rows back.
        gpu.queue.submit([encode_runs(&gpu, &layer.dispatches(), 1)]);
        let output = gpu.read_f32(&hidden).await.unwrap_or_else(|error| panic!("{error}"));
        // The largest difference, and the largest value, for the log.
        let largest = output.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let biggest = expected.iter().map(|value| value.abs()).fold(0.0, f32::max);
        println!("layer 0: largest difference from PyTorch {largest:e}, largest value {biggest}");
        // `<=` is false for NaN, so a NaN also fails.
        assert!(largest <= TOLERANCE, "largest difference {largest:e} is above {TOLERANCE:e}");
    });
}
