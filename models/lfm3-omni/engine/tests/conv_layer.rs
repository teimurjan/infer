// Layer 0 of the LFM3 language model, a short convolution layer, against the official PyTorch model.

// The shared helpers of the layer tests.
mod common;

use common::{layer_golden, layer_weights, run_and_compare};
use engine_core::gpu::Gpu;
use lfm3_engine::ConvLayer;

// The largest difference we accept. The output values of layer 0 reach about 0.9, and the first run differed from
// PyTorch by 2.3e-6, from float32 sums in another order. 1e-4 leaves room and still catches a wrong index.
const TOLERANCE: f32 = 1e-4;

#[test]
fn layer_0_matches_pytorch() {
    pollster::block_on(async {
        let gpu = Gpu::new().await.unwrap_or_else(|error| panic!("{error}"));
        let weights = layer_weights(&gpu, 0).await;
        let (hidden, rows, expected) = layer_golden(&gpu, 0);
        let layer = ConvLayer::bind(&gpu, &weights, "layers.0.", rows, &hidden).unwrap_or_else(|error| panic!("{error}"));
        let largest = run_and_compare(&gpu, "layer 0", &layer.dispatches(), &hidden, &expected).await;
        // `<=` is false for NaN, so a NaN also fails.
        assert!(largest <= TOLERANCE, "largest difference {largest:e} is above {TOLERANCE:e}");
    });
}
