// Layers 2 (full attention) and 4 (sliding attention) of the LFM3 language model, against the official PyTorch model.

// The shared helpers of the layer tests.
mod common;

use common::{layer_golden, layer_weights, run_and_compare};
use engine_core::gpu::Gpu;
use lfm3_engine::{AttentionKind, AttentionLayer};

// The same tolerance as layer 0. The attention adds a softmax over at most 32 keys, which keeps float32 errors small.
const TOLERANCE: f32 = 1e-4;

// Run one attention layer on its golden input and return the largest difference from its golden output.
async fn check(layer: usize, kind: AttentionKind) -> f32 {
    let gpu = Gpu::new().await.unwrap_or_else(|error| panic!("{error}"));
    let weights = layer_weights(&gpu, layer).await;
    let (hidden, rows, expected) = layer_golden(&gpu, layer);
    let bound = AttentionLayer::bind(&gpu, &weights, &format!("layers.{layer}."), kind, rows, &hidden).unwrap_or_else(|error| panic!("{error}"));
    run_and_compare(&gpu, &format!("layer {layer}"), &bound.dispatches(), &hidden, &expected).await
}

#[test]
fn attention_layers_match_pytorch() {
    pollster::block_on(async {
        // One test for both, so the two GPU runs do not slow each other down.
        let full = check(2, AttentionKind::Full).await;
        let sliding = check(4, AttentionKind::Sliding).await;
        // `<=` is false for NaN, so a NaN also fails.
        assert!(full <= TOLERANCE && sliding <= TOLERANCE, "largest differences {full:e} and {sliding:e}, tolerance {TOLERANCE:e}");
    });
}
