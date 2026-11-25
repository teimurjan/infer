// The whole LFM3 language model over the golden prompt, against the official PyTorch model:
// the token rows, the final hidden rows after the final norm, the scores of the next token, and the best token.
// It needs weights/layers.bin and weights/embed.bin from python/export_weights.py (see PROGRESS.md).

// The shared helpers of the layer tests.
mod common;

use common::{golden, model_folder};
use engine_core::dispatch::encode_runs;
use engine_core::gpu::Gpu;
use engine_core::memory::MemoryCategory;
use engine_core::weights::{WeightFile, load_to_gpu};
use lfm3_engine::{LanguageModel, token_rows};

// The token of "Paris", the first answer token of the golden run.
const PARIS: usize = 41176;
// The largest differences we accept. 24 layers add float32 rounding on top of each other. The first run differed by
// 1.49e-4 in the last hidden state and by 5.1e-5 in the scores. The tolerances leave about 7 and 20 times that room.
const HIDDEN_TOLERANCE: f32 = 1e-3;
const SCORE_TOLERANCE: f32 = 1e-3;

// The largest difference between two lists.
fn largest_difference(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn language_model_matches_pytorch() {
    pollster::block_on(async {
        let gpu = Gpu::new().await.unwrap_or_else(|error| panic!("{error}"));
        // The 32 prompt ids of the golden run.
        let (ids, _) = golden("prompt_ids");
        let ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        // The token rows on the CPU must be the input of layer 0 exactly: the same float32 values, copied.
        let table = WeightFile::open(&model_folder().join("weights/embed.bin")).unwrap_or_else(|error| panic!("{error}. Run python/export_weights.py embed embed first."));
        let rows = token_rows(&table, &ids).unwrap_or_else(|error| panic!("{error}"));
        let (first, _) = golden("hidden_00");
        println!("token rows: largest difference from PyTorch {:e}", largest_difference(&rows, &first));
        // The 24 layers, streamed into GPU buffers.
        let layers = load_to_gpu(&gpu, &model_folder().join("weights/layers.bin")).await.unwrap_or_else(|error| panic!("{error}. Run python/export_weights.py layers layers. first."));
        let hidden = gpu.storage_buffer_from("hidden", &rows, MemoryCategory::Activations);
        let model = LanguageModel::bind(&gpu, &layers, &table, ids.len() as u32, &hidden).unwrap_or_else(|error| panic!("{error}"));
        println!("dispatches: {}", model.dispatches().len());
        // One submit for the whole prompt. The Mac has no 250 ms limit to keep in this test.
        gpu.queue.submit([encode_runs(&gpu, model.dispatches(), 1)]);
        // The final hidden rows, after the final norm.
        let last = gpu.read_f32(model.last_hidden()).await.unwrap_or_else(|error| panic!("{error}"));
        let (expected_last, _) = golden("last_hidden_state");
        let hidden_difference = largest_difference(&last, &expected_last);
        println!("last hidden state: largest difference from PyTorch {hidden_difference:e}");
        // The scores of the next token, from the last prompt row.
        let scores = model.scores_of_row(&gpu, ids.len() - 1).await.unwrap_or_else(|error| panic!("{error}"));
        let (expected_scores, _) = golden("next_logits");
        let score_difference = largest_difference(&scores, &expected_scores);
        let best = scores.iter().enumerate().fold((0, f32::MIN), |best, (index, &score)| if score > best.1 { (index, score) } else { best }).0;
        println!("next scores: largest difference from PyTorch {score_difference:e}, best token {best}");
        // `<=` is false for NaN, so a NaN also fails.
        assert!(hidden_difference <= HIDDEN_TOLERANCE, "last hidden state {hidden_difference:e} is above {HIDDEN_TOLERANCE:e}");
        assert!(score_difference <= SCORE_TOLERANCE, "scores {score_difference:e} are above {SCORE_TOLERANCE:e}");
        assert_eq!(best, PARIS, "the best next token must be Paris");
    });
}
