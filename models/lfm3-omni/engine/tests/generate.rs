// The word-by-word loop against the official PyTorch model: a prompt of 353 tokens, longer than the 256-token window of
// the sliding layers, and a greedy answer of 24 tokens. Every step's scores and every token must match golden/loop.
// It needs `uv run python/dump_loop_golden.py`, and weights/layers.bin and weights/embed.bin.

// The shared helpers of the tests.
mod common;

use compare::safetensors::read_tensors;
use common::model_folder;
use engine_core::gpu::Gpu;
use engine_core::weights::{WeightFile, load_to_gpu};
use lfm3_engine::Generator;

// The largest score difference we accept at any step. The prompt pass of the 32-token golden differed by 5.1e-5.
// A longer prompt and 23 more passes add rounding, so the tolerance is the same 1e-3 as the language model test.
const SCORE_TOLERANCE: f32 = 1e-3;
// The memory holds the prompt and the answer: 353 + 24 = 377 positions, rounded up.
const CAPACITY: u32 = 512;

// The values of golden/loop/<name>.safetensors, and their shape.
fn loop_golden(name: &str) -> (Vec<f32>, Vec<usize>) {
    let path = model_folder().join("golden/loop").join(format!("{name}.safetensors"));
    let tensor = read_tensors(&path).unwrap_or_else(|error| panic!("{}: {error}. Run python/dump_loop_golden.py first.", path.display())).remove(0);
    (tensor.values, tensor.shape)
}

#[test]
fn answer_matches_pytorch_word_by_word() {
    pollster::block_on(async {
        let gpu = Gpu::new().await.unwrap_or_else(|error| panic!("{error}"));
        let (prompt, _) = loop_golden("prompt_ids");
        let prompt: Vec<u32> = prompt.iter().map(|&id| id as u32).collect();
        let (expected, _) = loop_golden("answer_ids");
        let expected: Vec<u32> = expected.iter().map(|&id| id as u32).collect();
        let (step_scores, shape) = loop_golden("step_scores");
        let vocabulary = shape[1];
        // The weights, and the generator with its memory.
        let table = WeightFile::open(&model_folder().join("weights/embed.bin")).unwrap_or_else(|error| panic!("{error}"));
        let layers = load_to_gpu(&gpu, &model_folder().join("weights/layers.bin")).await.unwrap_or_else(|error| panic!("{error}"));
        let generator = Generator::new(&gpu, &layers, &table, CAPACITY).unwrap_or_else(|error| panic!("{error}"));
        // The largest score difference of each step, against the PyTorch scores of the same step.
        let mut differences = Vec::new();
        let answer = generator.generate(&gpu, &prompt, expected.len(), |step, scores| {
            let golden = &step_scores[step * vocabulary..(step + 1) * vocabulary];
            differences.push(scores.iter().zip(golden).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max));
        }).await.unwrap_or_else(|error| panic!("{error}"));
        println!("prompt tokens: {}, answer tokens: {}", prompt.len(), answer.len());
        println!("largest score difference per step: {:?}", differences.iter().map(|value| format!("{value:.1e}")).collect::<Vec<_>>());
        let largest = differences.iter().copied().fold(0.0, f32::max);
        println!("largest score difference of all steps: {largest:e}");
        println!("same answer as PyTorch: {}", answer == expected);
        assert_eq!(answer, expected, "the answer must be the PyTorch answer");
        // `<=` is false for NaN, so a NaN also fails.
        assert!(largest <= SCORE_TOLERANCE, "largest score difference {largest:e} is above {SCORE_TOLERANCE:e}");
    });
}
