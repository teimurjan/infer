// Path builds the file paths.
use std::path::Path;

// The golden reader from our compare crate.
use compare::safetensors::read_tensors;
// The engine's noise.
use engine::{NOISE_VALUES, starting_noise};

// The largest difference allowed. The engine and PyTorch can call different log, sin and cos functions, which may differ in the last bit.
const TOLERANCE: f32 = 1e-5;

// The engine must give the noise of torch.randn for every seed of testdata/noise/noise.safetensors.
// A different noise is a different image for the same seed.
#[test]
fn noise_matches_torch_randn() {
    // The file that make_noise.py writes.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/noise/noise.safetensors");
    let tensors = read_tensors(&path).unwrap_or_else(|error| panic!("{error}. Run `uv run testdata/noise/make_noise.py` first."));
    // The seeds whose noise differs too much, with the difference.
    let mut failures = Vec::new();
    for tensor in &tensors {
        // The seed is in the name, for example seed_42.
        let seed: u32 = tensor.name.trim_start_matches("seed_").parse().expect("the name is seed_<number>");
        let noise = starting_noise(seed);
        assert_eq!(noise.len(), NOISE_VALUES);
        // The largest difference of this seed.
        let largest = noise.iter().zip(&tensor.values).map(|(engine, torch)| (engine - torch).abs()).fold(0.0, f32::max);
        println!("seed {seed}: largest difference from torch.randn {largest:e}");
        if largest > TOLERANCE {
            failures.push(format!("seed {seed}: {largest:e}"));
        }
    }
    assert!(failures.is_empty(), "largest difference above {TOLERANCE:e}: {failures:?}");
}
