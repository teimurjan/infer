// The shared golden checks.
mod common;

// The golden reader and the check of one block.
use common::{check_block, golden, project_folder};
// The engine parts that this test uses.
use engine::{Gpu, TAESD_PARTS, WeightFile, run_taesd};

// The largest error we accept for each part, in data-flow order. A part without an entry fails.
// Each convolution sums 576 products (64 channels x 9), and the image is 20 convolutions deep, so f32 rounding grows along the way.
// 1e-3 leaves room for that. A wrong index gives errors of the size of the values themselves, about 0.1 or more.
const TOLERANCES: [(&str, f64); 9] = [
    ("taesd_conv_in", 1e-3),
    ("taesd_blocks_0", 1e-3),
    ("taesd_upsample_0", 1e-3),
    ("taesd_blocks_1", 1e-3),
    ("taesd_upsample_1", 1e-3),
    ("taesd_blocks_2", 1e-3),
    ("taesd_upsample_2", 1e-3),
    ("taesd_blocks_3", 1e-3),
    ("taesd_image", 1e-3),
];

// Every TAESD part must match the diffusers AutoencoderTiny output. The run stops at the first part that does not.
#[test]
fn taesd_matches_golden_part_by_part() {
    // Open the GPU.
    let gpu = pollster::block_on(Gpu::new()).unwrap_or_else(|error| panic!("{error}"));
    // The flat weights from convert_weights.py.
    let weights = WeightFile::open(&project_folder().join("weights").join("taesd.bin")).unwrap_or_else(|error| panic!("{error}. Run `uv run convert_weights.py taesd` first."));
    // The SD-Turbo latent that diffusers gave to the decoder.
    let latent = golden("final_latent").unwrap_or_else(|error| panic!("{error}"));
    // Run the parts and check each one.
    let result = pollster::block_on(run_taesd(&gpu, &weights, &latent.values, &TAESD_PARTS, |block| check_block(block, &TOLERANCES)));
    // Print the first failure.
    result.unwrap_or_else(|error| panic!("{error}"));
}
