// The shared golden checks.
mod common;

// The golden reader and the check of one block.
use common::{check_block, golden, project_folder};
// The engine parts that this test uses.
use engine::{Gpu, MemoryCategory, UnetInputs, UnetStage, WeightFile};

// The parts to check, in data-flow order, with the largest error we accept for each one. The stage must show exactly these.
// conv_in sums 36 products (4 channels x 9), so f32 rounding stays near 1e-6. 1e-4 still catches a wrong index.
// time_embedding sums 320 products, then 1280. The first run measured 9.5e-7. 1e-4 still catches one wrong frequency.
// down_blocks.0.resnets.0 has two GroupNorms and two convolutions of 2880 products. The first run measured 1.05e-5.
// The parts of the first transformer: the first run measured 1.5e-6 to 9.8e-6.
// resnets.1, attentions.1 and down_block_0: the first run measured 1.19e-5, 8.8e-6 and 1.10e-5.
// The later levels: 2 times the difference between PyTorch in float32 and PyTorch in float64, rounded up.
// That difference is the rounding noise of float32 at each part: 1.46e-4 at down_block_1, 4.85e-4 at up_block_1, 1.13e-5 at unet_output.
const TOLERANCES: [(&str, f64); 20] = [
    ("time_embedding", 1e-4),
    ("conv_in", 1e-4),
    ("down_blocks.0.resnets.0", 1e-4),
    ("down_blocks.0.attentions.0.proj_in", 1e-4),
    ("down_blocks.0.attentions.0.transformer_blocks.0.attn1", 1e-4),
    ("down_blocks.0.attentions.0.transformer_blocks.0.attn2", 1e-4),
    ("down_blocks.0.attentions.0.transformer_blocks.0.ff", 1e-4),
    ("down_blocks.0.attentions.0", 1e-4),
    ("down_blocks.0.resnets.1", 1e-4),
    ("down_blocks.0.attentions.1", 1e-4),
    ("down_block_0", 1e-4),
    ("down_block_1", 3e-4),
    ("down_block_2", 3e-4),
    ("down_block_3", 3e-4),
    ("mid_block", 3e-4),
    ("up_block_0", 3e-4),
    ("up_block_1", 1e-3),
    ("up_block_2", 1e-3),
    ("up_block_3", 7e-4),
    ("unet_output", 3e-5),
];
// The golden name of the UNet output, which every run returns.
const OUTPUT: &str = "unet_output";
// Bytes in one MB, as in PROGRESS.md.
const MB: f64 = 1e6;
// Run 1 and run 3 check every part. Run 2 checks none, as a real generation does.
const CHECKED_RUNS: [bool; 3] = [true, false, true];

// Every UNet part must match the PyTorch output. The run stops at the first part that does not.
// The activations must use only the planned slot buffers, and a run must make no bind group.
#[test]
fn unet_matches_golden_block_by_block() {
    // Open the GPU.
    let gpu = pollster::block_on(Gpu::new()).unwrap_or_else(|error| panic!("{error}"));
    // The flat weights from convert_weights.py.
    let weights = WeightFile::open(&project_folder().join("weights").join("unet.bin")).unwrap_or_else(|error| panic!("{error}. Run `uv run convert_weights.py unet` first."));
    // The scaled latent that the PyTorch UNet read.
    let sample = golden("unet_input").unwrap_or_else(|error| panic!("{error}"));
    // The timestep that the PyTorch UNet read, one value.
    let timestep = golden("timestep").unwrap_or_else(|error| panic!("{error}")).values[0];
    // The text encoder output that the PyTorch UNet read.
    let text = golden("text_encoder").unwrap_or_else(|error| panic!("{error}"));
    // The three inputs.
    let inputs = UnetInputs { sample: &sample.values, timestep, text: &text.values };
    // The PyTorch output, to compare with what every run returns.
    let expected = golden(OUTPUT).unwrap_or_else(|error| panic!("{error}"));
    // The tolerance of the output.
    let output_tolerance = TOLERANCES.iter().find(|(name, _)| *name == OUTPUT).map(|(_, tolerance)| *tolerance).unwrap_or_else(|| panic!("{OUTPUT} has no tolerance"));
    // The names of the parts to check.
    let names: Vec<&str> = TOLERANCES.iter().map(|(name, _)| *name).collect();
    // Make the slot buffers, read the weights and bind the kernels.
    let stage = UnetStage::prepare(&gpu, &weights).unwrap_or_else(|error| panic!("{error}"));
    // The planned peak, before the run.
    println!("planned activations: {} bytes ({:.1} MB)", stage.planned_bytes(), stage.planned_bytes() as f64 / MB);
    // The bind groups and buffers that prepare made.
    let prepared_bind_groups = gpu.bind_groups_made();
    println!("after prepare: {prepared_bind_groups} bind groups, {} buffers", gpu.buffers_made());
    // Run the stage more than once.
    for (run, checked) in (1..).zip(CHECKED_RUNS) {
        // The parts that this run checks.
        let wanted: &[&str] = if checked { &names } else { &[] };
        // The buffers made before this run.
        let buffers_before = gpu.buffers_made();
        // The parts that this run showed, in order.
        let mut seen = Vec::new();
        // Run the steps and check each wanted part. Print the first failure.
        let output = pollster::block_on(stage.run(&gpu, &inputs, wanted, |block| {
            // Remember the part.
            seen.push(block.name.to_string());
            // Compare with the golden tensor.
            check_block(block, &TOLERANCES)
        }))
        .unwrap_or_else(|error| panic!("{error}"));
        // The buffers that this run made.
        let buffers_made = gpu.buffers_made() - buffers_before;
        // The largest difference of the returned output from PyTorch. Run 2 checks no part, so this is its only check.
        let output_error = output.iter().zip(&expected.values).map(|(engine, golden)| f64::from((engine - golden).abs())).fold(0.0, f64::max);
        // The GPU buffers of the engine after this run.
        let memory = gpu.memory_report();
        // The measured numbers, after the run.
        println!(
            "run {run} ({} checks): returned output error {output_error:e}, {} values; measured peak {} bytes ({:.1} MB); live now: activations {}, weights {}, uniforms {}, read-back {}; bind groups {}, buffers made in the run {buffers_made}",
            seen.len(),
            output.len(),
            memory.peak(),
            memory.peak() as f64 / MB,
            memory.bytes(MemoryCategory::Activations),
            memory.bytes(MemoryCategory::Weights),
            memory.bytes(MemoryCategory::Uniforms),
            memory.bytes(MemoryCategory::ReadBack),
            gpu.bind_groups_made()
        );
        // The run showed exactly the wanted parts, in data-flow order. A part in the list that the stage never shows fails here.
        assert_eq!(seen, wanted, "run {run}: the parts that the stage showed");
        // A run reuses the bind groups of prepare.
        assert_eq!(gpu.bind_groups_made(), prepared_bind_groups, "run {run} made a bind group");
        // The slots are the only activation buffers that stay.
        assert_eq!(memory.bytes(MemoryCategory::Activations), stage.planned_bytes(), "run {run}: activations outside the plan");
        // A run makes only the read-back copies of its checks and of its output. Any other buffer, even one it frees again, is outside the plan.
        assert_eq!(buffers_made, seen.len() as u64 + 1, "run {run}: buffers outside the plan");
        // The returned output is the PyTorch output. `<=` is false for NaN, so a NaN also fails.
        assert!(output.len() == expected.values.len() && output_error <= output_tolerance, "run {run}: returned output error {output_error:e}");
    }
}
