// The shared golden checks.
mod common;

// The golden reader and the check of one part.
use common::{check_block, golden, project_folder};
// The engine parts that this test uses.
use engine::{Gpu, MemoryCategory, TextStage, WeightFile};

// The parts to check, in data-flow order, with the largest error we accept for each one. The stage must show exactly these.
// text_embeddings: 0, because the CPU does the same one f32 add per value as PyTorch.
// The others: 2 times the difference between PyTorch in float32 and in float64, rounded up (6.83e-5 at layer 0, 2.13e-5 at the output).
const TOLERANCES: [(&str, f64); 3] = [("text_embeddings", 0.0), ("text_layer_0", 2e-4), ("text_encoder", 5e-5)];
// The golden name of the output, which every run returns.
const OUTPUT: &str = "text_encoder";
// Bytes in one MB, as in PROGRESS.md.
const MB: f64 = 1e6;
// Run 1 and run 3 check every part. Run 2 checks none, as a real generation does.
const CHECKED_RUNS: [bool; 3] = [true, false, true];

// The text encoder must match the PyTorch output. It must use only its planned slot buffers, and a run must make no bind group.
#[test]
fn text_encoder_matches_golden() {
    // Open the GPU.
    let gpu = pollster::block_on(Gpu::new()).unwrap_or_else(|error| panic!("{error}"));
    // The flat weights from convert_weights.py.
    let weights = WeightFile::open(&project_folder().join("weights").join("text_encoder.bin")).unwrap_or_else(|error| panic!("{error}. Run `uv run convert_weights.py text_encoder` first."));
    // The 77 token ids of the golden prompt, saved as float32 by dump_text_parts.py. Every id is a whole number below 2^24.
    let ids: Vec<u32> = golden("text_input_ids").unwrap_or_else(|error| panic!("{error}. Run `uv run dump_text_parts.py` first.")).values.iter().map(|&id| id as u32).collect();
    // The PyTorch output, to compare with what every run returns.
    let expected = golden(OUTPUT).unwrap_or_else(|error| panic!("{error}"));
    // The tolerance of the output.
    let output_tolerance = TOLERANCES.iter().find(|(name, _)| *name == OUTPUT).map(|(_, tolerance)| *tolerance).unwrap_or_else(|| panic!("{OUTPUT} has no tolerance"));
    // The names of the parts to check.
    let names: Vec<&str> = TOLERANCES.iter().map(|(name, _)| *name).collect();
    // Make the slot buffers, read the weights and bind the kernels.
    let stage = TextStage::prepare(&gpu, &weights).unwrap_or_else(|error| panic!("{error}"));
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
        let output = pollster::block_on(stage.run(&gpu, &ids, wanted, |block| {
            // Remember the part.
            seen.push(block.name.to_string());
            // Compare with the golden tensor.
            check_block(block, &TOLERANCES)
        }))
        .unwrap_or_else(|error| panic!("{error}"));
        // The buffers that this run made.
        let buffers_made = gpu.buffers_made() - buffers_before;
        // The largest difference of the returned output from PyTorch.
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
        // The run showed exactly the wanted parts, in data-flow order.
        assert_eq!(seen, wanted, "run {run}: the parts that the stage showed");
        // A run reuses the bind groups of prepare.
        assert_eq!(gpu.bind_groups_made(), prepared_bind_groups, "run {run} made a bind group");
        // The slots are the only activation buffers that stay.
        assert_eq!(memory.bytes(MemoryCategory::Activations), stage.planned_bytes(), "run {run}: activations outside the plan");
        // A run makes only the read-back copies of its checks and of its output.
        assert_eq!(buffers_made, seen.len() as u64 + 1, "run {run}: buffers outside the plan");
        // The returned output is the PyTorch output. `<=` is false for NaN, so a NaN also fails.
        assert!(output.len() == expected.values.len() && output_error <= output_tolerance, "run {run}: returned output error {output_error:e}");
    }
}
