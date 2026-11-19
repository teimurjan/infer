// The shared golden checks.
mod common;

// The golden readers and the check of one block.
use common::{check_block_in, golden, golden_in, project_folder};
// The engine parts that this test uses.
use engine::{Gpu, MemoryCategory, UnetInputs, UnetStage, load_to_gpu};

// The golden set of the packed UNet, from dump_packed_golden.py.
const SET: &str = "packed";
// The parts to check, in data-flow order, with the largest error we accept for each one. The stage must show exactly these.
// time_embedding, conv_in and down_block_0: 1e-4, as in tests/unet.rs. The first run measured 9.5e-7, 9.5e-7 and 2.4e-5.
// The deeper parts: 2 times the difference between this packed model in float32 and in float64 (dump_packed_golden.py), rounded up.
const TOLERANCES: [(&str, f64); 12] = [
    ("time_embedding", 1e-4),
    ("conv_in", 1e-4),
    ("down_block_0", 1e-4),
    ("down_block_1", 3e-4),
    ("down_block_2", 3e-4),
    ("down_block_3", 3e-4),
    ("mid_block", 3e-4),
    ("up_block_0", 3e-4),
    ("up_block_1", 1e-3),
    ("up_block_2", 1e-3),
    ("up_block_3", 5e-4),
    ("unet_output", 2e-5),
];
// The golden name of the UNet output, which every run returns.
const OUTPUT: &str = "unet_output";
// Bytes in one MB, as in PROGRESS.md.
const MB: f64 = 1e6;
// Run 1 and run 3 check every part. Run 2 checks none, as a real generation does.
const CHECKED_RUNS: [bool; 3] = [true, false, true];

// The UNet with the packed 4-bit and 8-bit weights must match PyTorch with the same weights.
// It must use only its planned slots and the loaded weight buffers, and a run must make no bind group.
#[test]
fn packed_unet_matches_golden() {
    // Open the GPU.
    let gpu = pollster::block_on(Gpu::new()).unwrap_or_else(|error| panic!("{error}"));
    // Stream the packed file into GPU buffers, as the browser does.
    let weights = pollster::block_on(load_to_gpu(&gpu, &project_folder().join("weights").join("unet_packed.bin"))).unwrap_or_else(|error| panic!("{error}"));
    // The loaded weights.
    let loaded = gpu.memory_report().bytes(MemoryCategory::Weights);
    println!("loaded weights: {weights}");
    // The inputs of the float32 golden run. The packed golden run used the same ones.
    let sample = golden("unet_input").unwrap_or_else(|error| panic!("{error}"));
    let timestep = golden("timestep").unwrap_or_else(|error| panic!("{error}")).values[0];
    let text = golden("text_encoder").unwrap_or_else(|error| panic!("{error}"));
    let inputs = UnetInputs { sample: &sample.values, timestep, text: &text.values };
    // The PyTorch output with the same weights, and its tolerance.
    let expected = golden_in(SET, OUTPUT).unwrap_or_else(|error| panic!("{error}"));
    let output_tolerance = TOLERANCES.iter().find(|(name, _)| *name == OUTPUT).map(|(_, tolerance)| *tolerance).unwrap_or_else(|| panic!("{OUTPUT} has no tolerance"));
    // The names of the parts to check.
    let names: Vec<&str> = TOLERANCES.iter().map(|(name, _)| *name).collect();
    // Make the slots and bind the kernels to the loaded buffers. The stage keeps the weights.
    let stage = UnetStage::prepare_loaded(&gpu, weights).unwrap_or_else(|error| panic!("{error}"));
    // The planned peak, before the run.
    println!("planned activations: {} bytes ({:.1} MB)", stage.planned_bytes(), stage.planned_bytes() as f64 / MB);
    // The bind groups and buffers that prepare made, and the weights that it added: the f32 copies of the float16 tensors and the zero biases.
    let prepared_bind_groups = gpu.bind_groups_made();
    let prepared_weights = gpu.memory_report().bytes(MemoryCategory::Weights);
    println!("after prepare: {prepared_bind_groups} bind groups, {} buffers, weights {prepared_weights} bytes ({} added by prepare)", gpu.buffers_made(), prepared_weights - loaded);
    // Run the stage more than once.
    for (run, checked) in (1..).zip(CHECKED_RUNS) {
        // The parts that this run checks.
        let wanted: &[&str] = if checked { &names } else { &[] };
        // The buffers made before this run.
        let buffers_before = gpu.buffers_made();
        // The parts that this run showed, in order.
        let mut seen = Vec::new();
        // Run the steps and check each wanted part against the packed set. Print the first failure.
        let output = pollster::block_on(stage.run(&gpu, &inputs, wanted, |block| {
            // Remember the part.
            seen.push(block.name.to_string());
            // Compare with the golden tensor.
            check_block_in(SET, block, &TOLERANCES)
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
            "run {run} ({} checks): returned output error {output_error:e}; measured peak {} bytes ({:.1} MB); live now: activations {}, weights {}, uniforms {}, read-back {}; bind groups {}, buffers made in the run {buffers_made}",
            seen.len(),
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
    // Dropping the stage frees the slots and the loaded weights.
    drop(stage);
    println!("after drop: live {} bytes", gpu.memory_report().total());
    assert_eq!(gpu.memory_report().total(), 0, "bytes live after the stage was dropped");
}
