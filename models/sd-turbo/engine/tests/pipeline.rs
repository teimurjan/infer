// The shared golden checks.
mod common;

// The golden readers and the project folder.
use common::{golden, golden_in, project_folder};
// The golden file reader, for the noise outside the golden sets.
use compare::safetensors::read_tensors;
// The engine parts that this test uses.
use engine::{FromReader, GenerateInputs, Gpu, Stage, StageEvent, StageWeights, StreamedFile, TextWeights, UnetWeights, WeightFile, generate, image_to_rgba};

// The largest error we accept for each output of the stages, against the PyTorch run with the same weights.
// text_encoder and unet_output: the tolerances of tests/text.rs and tests/unet.rs.
// final_latent: the Euler step multiplies the UNet error by sigma (14.61), so 3e-5 x 14.61 = 4.4e-4, rounded up.
// taesd_image: below 1/8 of one 8-bit step of the image (2 / 255 / 8 = 9.8e-4).
const TOLERANCES: [(&str, f32); 4] = [("text_encoder", 5e-5), ("unet_output", 3e-5), ("final_latent", 5e-4), ("taesd_image", 9.8e-4)];
// The image size.
const SIDE: usize = 512;

// Open one weight file of weights/.
fn weights(name: &str) -> WeightFile {
    // For example weights/unet.bin.
    WeightFile::open(&project_folder().join("weights").join(format!("{name}.bin"))).unwrap_or_else(|error| panic!("{name}: {error}"))
}

// The largest difference between two lists.
fn largest_difference(engine: &[f32], golden: &[f32]) -> f32 {
    // Lists of other lengths cannot be the same tensor.
    assert_eq!(engine.len(), golden.len());
    // The largest absolute difference.
    engine.iter().zip(golden).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

// Run the three stages with the weights `text` and `unet`, and compare the UNet output, the latent and the image with the golden set `set`.
// The text encoder output is always compared with the float32 set. The three stages must run one after the other,
// each must free all its buffers, and the measured peak of each planned stage must be its planned peak.
fn generate_and_check(text: impl TextWeights, unet: impl UnetWeights, set: &str) {
    // The 77 token ids, the noise, the sigma and the timestep of the golden run.
    let ids: Vec<u32> = golden("text_input_ids").unwrap_or_else(|error| panic!("{error}")).values.iter().map(|&id| id as u32).collect();
    let noise = read_tensors(&project_folder().join("golden").join("starting_noise.safetensors")).unwrap_or_else(|error| panic!("{error}")).remove(0).values;
    let sigma = golden("sigma").unwrap_or_else(|error| panic!("{error}")).values[0];
    let timestep = golden("timestep").unwrap_or_else(|error| panic!("{error}")).values[0];
    // The weight file of the decoder.
    let decoder = weights("taesd");
    // Open the GPU.
    let gpu = pollster::block_on(Gpu::new()).unwrap_or_else(|error| panic!("{error}"));
    // Every event, in order.
    let mut events = Vec::new();
    // Generate.
    let inputs = GenerateInputs { token_ids: &ids, noise: &noise, sigma, timestep };
    let result = pollster::block_on(generate(&gpu, StageWeights { text, unet, decoder: &decoder }, &inputs, |event| events.push(*event))).unwrap_or_else(|error| panic!("{error}"));
    // Print the events, for the log.
    events.iter().for_each(|event| println!("{event:?}"));
    // Three events per stage, in stage order.
    let stages: Vec<Stage> = events.iter().map(|event| match event { StageEvent::Prepared { stage, .. } | StageEvent::Ran { stage, .. } | StageEvent::Freed { stage, .. } => *stage }).collect();
    assert_eq!(stages, [Stage::TextEncoder; 3].into_iter().chain([Stage::Unet; 3]).chain([Stage::Decoder; 3]).collect::<Vec<_>>());
    // Each stage: the planned peak is the measured peak, and nothing is live after it.
    for stage_events in events.chunks(3) {
        // The three events of one stage.
        let [StageEvent::Prepared { stage, planned_peak, .. }, StageEvent::Ran { measured_peak, .. }, StageEvent::Freed { live, .. }] = stage_events else { panic!("events out of order: {stage_events:?}") };
        // A stage with a plan reaches exactly its planned peak.
        if let Some(planned) = planned_peak {
            assert_eq!(measured_peak, planned, "{stage}: measured peak");
        }
        // The next stage starts with no engine buffer from this one.
        assert_eq!(*live, 0, "{stage}: live bytes after the stage was freed");
    }
    // Every output against PyTorch. The text encoder has no packed weights, so it uses the float32 set.
    for ((name, tolerance), values) in TOLERANCES.into_iter().zip([&result.text, &result.noise_prediction, &result.latent, &result.image]) {
        // The golden set of this output.
        let tensor_set = if name == "text_encoder" { "float32" } else { set };
        // The difference from the golden tensor.
        let difference = largest_difference(values, &golden_in(tensor_set, name).unwrap_or_else(|error| panic!("{error}")).values);
        println!("{name}: largest difference from PyTorch ({tensor_set}) {difference:e}");
        // `<=` is false for NaN, so a NaN also fails.
        assert!(difference <= tolerance, "{name}: {difference:e} is above {tolerance:e}");
    }
    // The 8-bit pixels of the engine image and of the PyTorch image.
    let to_bytes = |image: &[f32]| image_to_rgba(image, SIDE, SIDE).unwrap_or_else(|error| panic!("{error}"));
    let (engine, pytorch) = (to_bytes(&result.image), to_bytes(&golden_in(set, "taesd_image").unwrap_or_else(|error| panic!("{error}")).values));
    // How many bytes differ, and by how much at most.
    let differing = engine.iter().zip(&pytorch).filter(|(a, b)| a != b).count();
    let largest = engine.iter().zip(&pytorch).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
    println!("image bytes: {differing} of {} differ, by at most {largest}", engine.len());
    // A difference below 1/8 of a step can still round to the next byte, but never by more than 1.
    assert!(largest <= 1, "an image byte differs by {largest}");
}

// The float32 UNet, with the weights of both planned stages read from CPU readers, against the float32 golden set.
#[test]
fn stages_with_float_unet_match_pytorch() {
    // Each kernel gets its own GPU copy of its float32 weights.
    let (text, unet) = (weights("text_encoder"), weights("unet"));
    generate_and_check(FromReader(&text), FromReader(&unet), "float32");
}

// The packed UNet, against the packed golden set. Both planned stages stream their file into GPU buffers when they start,
// with the loader of the browser, and bind to those buffers, as the web page does.
#[test]
fn stages_with_packed_unet_match_pytorch() {
    // The two files, streamed when their stage starts.
    let folder = project_folder().join("weights");
    let (text, unet) = (folder.join("text_encoder.bin"), folder.join("unet_packed.bin"));
    generate_and_check(StreamedFile(&text), StreamedFile(&unet), "packed");
}
