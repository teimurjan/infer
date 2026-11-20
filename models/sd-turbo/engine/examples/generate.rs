// fs writes the image file.
use std::fs;
// Path and PathBuf build the file paths.
use std::path::{Path, PathBuf};
// ExitCode sets the exit code: 0 for success, 1 for failure.
use std::process::ExitCode;

// The golden reader from our compare crate.
use compare::safetensors::read_tensors;
// The engine parts that this program uses.
use engine::{FromReader, GenerateInputs, Gpu, ProfileMode, StageWeights, StreamedFile, TextWeights, UnetWeights, WeightFile, best_mode, generate, image_to_rgba, start_profile};

// The image size.
const SIDE: usize = 512;

// Generate the golden prompt's image in three stages and print the planned and measured peak of each stage.
// Start it with `cargo run --release --example generate` in engine/, for the float32 UNet. It writes engine/target/generate.ppm.
// With `-- --packed`, the text encoder and UNet stages stream their files into GPU buffers as the web page does,
// the UNet file is the packed 4-bit and 8-bit one, and it writes engine/target/generate_packed.ppm.
// With `-- --q8`, it streams the files that the site loads, text_encoder_q8 and unet_q8, and writes engine/target/generate_q8.ppm.
// With `--profile`, it first measures the door number with the copy benchmark, then times every kernel and prints the table.
// `--wall-clock` with `--profile` times each dispatch with the CPU clock, as on a device without timestamps.
fn main() -> ExitCode {
    // block_on waits for the async function on this thread.
    match pollster::block_on(run()) {
        // The image is written.
        Ok(()) => ExitCode::SUCCESS,
        // Something failed. Print why.
        Err(error) => {
            // Errors go to stderr.
            eprintln!("generate failed: {error}");
            // Exit code 1 tells scripts that it failed.
            ExitCode::FAILURE
        }
    }
}

// The project folder, one level above engine/.
fn project_folder() -> PathBuf {
    // CARGO_MANIFEST_DIR is engine/, so the example works from any directory.
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("engine/ has a parent folder").to_path_buf()
}

// Read the one tensor of golden/<file>.
fn golden(file: &str) -> Result<Vec<f32>, String> {
    // For example golden/float32/sigma.safetensors.
    let path = project_folder().join("golden").join(file);
    // The first tensor of the file. Every golden file holds one.
    read_tensors(&path).map_err(|error| error.to_string())?.into_iter().next().map(|tensor| tensor.values).ok_or_else(|| format!("{} is empty", path.display()))
}

// Open one weight file of weights/.
fn weights(name: &str) -> Result<WeightFile, String> {
    // For example weights/unet.bin.
    WeightFile::open(&project_folder().join("weights").join(format!("{name}.bin"))).map_err(|error| error.to_string())
}

// The largest difference between the engine and a golden tensor.
fn largest_difference(engine: &[f32], golden: &[f32]) -> f32 {
    // Both lists have the same length when the shapes agree.
    engine.iter().zip(golden).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)
}

// True if the command line has `flag`.
fn has_flag(flag: &str) -> bool {
    std::env::args().any(|argument| argument == flag)
}

// Pick the UNet weights from the command line, then generate.
async fn run() -> Result<(), String> {
    // The streamed files of the packed runs.
    let folder = project_folder().join("weights");
    if has_flag("--q8") {
        // The files of the site. Their golden set holds the embeddings of the 8-bit text encoder too.
        generate_with(StreamedFile(&folder.join("text_encoder_q8.bin")), StreamedFile(&folder.join("unet_q8.bin")), "text_encoder_q8_unet_q8", "text_encoder_q8_unet_q8", "generate_q8.ppm").await
    } else if has_flag("--packed") {
        // The files stream into the GPU when their stage starts. Compare with the packed golden set.
        generate_with(StreamedFile(&folder.join("text_encoder.bin")), StreamedFile(&folder.join("unet_packed.bin")), "float32", "packed", "generate_packed.ppm").await
    } else {
        // Each kernel gets its own GPU copy of its float32 weights. Compare with the float32 golden set.
        generate_with(FromReader(&weights("text_encoder")?), FromReader(&weights("unet")?), "float32", "float32", "generate.ppm").await
    }
}

// Read the inputs, generate with the weights `text` and `unet`, compare the text embeddings with the golden set `text_set`
// and the other outputs with the golden set `set`, and write the image to target/<image>.
async fn generate_with(text: impl TextWeights, unet: impl UnetWeights, text_set: &str, set: &str, image: &str) -> Result<(), String> {
    // The 77 token ids, saved as float32 whole numbers.
    let ids: Vec<u32> = golden("float32/text_input_ids.safetensors")?.iter().map(|&id| id as u32).collect();
    // The starting noise, the sigma and the timestep of the golden run.
    let noise = golden("starting_noise.safetensors")?;
    let sigma = golden("float32/sigma.safetensors")?[0];
    let timestep = golden("float32/timestep.safetensors")?[0];
    // The weight file of the decoder.
    let decoder = weights("taesd")?;
    // Open the GPU.
    let gpu = Gpu::new().await.map_err(|error| error.to_string())?;
    // With --profile: measure the door, then time every kernel of the run. --wall-clock forces the CPU clock.
    if has_flag("--profile") {
        let mode = if has_flag("--wall-clock") { ProfileMode::WallClock } else { best_mode(&gpu) };
        println!("{}", start_profile(&gpu, mode).await.map_err(|error| error.to_string())?);
    }
    // Generate, and print each stage event as it happens.
    let inputs = GenerateInputs { token_ids: &ids, noise: &noise, sigma, timestep };
    let result = generate(&gpu, StageWeights { text, unet, decoder: &decoder }, &inputs, |event| println!("{event}")).await.map_err(|error| error.to_string())?;
    // Compare each output with the PyTorch run with the same weights. The text encoder output has its own set.
    for (name, values) in [("text_encoder", &result.text), ("unet_output", &result.noise_prediction), ("final_latent", &result.latent), ("taesd_image", &result.image)] {
        let tensor_set = if name == "text_encoder" { text_set } else { set };
        println!("{name}: largest difference from PyTorch ({tensor_set}) {:e}", largest_difference(values, &golden(&format!("{tensor_set}/{name}.safetensors"))?));
    }
    // The table, if a profile ran. take_profile also ends it.
    if let Some(profile) = gpu.take_profile() {
        println!("{}", profile.table());
    }
    // The image as RGB bytes: RGBA without the alpha byte.
    let rgb: Vec<u8> = image_to_rgba(&result.image, SIDE, SIDE).map_err(|error| error.to_string())?.chunks(4).flat_map(|pixel| pixel[..3].to_vec()).collect();
    // A binary PPM file: a short text header, then the bytes. Preview on the Mac opens it.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join(image);
    fs::write(&path, [format!("P6\n{SIDE} {SIDE}\n255\n").into_bytes(), rgb].concat()).map_err(|error| format!("{}: {error}", path.display()))?;
    println!("wrote {}", path.display());
    // Done.
    Ok(())
}
