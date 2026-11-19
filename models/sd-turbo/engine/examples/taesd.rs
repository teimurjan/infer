// fs reads the weight files into memory.
use std::fs;
// Path and PathBuf build the file paths.
use std::path::{Path, PathBuf};
// ExitCode sets the exit code: 0 for success, 1 for failure.
use std::process::ExitCode;

// The golden reader from our compare crate.
use compare::safetensors::read_tensors;
// The engine parts that this program uses.
use engine::{Gpu, WeightBytes, decode_and_report};

// Decode the golden latent with TAESD and print the same lines as engine/web/taesd.html.
// It reads the weights into memory with WeightBytes, as the page does. Start it with `cargo run --release --example taesd`.
fn main() -> ExitCode {
    // block_on waits for the async function on this thread.
    match pollster::block_on(run()) {
        // The decoder finished.
        Ok(()) => ExitCode::SUCCESS,
        // Something failed. Print why.
        Err(error) => {
            // Errors go to stderr.
            eprintln!("taesd failed: {error}");
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

// Read the one tensor of golden/float32/<name>.safetensors.
fn golden(name: &str) -> Result<Vec<f32>, String> {
    // For example golden/float32/final_latent.safetensors.
    let path = project_folder().join("golden").join("float32").join(format!("{name}.safetensors"));
    // Read every tensor of the file.
    let tensors = read_tensors(&path).map_err(|error| error.to_string())?;
    // Take the tensor with this name.
    tensors.into_iter().find(|tensor| tensor.name == name).map(|tensor| tensor.values).ok_or_else(|| format!("{} has no tensor {name}", path.display()))
}

// Read the files, decode, and print one line each.
async fn run() -> Result<(), String> {
    // The weights folder.
    let folder = project_folder().join("weights");
    // The index text.
    let index = fs::read_to_string(folder.join("taesd.json")).map_err(|error| format!("weights/taesd.json: {error}"))?;
    // Every byte of the weight file, as the page fetches it.
    let bytes = fs::read(folder.join("taesd.bin")).map_err(|error| format!("weights/taesd.bin: {error}"))?;
    // The reader over those bytes.
    let weights = WeightBytes::new(&index, bytes).map_err(|error| error.to_string())?;
    // The SD-Turbo latent and the diffusers image.
    let (latent, expected) = (golden("final_latent")?, golden("taesd_image")?);
    // Open the GPU.
    let gpu = Gpu::new().await.map_err(|error| error.to_string())?;
    // Decode and report.
    let report = decode_and_report(&gpu, &weights, &latent, &expected).await.map_err(|error| error.to_string())?;
    // Print each line.
    report.lines.iter().for_each(|line| println!("{line}"));
    // Done.
    Ok(())
}
