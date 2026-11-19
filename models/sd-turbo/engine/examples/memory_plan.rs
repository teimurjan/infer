// fs reads the weight index files.
use std::fs;
// Path builds the file paths.
use std::path::Path;
// ExitCode sets the exit code: 0 for success, 1 for failure.
use std::process::ExitCode;

// The engine parts that this program uses.
use engine::{EngineError, plan_unet_activations, weight_bytes_on_gpu};

// iOS kills the tab at about 1920 MB (crash-record/2026-11-15-iphone16promax.md).
const TAB_LIMIT_BYTES: u64 = 1_920_000_000;
// Bytes in one MB, as in PROGRESS.md.
const MB: f64 = 1e6;

// Print the planned GPU peak of each stage: text encoder, UNet, decoder. Each stage frees its buffers before the next one loads.
// Start it with `cargo run --release --example memory_plan` in engine/.
fn main() -> ExitCode {
    // Print the plan, or say why it failed.
    match run() {
        // The plan printed.
        Ok(()) => ExitCode::SUCCESS,
        // Something failed. Print why.
        Err(error) => {
            // Errors go to stderr.
            eprintln!("memory plan failed: {error}");
            // Exit code 1 tells scripts that it failed.
            ExitCode::FAILURE
        }
    }
}

// Read the three weight indexes, plan the UNet activations, and print one line per stage.
fn run() -> Result<(), EngineError> {
    // The weights folder is in the project folder, one level above engine/.
    let folder = Path::new(env!("CARGO_MANIFEST_DIR")).join("../weights");
    // The GPU bytes of the weights of each stage.
    let text_encoder = stage_weights(&folder, "text_encoder")?;
    let unet = stage_weights(&folder, "unet_packed")?;
    let taesd = stage_weights(&folder, "taesd")?;
    // The UNet activation plan.
    let plan = plan_unet_activations();
    // The forward pass.
    println!("unet forward pass: {} ops, {} tensors", plan.ops, plan.tensors);
    // The lower bound.
    println!("unet activations, most live at one op: {} bytes ({:.1} MB) at {}", plan.live_peak_bytes, plan.live_peak_bytes as f64 / MB, plan.live_peak_op);
    // The plan.
    println!("unet activations, planned: {} buffers, {} bytes ({:.1} MB), largest buffer {} bytes", plan.buffers, plan.planned_bytes, plan.planned_bytes as f64 / MB, plan.largest_buffer_bytes);
    // The stages. The text encoder plan needs a GPU stage; `cargo run --release --example generate` prints it.
    println!("stage 1 text_encoder: weights in the file {text_encoder} bytes ({:.1} MB)", text_encoder as f64 / MB);
    // The UNet stage: weights and planned activations.
    let unet_total = unet + plan.planned_bytes;
    println!("stage 2 unet: weights {unet} + activations {} = {unet_total} bytes ({:.1} MB, {:.1}% of the {:.0} MB tab limit)", plan.planned_bytes, unet_total as f64 / MB, 100.0 * unet_total as f64 / TAB_LIMIT_BYTES as f64, TAB_LIMIT_BYTES as f64 / MB);
    // The decoder stage.
    println!("stage 3 taesd: weights {taesd} bytes ({:.1} MB), activations not planned", taesd as f64 / MB);
    // Done.
    Ok(())
}

// The GPU bytes of the weights in `<name>.json`.
fn stage_weights(folder: &Path, name: &str) -> Result<u64, EngineError> {
    // Read the index. A missing file means the converter did not run.
    let json = fs::read_to_string(folder.join(format!("{name}.json"))).map_err(|error| EngineError::WeightIndex(format!("{name}.json: {error}")))?;
    // Add up the tensor bytes.
    weight_bytes_on_gpu(&json)
}
