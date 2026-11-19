// Code that the golden tests share: read a golden tensor, compare one block with it, and print one line.
// Each test file includes it with `mod common;`. A test file that uses only some of it would warn about the rest, so the warning is off here.
#![allow(dead_code)]

// Display lets us print the test errors.
use std::fmt;
// Path and PathBuf build the file paths.
use std::path::{Path, PathBuf};

// The golden reader and the value comparison from our compare crate.
use compare::diff::compare as diff;
use compare::safetensors::{Tensor, read_tensors};
// The engine parts that the checks use.
use engine::{BlockOutput, EngineError};

// Every way this test can fail, so that the first mismatch stops the UNet with a clear message.
pub enum CheckError {
    // The engine failed: a GPU error, a weight error, or a submit over the 0.25 s limit.
    Engine(EngineError),
    // A golden file is missing or broken.
    Golden(String),
    // A block has no tolerance in the list.
    NoTolerance(String),
    // A block output does not have the golden shape.
    Shape { block: String, engine: Vec<usize>, golden: Vec<usize> },
    // A block output is too far from the golden output.
    Mismatch { block: String, max: f64, tolerance: f64 },
}

// `?` turns an engine error into a test error.
impl From<EngineError> for CheckError {
    // Wrap it.
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

// One line per error.
impl fmt::Display for CheckError {
    // Write the message for this error.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // Pick the message that matches the kind of error.
        match self {
            // Show the engine message.
            Self::Engine(error) => write!(formatter, "{error}"),
            // Show the reason.
            Self::Golden(reason) => write!(formatter, "golden: {reason}"),
            // Say which block.
            Self::NoTolerance(block) => write!(formatter, "{block}: no tolerance in the list"),
            // Show both shapes.
            Self::Shape { block, engine, golden } => write!(formatter, "{block}: engine shape {engine:?}, golden shape {golden:?}"),
            // Show the error and the limit.
            Self::Mismatch { block, max, tolerance } => write!(formatter, "{block}: first mismatch, largest error {max:e} is above {tolerance:e}"),
        }
    }
}

// The project folder, one level above engine/.
pub fn project_folder() -> PathBuf {
    // CARGO_MANIFEST_DIR is engine/, so the test works from any directory.
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("engine/ has a parent folder").to_path_buf()
}

// Read golden/float32/<name>.safetensors, which holds one tensor with the same name.
pub fn golden(name: &str) -> Result<Tensor, CheckError> {
    // The float32 set of dump_golden.py.
    golden_in("float32", name)
}

// Read golden/<set>/<name>.safetensors. `set` is "float32", or "packed" for the packed UNet of dump_packed_golden.py.
pub fn golden_in(set: &str, name: &str) -> Result<Tensor, CheckError> {
    // For example golden/packed/conv_in.safetensors.
    let path = project_folder().join("golden").join(set).join(format!("{name}.safetensors"));
    // Read every tensor of the file. A missing file means dump_golden.py did not run yet.
    let tensors = read_tensors(&path).map_err(|error| CheckError::Golden(format!("{error}. Run the dump script of this set first: dump_golden.py, dump_unet_parts.py, dump_text_parts.py or dump_packed_golden.py.")))?;
    // Take the tensor with this name.
    tensors.into_iter().find(|tensor| tensor.name == name).ok_or_else(|| CheckError::Golden(format!("{} has no tensor {name}", path.display())))
}

// The tolerance of one block, from a list of (name, tolerance) pairs.
fn tolerance(tolerances: &[(&str, f64)], block: &str) -> Result<f64, CheckError> {
    // Look it up by name.
    tolerances.iter().find(|(name, _)| *name == block).map(|(_, value)| *value).ok_or_else(|| CheckError::NoTolerance(block.to_string()))
}

// Compare one block with its golden tensor of the float32 set, print one line, and fail on a mismatch.
pub fn check_block(block: &BlockOutput, tolerances: &[(&str, f64)]) -> Result<(), CheckError> {
    // The float32 set.
    check_block_in("float32", block, tolerances)
}

// The same, with the golden tensor of `set`.
pub fn check_block_in(set: &str, block: &BlockOutput, tolerances: &[(&str, f64)]) -> Result<(), CheckError> {
    // The PyTorch output of the same block.
    let expected = golden_in(set, block.name)?;
    // The shapes must agree before the values can.
    if expected.shape != block.shape {
        return Err(CheckError::Shape { block: block.name.to_string(), engine: block.shape.clone(), golden: expected.shape });
    }
    // Largest error, average error, and the worst position.
    let stats = diff(block.values, &expected.values, &expected.shape);
    // One line per block. Run with `-- --nocapture` to see it.
    println!(
        "{}: largest error {:e}, mean error {:e}, worst at {:?} (engine {}, golden {}), {} submits, longest {:.1} ms, peak {} bytes",
        block.name,
        stats.max,
        stats.mean,
        stats.worst_index,
        stats.worst_a,
        stats.worst_b,
        block.submits,
        block.longest_submit_seconds * 1000.0,
        block.peak_bytes,
    );
    // The limit for this block.
    let tolerance = tolerance(tolerances, block.name)?;
    // `<=` is false for NaN, so a NaN also fails.
    if !(stats.max <= tolerance) {
        return Err(CheckError::Mismatch { block: block.name.to_string(), max: stats.max, tolerance });
    }
    // This block matches.
    Ok(())
}
