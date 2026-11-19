// env reads the command line arguments.
use std::env;
// Path holds a file path.
use std::path::Path;
// ExitCode sets the exit code: 0 for success, 1 for failure.
use std::process::ExitCode;

// The value comparison from the library part of this crate.
use compare::diff;
// Our error type.
use compare::error::CompareError;
// Our tensor type and file reader.
use compare::safetensors::{Tensor, read_tensors};

// Run the tool. Print any error and exit with code 1.
fn main() -> ExitCode {
    // Skip argument 0, which is the program name.
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        // Every tensor is within the threshold.
        Ok(()) => ExitCode::SUCCESS,
        // Something failed. Print why.
        Err(error) => {
            // Errors go to stderr, so they do not mix with the table.
            eprintln!("error: {error}");
            // Exit code 1.
            ExitCode::FAILURE
        }
    }
}

// Read both files, compare each tensor, and fail if any difference is above the threshold.
fn run(args: &[String]) -> Result<(), CompareError> {
    // We need exactly three arguments: file A, file B, threshold.
    let [a_path, b_path, threshold_text] = args else {
        // Any other count: show the usage line.
        return Err(CompareError::Usage);
    };
    // Turn the threshold text into a number.
    let threshold = parse_threshold(threshold_text)?;
    // Read every tensor in file A.
    let a_tensors = read_tensors(Path::new(a_path))?;
    // Read every tensor in file B.
    let b_tensors = read_tensors(Path::new(b_path))?;
    // Every tensor in B must also be in A, or we would skip it without a word.
    for b_tensor in &b_tensors {
        // Fail if this name is not in A.
        find(&a_tensors, &b_tensor.name, a_path)?;
    }
    // Compare each tensor in A with the tensor of the same name in B. true means "above the threshold".
    let above = a_tensors
        // Go through the tensors of A.
        .iter()
        // Compare one pair and print its line.
        .map(|a_tensor| compare_pair(a_tensor, find(&b_tensors, &a_tensor.name, b_path)?, threshold))
        // Stop at the first error, or collect all results.
        .collect::<Result<Vec<bool>, CompareError>>()?;
    // Count the tensors above the threshold.
    let count = above.iter().filter(|&&is_above| is_above).count();
    // Any tensor above the threshold makes the whole run fail.
    if count > 0 {
        // Exit with code 1 and say how many failed.
        return Err(CompareError::AboveThreshold { count, threshold });
    }
    // Say that everything passed.
    println!("all tensors within {threshold}");
    // Exit with code 0.
    Ok(())
}

// Turn the threshold text into a number of zero or more.
fn parse_threshold(text: &str) -> Result<f64, CompareError> {
    // Parse the text, and accept only finite numbers of zero or more.
    text.parse::<f64>()
        // Drop the parse error; we make our own message.
        .ok()
        // Reject NaN, infinity and negative numbers.
        .filter(|value| value.is_finite() && *value >= 0.0)
        // Anything else is a bad threshold.
        .ok_or_else(|| CompareError::BadThreshold(text.to_owned()))
}

// Find the tensor with this name, or fail with the name of the file that lacks it.
fn find<'a>(tensors: &'a [Tensor], name: &str, path: &str) -> Result<&'a Tensor, CompareError> {
    // Look for the first tensor with this name.
    tensors
        // Go through the tensors.
        .iter()
        // Pick the one with a matching name.
        .find(|tensor| tensor.name == name)
        // No match means the file does not have it.
        .ok_or_else(|| CompareError::MissingTensor { name: name.to_owned(), path: path.into() })
}

// Compare one pair of tensors, print one line, and return true if it is above the threshold.
fn compare_pair(a: &Tensor, b: &Tensor, threshold: f64) -> Result<bool, CompareError> {
    // The two tensors must have the same shape to compare them value by value.
    if a.shape != b.shape {
        // Different shapes cannot be compared.
        return Err(CompareError::ShapeMismatch { name: a.name.clone(), a: a.shape.clone(), b: b.shape.clone() });
    }
    // Compute the largest difference, the average difference and the worst position.
    let stats = diff::compare(&a.values, &b.values, &a.shape);
    // A NaN means something is broken, so it also counts as above.
    let is_above = stats.max.is_nan() || stats.max > threshold;
    // A short word for the end of the line.
    let verdict = if is_above { "ABOVE" } else { "ok" };
    // One line per tensor: name, largest difference, average difference, worst position and the two values there.
    println!(
        // Fixed-width columns, so the lines line up.
        "{:<16} max={:<11.3e} mean={:<11.3e} worst={:?} a={} b={} {verdict}",
        // The tensor name.
        a.name,
        // Largest difference.
        stats.max,
        // Average difference.
        stats.mean,
        // Worst position.
        stats.worst_index,
        // Value in A at that position.
        stats.worst_a,
        // Value in B at that position.
        stats.worst_b,
    );
    // Tell the caller whether this tensor failed.
    Ok(is_above)
}
