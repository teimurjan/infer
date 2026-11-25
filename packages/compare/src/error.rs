// Display lets us turn each error into a readable message.
use std::fmt;
// PathBuf stores the file path that caused an error.
use std::path::PathBuf;

// Every way the tool can fail. main prints the message and exits with code 1.
pub enum CompareError {
    // The user did not pass exactly three arguments.
    Usage,
    // The threshold argument is not a number of zero or more.
    BadThreshold(String),
    // The file could not be read from disk.
    Io { path: PathBuf, source: std::io::Error },
    // The file is not valid safetensors. `reason` says what is wrong.
    Header { path: PathBuf, reason: String },
    // A tensor uses a number type this tool cannot read.
    UnsupportedDtype { path: PathBuf, name: String, dtype: String },
    // A tensor name is in one file but not in the other.
    MissingTensor { name: String, path: PathBuf },
    // The same tensor has a different shape in the two files.
    ShapeMismatch { name: String, a: Vec<usize>, b: Vec<usize> },
    // At least one tensor has a difference above the threshold.
    AboveThreshold { count: usize, threshold: f64 },
}

// Turn each error into one line of plain text.
impl fmt::Display for CompareError {
    // Write the message for this error.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // Pick the message that matches the kind of error.
        match self {
            // Show how to call the tool.
            Self::Usage => write!(formatter, "usage: compare <a.safetensors> <b.safetensors> <threshold>"),
            // Show the bad threshold text.
            Self::BadThreshold(text) => write!(formatter, "threshold must be a number of 0 or more, got {text:?}"),
            // Show the path and the reason from the operating system.
            Self::Io { path, source } => write!(formatter, "cannot read {}: {source}", path.display()),
            // Show the path and what is wrong with its header.
            Self::Header { path, reason } => write!(formatter, "bad safetensors file {}: {reason}", path.display()),
            // Show which tensor uses which unknown type.
            Self::UnsupportedDtype { path, name, dtype } => {
                // Only F32 and F16 are supported.
                write!(formatter, "{}: tensor {name} has dtype {dtype}, only F32 and F16 are supported", path.display())
            }
            // Show which name is missing from which file.
            Self::MissingTensor { name, path } => write!(formatter, "tensor {name} is not in {}", path.display()),
            // Show both shapes.
            Self::ShapeMismatch { name, a, b } => write!(formatter, "tensor {name} has shape {a:?} in A but {b:?} in B"),
            // Show how many tensors failed.
            Self::AboveThreshold { count, threshold } => {
                // One line that a script can also read.
                write!(formatter, "{count} tensor(s) have a difference above {threshold}")
            }
        }
    }
}
