// Display lets us turn each error into a readable message.
use std::fmt;

// The project limit for one submit, for the messages.
use crate::submit::SUBMIT_LIMIT_SECONDS;

// Every way the engine can fail.
#[derive(Debug)]
pub enum EngineError {
    // The system or the browser has no GPU adapter (a handle to one physical GPU) that we can use.
    NoAdapter(wgpu::RequestAdapterError),
    // The adapter did not give us a device (our private connection to the GPU).
    NoDevice(wgpu::RequestDeviceError),
    // The kernel got zero values. The GPU cannot read back an empty buffer.
    EmptyInput,
    // The input needs more workgroups than the GPU can start in one dispatch.
    TooManyValues { count: usize, max: usize },
    // Two lists that must have the same length do not.
    LengthMismatch { actual: usize, expected: usize },
    // A tensor has the wrong number of values for the layer shape.
    TensorLength { tensor: &'static str, actual: usize, expected: usize },
    // A layer dimension is 0 or not a multiple of 4. The kernels work in blocks of 4.
    BadDimension { dimension: &'static str, value: u32 },
    // A packed matrix has a number of scales per row that this kernel cannot read.
    ScaleGroups { groups: u32, rule: &'static str },
    // A layer dimension is 0.
    ZeroDimension { dimension: &'static str },
    // A convolution stride of 0.
    ZeroStride,
    // Attention keeps one query row of one head in registers, so the head size is fixed.
    HeadSize { value: u32, supported: u32 },
    // A dispatch needs more workgroups along one axis than the GPU allows.
    TooManyWorkgroups { workgroups: u32, max: u32 },
    // No test case has this name.
    UnknownCase(String),
    // Waiting for the GPU to finish failed.
    Poll(wgpu::PollError),
    // The GPU could not give us read access to the result buffer.
    Map(wgpu::BufferAsyncError),
    // A GPU callback never ran, because wgpu dropped it.
    CallbackDropped,
    // One submit took too long. iOS loses the device near 0.5 s, so we stop at 0.25 s.
    SubmitTooLong { benchmark: &'static str, phase: SubmitPhase, runs: u32, seconds: f64 },
    // The result buffer was mapped, but we could not get its bytes.
    MapRange(wgpu::MapRangeError),
    // One submit of a model run took too long. `label` names the block.
    SubmitOverLimit { label: String, seconds: f64 },
    // A profile asked for GPU timestamps, but this device has no "timestamp-query" feature.
    NoTimestamps,
    // A step has more dispatches than one query set can time.
    TooManyTimestamps { label: String, dispatches: usize, max: usize },
    // The GPU reported an error inside a checked part, such as a profiled submit. `label` names the part.
    GpuError { label: String, message: String },
    // The weight index is not the JSON list that convert_weights.py writes.
    WeightIndex(String),
    // The tokenizer files (tokenizer_vocab.json, tokenizer_merges.txt) are not what export_tokenizer.py writes.
    TokenizerFiles(String),
    // The activation plan has no tensor with this name.
    UnplannedTensor(String),
    // A kernel would write a tensor of another size than the activation plan has for it.
    PlanMismatch { tensor: String, planned: [u32; 3], actual: [u32; 3] },
    // The weight index has no tensor with this name.
    MissingWeight(String),
    // The kernels read float32 only.
    WeightDtype { name: String, dtype: String },
    // A weight has a shape that does not fit the layer that uses it.
    WeightShape { name: String, shape: Vec<usize>, expected: String },
    // The weight file or its index could not be read from disk.
    WeightRead(String),
    // The index has a dtype that convert_weights.py does not write.
    UnknownDtype { name: String, dtype: String },
    // The index puts a tensor where write_buffer cannot fill it, or two tensors share bytes.
    WeightLayout(String),
    // One tensor needs a GPU buffer over the limit. `max` is the smaller of the iPhone limit and the device limit.
    TensorTooLarge { name: String, bytes: u64, max: u64 },
    // The weight stream ended before the last tensor. A download broke, or a cached file is cut short.
    StreamEnded { received: u64, needed: u64 },
}

// Turn each error into one line of plain text.
impl fmt::Display for EngineError {
    // Write the message for this error.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // Pick the message that matches the kind of error.
        match self {
            // Most often: the browser has no WebGPU, or the page is not on https or localhost.
            Self::NoAdapter(source) => write!(formatter, "no GPU adapter: {source}"),
            // Show the reason from wgpu.
            Self::NoDevice(source) => write!(formatter, "no GPU device: {source}"),
            // Say what was empty.
            Self::EmptyInput => write!(formatter, "the input has no values"),
            // Show the count and the limit.
            Self::TooManyValues { count, max } => write!(formatter, "{count} values, but one dispatch can do at most {max}"),
            // Show both lengths.
            Self::LengthMismatch { actual, expected } => write!(formatter, "got {actual} values, expected {expected}"),
            // Say which tensor, and both lengths.
            Self::TensorLength { tensor, actual, expected } => write!(formatter, "{tensor} has {actual} values, the shape needs {expected}"),
            // Say which dimension and its value.
            Self::BadDimension { dimension, value } => write!(formatter, "{dimension} is {value}, it must be a positive multiple of 4"),
            // Say the count and the rule.
            Self::ScaleGroups { groups, rule } => write!(formatter, "a packed matrix has {groups} scales per row, {rule}"),
            // Say which dimension.
            Self::ZeroDimension { dimension } => write!(formatter, "{dimension} is 0, it must be 1 or more"),
            // Say what is wrong.
            Self::ZeroStride => write!(formatter, "the stride must be 1 or more"),
            // Show the size we got and the one we support.
            Self::HeadSize { value, supported } => write!(formatter, "head size is {value}, the attention kernel supports only {supported}"),
            // Show both sizes and the fix.
            // Show the count and the limit.
            Self::TooManyWorkgroups { workgroups, max } => write!(formatter, "{workgroups} workgroups along one axis, the limit is {max}"),
            // Show the name.
            Self::UnknownCase(name) => write!(formatter, "no test case is named {name:?}"),
            // Show the reason from wgpu.
            Self::Poll(source) => write!(formatter, "waiting for the GPU failed: {source}"),
            // Show the reason from wgpu.
            Self::Map(source) => write!(formatter, "cannot map the result buffer: {source}"),
            // Say that the callback did not run.
            Self::CallbackDropped => write!(formatter, "a GPU callback was dropped before it ran"),
            // Say which benchmark, which submit, how big it was, and how long it took.
            Self::SubmitTooLong { benchmark, phase, runs, seconds } => write!(
                formatter,
                "{benchmark}: {phase} submit with {runs} runs took {:.1} ms, the limit is {:.1} ms",
                seconds * 1000.0,
                SUBMIT_LIMIT_SECONDS * 1000.0
            ),
            // Show the reason from wgpu.
            Self::MapRange(source) => write!(formatter, "cannot read the mapped buffer: {source}"),
            // Say which block and how long its submit took.
            Self::SubmitOverLimit { label, seconds } => write!(formatter, "{label}: one submit took {:.1} ms, the limit is {:.1} ms", seconds * 1000.0, SUBMIT_LIMIT_SECONDS * 1000.0),
            // Say that the device lacks the feature, and what to use.
            Self::NoTimestamps => write!(formatter, "this device has no timestamp-query feature, use the wall clock profile"),
            // Say which step and the counts.
            Self::TooManyTimestamps { label, dispatches, max } => write!(formatter, "{label}: {dispatches} dispatches, one query set can time at most {max}"),
            // Say where and what the GPU said.
            Self::GpuError { label, message } => write!(formatter, "{label}: the GPU reported an error: {message}"),
            // Say which tensor is missing from the plan.
            Self::UnplannedTensor(name) => write!(formatter, "the activation plan has no tensor {name}"),
            // Show both sizes.
            Self::PlanMismatch { tensor, planned, actual } => write!(formatter, "{tensor}: the plan has size {planned:?}, the kernel writes {actual:?}"),
            // Show the reason from serde_json.
            Self::WeightIndex(reason) => write!(formatter, "the weight index is not valid: {reason}"),
            // Say what is wrong in which file.
            Self::TokenizerFiles(reason) => write!(formatter, "the tokenizer files are not valid: {reason}"),
            // Show the name.
            Self::MissingWeight(name) => write!(formatter, "the weight index has no tensor named {name:?}"),
            // Show the name and its dtype.
            Self::WeightDtype { name, dtype } => write!(formatter, "{name} is {dtype}, the kernels read only float32"),
            // Show the name, its shape and the shape the layer needs.
            Self::WeightShape { name, shape, expected } => write!(formatter, "{name} has shape {shape:?}, the layer needs {expected}"),
            // Show the path or name and the reason from the operating system.
            Self::WeightRead(reason) => write!(formatter, "cannot read weights: {reason}"),
            // Show the name and its dtype.
            Self::UnknownDtype { name, dtype } => write!(formatter, "{name} is {dtype}, the loader knows float32, float16, int8 and int4"),
            // Show the reason.
            Self::WeightLayout(reason) => write!(formatter, "the weight index layout is not valid: {reason}"),
            // Show the name, the size and the limit.
            Self::TensorTooLarge { name, bytes, max } => write!(formatter, "{name} needs a GPU buffer of {bytes} bytes, the limit is {max} bytes"),
            // Show how far the stream came, and how far it had to come.
            Self::StreamEnded { received, needed } => write!(formatter, "the weight stream ended after {received} bytes, the index needs {needed} bytes"),
        }
    }
}

// Which submit of a benchmark took too long.
#[derive(Debug, Clone, Copy)]
pub enum SubmitPhase {
    // The first submit, whose time we throw away.
    WarmUp,
    // A submit that finds how many runs fit.
    Calibration,
    // A timed submit. The number counts from 1.
    Timed(usize),
}

// Turn the phase into words for the error message.
impl fmt::Display for SubmitPhase {
    // Write the words for this phase.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        // Pick the words that match the phase.
        match self {
            // The skipped first submit.
            Self::WarmUp => write!(formatter, "warm-up"),
            // A calibration submit.
            Self::Calibration => write!(formatter, "calibration"),
            // A timed submit and its number.
            Self::Timed(number) => write!(formatter, "timed submit {number}"),
        }
    }
}

// Mark EngineError as a normal Rust error. wasm-bindgen uses this to turn it into a JavaScript Error.
impl std::error::Error for EngineError {}
