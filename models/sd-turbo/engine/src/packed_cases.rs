// The float cases give the layer shapes, and report_layer compares and times a layer.
use crate::cases::{CASES, Case, CaseReport, report_layer};
// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;
// The packed weight types and the function that prepares a packed layer.
use crate::layers::{PackedTensors, WeightBits, prepare_packed_layer};

// One float case, run with packed weights. testdata/make_packed_layers.py writes its files to testdata/<name>/.
pub struct PackedCase {
    // The folder name, for example "linear_q4".
    pub name: &'static str,
    // The float case with the same layer and shape.
    pub base: &'static Case,
    // 4 or 8 bits per weight.
    pub bits: WeightBits,
}

// Each of the four float cases, at 4 bits and at 8 bits. The names must match testdata/make_packed_layers.py.
pub const PACKED_CASES: [PackedCase; 8] = [
    PackedCase { name: "linear_q4", base: &CASES[0], bits: WeightBits::Four },
    PackedCase { name: "linear_q8", base: &CASES[0], bits: WeightBits::Eight },
    PackedCase { name: "conv1x1_q4", base: &CASES[1], bits: WeightBits::Four },
    PackedCase { name: "conv1x1_q8", base: &CASES[1], bits: WeightBits::Eight },
    PackedCase { name: "conv3x3_q4", base: &CASES[2], bits: WeightBits::Four },
    PackedCase { name: "conv3x3_q8", base: &CASES[2], bits: WeightBits::Eight },
    PackedCase { name: "conv3x3_stride2_q4", base: &CASES[3], bits: WeightBits::Four },
    PackedCase { name: "conv3x3_stride2_q8", base: &CASES[3], bits: WeightBits::Eight },
];

// Run one packed case on the GPU: compare with PyTorch once, then time it.
pub async fn check_packed_case(gpu: &Gpu, case: &PackedCase, tensors: &PackedTensors<'_>, expected: &[f32]) -> Result<CaseReport, EngineError> {
    // Check the sizes, copy the data to the GPU and compile the packed kernel.
    let layer = prepare_packed_layer(gpu, &case.base.shape, case.bits, tensors)?;
    // Compare and time it.
    report_layer(gpu, case.name, &layer, expected).await
}
