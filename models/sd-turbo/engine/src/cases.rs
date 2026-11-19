// Display lets us print a report as one line.
use std::fmt;

// The shared timing loop and its report.
use crate::bench::{BenchReport, Kernel, measure};
// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;
// The layer kinds and the code that prepares them.
use crate::layers::{Conv1x1Shape, Conv3x3Shape, LayerShape, LinearShape, PreparedLayer, Tensors, prepare_layer};
// The function that compares two lists.
use crate::max_error::max_abs_error;

// One real SD-Turbo layer that we test. testdata/make_layers.py writes its files to testdata/<name>/.
pub struct Case {
    // The folder name, for example "linear".
    pub name: &'static str,
    // The layer in the SD-Turbo UNet that has this shape.
    pub layer: &'static str,
    // The layer kind and sizes. They must match testdata/make_layers.py.
    pub shape: LayerShape,
}

// The four layers we test. The shapes come from the SD-Turbo UNet at a 512 x 512 image (64 x 64 latent).
pub const CASES: [Case; 4] = [
    // Attention input projection: 64 x 64 = 4096 tokens of 320 values.
    Case {
        name: "linear",
        layer: "down_blocks.0.attentions.0.proj_in",
        shape: LayerShape::Linear(LinearShape { rows: 4096, in_features: 320, out_features: 320 }),
    },
    // ResNet shortcut where the channels grow from 320 to 640, at 32 x 32.
    Case {
        name: "conv1x1",
        layer: "down_blocks.1.resnets.0.conv_shortcut",
        shape: LayerShape::Conv1x1(Conv1x1Shape { in_channels: 320, out_channels: 640, height: 32, width: 32 }),
    },
    // First ResNet convolution of the first block, at 64 x 64.
    Case {
        name: "conv3x3",
        layer: "down_blocks.0.resnets.0.conv1",
        shape: LayerShape::Conv3x3(Conv3x3Shape { in_channels: 320, out_channels: 320, height: 64, width: 64, stride: 1 }),
    },
    // Downsampler: 64 x 64 in, 32 x 32 out.
    Case {
        name: "conv3x3_stride2",
        layer: "down_blocks.0.downsamplers.0.conv",
        shape: LayerShape::Conv3x3(Conv3x3Shape { in_channels: 320, out_channels: 320, height: 64, width: 64, stride: 2 }),
    },
];

// Find a case by its name.
pub fn find_case(name: &str) -> Result<&'static Case, EngineError> {
    // Look through the list.
    CASES.iter().find(|case| case.name == name).ok_or_else(|| EngineError::UnknownCase(name.to_string()))
}

// The result of one case: how wrong and how fast.
pub struct CaseReport {
    // The largest difference from the PyTorch output.
    max_error: f32,
    // The speed in GB/s, counting every tensor read or written once.
    speed: BenchReport,
}

impl CaseReport {
    // The largest difference from the PyTorch output.
    pub fn max_error(&self) -> f32 {
        self.max_error
    }
}

// One line, for example "linear: 40.12 GB/s median, ... , largest error 1.9e-6".
impl fmt::Display for CaseReport {
    // Write the speed line, then the error.
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}, largest error {:e}", self.speed, self.max_error)
    }
}

// Run one case on the GPU: compare with PyTorch once, then time it.
pub async fn check_case(gpu: &Gpu, case: &Case, tensors: &Tensors<'_>, expected: &[f32]) -> Result<CaseReport, EngineError> {
    // Check the sizes, copy the data to the GPU and compile the kernel.
    let layer = prepare_layer(gpu, &case.shape, tensors)?;
    // Compare and time it.
    report_layer(gpu, case.name, &layer, expected).await
}

// Run a prepared layer once and compare it with the PyTorch output, then time it. Float and packed cases share this.
pub(crate) async fn report_layer(gpu: &Gpu, name: &'static str, layer: &PreparedLayer, expected: &[f32]) -> Result<CaseReport, EngineError> {
    // Run once and copy the output back.
    let output = layer.run(gpu).await?;
    // Compare with the PyTorch output.
    let max_error = max_abs_error(&output, expected)?;
    // Time it: one skipped submit, calibration, then the timed submits.
    let timing = measure(gpu, &Kernel { name, dispatches: std::slice::from_ref(layer.dispatch()) }).await?;
    // Bytes per run divided by the time, in GB/s.
    let speed = BenchReport::new(name, "GB/s", layer.bytes_moved() as f64, &timing);
    // Both results.
    Ok(CaseReport { max_error, speed })
}
