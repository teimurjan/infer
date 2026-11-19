// fs reads the test files.
use std::fs;
// Path builds the file paths.
use std::path::Path;

// The engine parts that this test uses.
use engine::{Gpu, PACKED_CASES, PackedCase, PackedTensors, check_packed_case, f32_from_le_bytes};

// The largest error we accept, the same as the float layer test. Each output sums up to 2880 products in f32.
// The integers are at most 127 and the scale is applied once at the end, so the f32 rounding stays as small as in the float kernels.
const TOLERANCE: f32 = 1e-4;

// Read one file that make_packed_layers.py wrote for a case, as raw bytes.
fn read_case_bytes(case: &PackedCase, file: &str) -> Vec<u8> {
    // testdata/<case>/<file>, found from the crate folder, so the test works from any directory.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata").join(case.name).join(file);
    // A missing file means the data script did not run yet. Say so.
    fs::read(&path).unwrap_or_else(|error| panic!("cannot read {}: {error}. Run `uv run testdata/make_packed_layers.py` first.", path.display()))
}

// Every packed layer kernel must match PyTorch running the same rounded weights, at its real SD-Turbo shape.
// One test runs the cases one after another, because cases that run at the same time would slow each other down.
#[test]
fn packed_layers_match_pytorch() {
    // Open the GPU once for all cases.
    let gpu = pollster::block_on(Gpu::new()).unwrap_or_else(|error| panic!("{error}"));
    // Run each case, print its line, and keep the names of the cases that are too wrong.
    let failures: Vec<String> = PACKED_CASES
        .iter()
        .filter_map(|case| {
            // The f32 files of this case.
            let input = f32_from_le_bytes(&read_case_bytes(case, "input.bin"));
            let bias = f32_from_le_bytes(&read_case_bytes(case, "bias.bin"));
            let expected = f32_from_le_bytes(&read_case_bytes(case, "expected.bin"));
            // The packed words and the float16 scales stay raw bytes. The GPU reads them as they are.
            let weight = read_case_bytes(case, "weight.bin");
            let scale = read_case_bytes(case, "scale.bin");
            // The four layer tensors.
            let tensors = PackedTensors { input: &input, weight: &weight, scale: &scale, bias: &bias };
            // Compare with PyTorch and time it. block_on waits for the async function on this thread.
            let report = pollster::block_on(check_packed_case(&gpu, case, &tensors, &expected)).unwrap_or_else(|error| panic!("{}: {error}", case.name));
            // Print the line. Run with `-- --nocapture` to see it.
            println!("{report}");
            // A NaN also fails, because NaN <= x is false.
            let within_tolerance = report.max_error() <= TOLERANCE;
            // Keep a message only for a case that failed.
            (!within_tolerance).then(|| format!("{}: {:e}", case.name, report.max_error()))
        })
        .collect();
    // All cases must be within the tolerance.
    assert!(failures.is_empty(), "largest error above {TOLERANCE:e}: {failures:?}");
}
