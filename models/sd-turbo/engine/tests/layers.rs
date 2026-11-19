// fs reads the test files.
use std::fs;
// Path builds the file paths.
use std::path::Path;

// The engine parts that this test uses.
use engine::{CASES, Case, Gpu, Tensors, check_case, f32_from_le_bytes};

// The largest error we accept. Each output is a sum of up to 2880 products (320 channels x 9).
// f32 rounding in such a sum stays near 1e-6 for values near 1, so 1e-4 leaves room and still catches a wrong index.
const TOLERANCE: f32 = 1e-4;

// Read one raw float32 file that make_layers.py wrote for a case.
fn read_case_file(case: &Case, file: &str) -> Vec<f32> {
    // testdata/<case>/<file>, found from the crate folder, so the test works from any directory.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata").join(case.name).join(file);
    // A missing file means the data script did not run yet. Say so.
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("cannot read {}: {error}. Run testdata/make_layers.py first.", path.display()));
    // 4 bytes per value.
    f32_from_le_bytes(&bytes)
}

// Every layer kernel must match PyTorch at its real SD-Turbo shape.
// One test runs the cases one after another, because cases that run at the same time would slow each other down.
#[test]
fn layers_match_pytorch() {
    // Open the GPU once for all cases.
    let gpu = pollster::block_on(Gpu::new()).unwrap_or_else(|error| panic!("{error}"));
    // Run each case, print its line, and keep the names of the cases that are too wrong.
    let failures: Vec<String> = CASES
        .iter()
        .filter_map(|case| {
            // The four files of this case.
            let input = read_case_file(case, "input.bin");
            let weight = read_case_file(case, "weight.bin");
            let bias = read_case_file(case, "bias.bin");
            let expected = read_case_file(case, "expected.bin");
            // The three layer tensors.
            let tensors = Tensors { input: &input, weight: &weight, bias: &bias };
            // Compare with PyTorch and time it. block_on waits for the async function on this thread.
            let report = pollster::block_on(check_case(&gpu, case, &tensors, &expected)).unwrap_or_else(|error| panic!("{}: {error}", case.name));
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
