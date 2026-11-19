// fs reads the test files.
use std::fs;
// Path builds the file paths.
use std::path::Path;

// The engine parts that this test uses.
use engine::{f32_from_le_bytes, silu_max_error};

// The largest error we accept. One f32 step (ULP) at 80, our largest input, is about 7.6e-6.
// WGSL allows exp() and division to be a few steps off, so 1e-4 leaves room for that and nothing more.
const TOLERANCE: f32 = 1e-4;

// Read one raw float32 file that make_data.py wrote.
fn read_f32_file(name: &str) -> Vec<f32> {
    // The test data folder, found from the crate folder, so the test works from any directory.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/silu").join(name);
    // A missing file means the data script did not run yet. Say so.
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("cannot read {}: {error}. Run make_data.py first.", path.display()));
    // 4 bytes per value.
    f32_from_le_bytes(&bytes)
}

// The GPU SiLU must match PyTorch's SiLU on the same input.
#[test]
fn silu_matches_pytorch() {
    // The input that PyTorch used.
    let input = read_f32_file("input.bin");
    // What PyTorch computed.
    let expected = read_f32_file("expected.bin");
    // Run the kernel on the GPU and measure. block_on waits for the async function on this thread.
    let max_error = pollster::block_on(silu_max_error(&input, &expected)).unwrap_or_else(|error| panic!("{error}"));
    // Print the result. Run with `-- --nocapture` to see it.
    println!("SiLU: {} values, largest error {max_error:e}", input.len());
    // Fail if the error is too large. A NaN also fails, because NaN <= x is false.
    assert!(max_error <= TOLERANCE, "largest error {max_error:e} is above {TOLERANCE:e}");
}
