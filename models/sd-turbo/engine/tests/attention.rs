// fs reads the test files.
use std::fs;
// Path builds the file paths.
use std::path::Path;

// The engine parts that this test uses.
use engine::{AttentionCase, AttentionTensors, Gpu, check_attention, f32_from_le_bytes, find_attention_case};

// The largest error we accept. Each output is an average of value rows near -3..3, so f32 rounding stays near 1e-6.
// 1e-4 leaves room and still catches a wrong index or a wrong softmax.
const TOLERANCE: f32 = 1e-4;
// The chunk size for the main runs: 256 rows, as in the UNet.
const CHUNK_ROWS: u32 = 256;
// A chunk size that does not divide 4096 or 64, so the last chunk and the last workgroup are partly empty.
const UNEVEN_CHUNK_ROWS: u32 = 1000;

// Read one raw float32 file that make_attention.py wrote for a case.
fn read_case_file(case: &AttentionCase, file: &str) -> Vec<f32> {
    // testdata/<case>/<file>, found from the crate folder, so the test works from any directory.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata").join(case.name).join(file);
    // A missing file means the data script did not run yet. Say so.
    let bytes = fs::read(&path).unwrap_or_else(|error| panic!("cannot read {}: {error}. Run testdata/make_attention.py first.", path.display()));
    // 4 bytes per value.
    f32_from_le_bytes(&bytes)
}

// Self and cross attention must match PyTorch, and the peak memory must stay below the full scores table.
// One test runs the cases one after another, because cases that run at the same time would slow each other down.
#[test]
fn attention_matches_pytorch_without_the_full_table() {
    // Open the GPU once for all cases.
    let gpu = pollster::block_on(Gpu::new()).unwrap_or_else(|error| panic!("{error}"));
    // (case, chunk rows). The uneven chunk tests the short last chunk.
    let runs = [("attention_self", CHUNK_ROWS), ("attention_cross", CHUNK_ROWS), ("attention_self", UNEVEN_CHUNK_ROWS)];
    // Run each one, print its line, and keep a message for each one that fails.
    let failures: Vec<String> = runs
        .iter()
        .filter_map(|&(name, chunk_rows)| {
            // The case and its four files.
            let case = find_attention_case(name).unwrap_or_else(|error| panic!("{error}"));
            let query = read_case_file(case, "query.bin");
            let key = read_case_file(case, "key.bin");
            let value = read_case_file(case, "value.bin");
            let expected = read_case_file(case, "expected.bin");
            // The three attention tensors.
            let tensors = AttentionTensors { query: &query, key: &key, value: &value };
            // Compare with PyTorch and time it. block_on waits for the async function on this thread.
            let report = pollster::block_on(check_attention(&gpu, case, chunk_rows, &tensors, &expected)).unwrap_or_else(|error| panic!("{name}: {error}"));
            // Print the line. Run with `-- --nocapture` to see it.
            println!("{report}");
            // A NaN also fails, because NaN <= x is false.
            let within_tolerance = report.max_error() <= TOLERANCE;
            // For self attention, the full table is larger than everything we allocate. For cross attention it is
            // tiny (4096 x 77), so only the error counts there.
            let below_full_table = case.shape.key_tokens < case.shape.query_tokens || report.peak_bytes() < report.full_scores_bytes();
            // Keep a message only for a run that failed.
            (!(within_tolerance && below_full_table)).then(|| format!("{name} chunk {chunk_rows}: error {:e}, peak {} bytes", report.max_error(), report.peak_bytes()))
        })
        .collect();
    // All runs must pass.
    assert!(failures.is_empty(), "failed: {failures:?}");
}
