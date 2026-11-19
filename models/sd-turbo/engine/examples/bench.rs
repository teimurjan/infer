// ExitCode sets the exit code: 0 for success, 1 for failure.
use std::process::ExitCode;

// The engine parts that this program uses.
use engine::{EngineError, Gpu, run_benchmarks};

// Run the benchmarks on the native GPU. Start it with `cargo run --release --example bench`.
fn main() -> ExitCode {
    // block_on waits for the async function on this thread.
    match pollster::block_on(run()) {
        // Every benchmark finished.
        Ok(()) => ExitCode::SUCCESS,
        // Something failed. Print why.
        Err(error) => {
            // Errors go to stderr.
            eprintln!("benchmark failed: {error}");
            // Exit code 1 tells scripts that it failed.
            ExitCode::FAILURE
        }
    }
}

// Open the GPU, run every benchmark, and print one line each.
async fn run() -> Result<(), EngineError> {
    // Open the GPU.
    let gpu = Gpu::new().await?;
    // Print each report as one line.
    run_benchmarks(&gpu).await?.iter().for_each(|report| println!("{report}"));
    // Done.
    Ok(())
}
