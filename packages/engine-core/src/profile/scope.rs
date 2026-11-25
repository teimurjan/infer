// The type of the work that runs inside the scopes.
use std::future::Future;

// Our error type.
use crate::error::EngineError;
// Our GPU connection.
use crate::gpu::Gpu;

// Every kind of GPU error. A query set that the driver cannot allocate is an internal or an out-of-memory error, not a validation error.
const FILTERS: [wgpu::ErrorFilter; 3] = [wgpu::ErrorFilter::Validation, wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal];

// Run `work` while the GPU catches its errors, and turn the first caught error into an EngineError.
// Without this, a browser only prints a GPU error to the console and skips the bad submit, so its kernels never run.
pub(super) async fn caught<T>(gpu: &Gpu, label: &str, work: impl Future<Output = Result<T, EngineError>>) -> Result<T, EngineError> {
    // One scope per kind of error. Scopes are a stack, so they must come off in the reverse order.
    let scopes = FILTERS.map(|filter| gpu.device.push_error_scope(filter));
    // The work, inside the scopes.
    let result = work.await;
    // Take the scopes off, last one first, and keep the first error.
    let mut first_error = None;
    for scope in scopes.into_iter().rev() {
        // Each pop gives the error that its scope caught, if any.
        if let Some(error) = scope.pop().await {
            first_error.get_or_insert(error);
        }
    }
    // A GPU error wins over the result, because the result can come from work that did not run.
    match first_error {
        Some(error) => Err(EngineError::GpuError { label: label.to_string(), message: error.to_string() }),
        None => result,
    }
}
