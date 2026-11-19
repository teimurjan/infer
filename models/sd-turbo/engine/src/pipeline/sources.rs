// A future is the value of an async function before it runs.
use std::future::Future;
// Path is the location of a weight file.
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The two planned stages.
use crate::text::TextStage;
use crate::unet::UnetStage;
// The CPU weight reader, and the weights on the GPU.
use crate::weights::{GpuWeights, WeightSource};
// The native streaming loader.
#[cfg(not(target_arch = "wasm32"))]
use crate::weights::load_to_gpu;

// How the text encoder stage gets its weights. The pipeline calls `prepare` when the stage starts.
pub trait TextWeights {
    // Get the weights, make the slots, and bind the kernels.
    fn prepare(self, gpu: &Gpu) -> impl Future<Output = Result<TextStage, EngineError>>;
}

// How the UNet stage gets its weights. The pipeline calls `prepare` after the text encoder stage is freed,
// so the UNet weights are on the GPU only during the UNet stage.
pub trait UnetWeights {
    // Get the weights, make the slots, and bind the kernels.
    fn prepare(self, gpu: &Gpu) -> impl Future<Output = Result<UnetStage, EngineError>>;
}

// Float32 weights from a CPU reader, such as weights/unet.bin. Each kernel gets its own GPU copy of its weights.
pub struct FromReader<'a, W: WeightSource>(pub &'a W);

impl<W: WeightSource> TextWeights for FromReader<'_, W> {
    // Read and bind.
    async fn prepare(self, gpu: &Gpu) -> Result<TextStage, EngineError> {
        TextStage::prepare(gpu, self.0)
    }
}

impl<W: WeightSource> UnetWeights for FromReader<'_, W> {
    // Read and bind.
    async fn prepare(self, gpu: &Gpu) -> Result<UnetStage, EngineError> {
        UnetStage::prepare(gpu, self.0)
    }
}

// A weight file that is on the GPU already. The browser streams each file with fetch just before its stage.
// The stage takes the weights, so dropping the stage frees them.
pub struct Loaded(pub GpuWeights);

impl TextWeights for Loaded {
    // Bind to the loaded buffers.
    async fn prepare(self, gpu: &Gpu) -> Result<TextStage, EngineError> {
        TextStage::prepare_loaded(gpu, self.0)
    }
}

impl UnetWeights for Loaded {
    // Bind to the loaded buffers.
    async fn prepare(self, gpu: &Gpu) -> Result<UnetStage, EngineError> {
        UnetStage::prepare_loaded(gpu, self.0)
    }
}

// A weight file on disk that streams into GPU buffers when the stage starts, with the loader of the browser. Native only.
#[cfg(not(target_arch = "wasm32"))]
pub struct StreamedFile<'a>(pub &'a Path);

#[cfg(not(target_arch = "wasm32"))]
impl TextWeights for StreamedFile<'_> {
    // Stream, then bind.
    async fn prepare(self, gpu: &Gpu) -> Result<TextStage, EngineError> {
        TextStage::prepare_loaded(gpu, load_to_gpu(gpu, self.0).await?)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl UnetWeights for StreamedFile<'_> {
    // Stream, then bind.
    async fn prepare(self, gpu: &Gpu) -> Result<UnetStage, EngineError> {
        UnetStage::prepare_loaded(gpu, load_to_gpu(gpu, self.0).await?)
    }
}
