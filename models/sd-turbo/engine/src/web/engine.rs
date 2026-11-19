// RefCell lets the stage methods change the state through a shared handle.
use std::cell::RefCell;
// Rc shares the GPU and the state with the promises that the methods give to JavaScript.
use std::rc::Rc;

// Promise is a JavaScript Promise. future_to_promise turns a Rust future into one.
use wasm_bindgen_futures::{future_to_promise, js_sys::Promise};
// wasm_bindgen marks what JavaScript can call. JsError is a JavaScript Error. JsValue is any JavaScript value.
use wasm_bindgen::prelude::{JsError, JsValue, wasm_bindgen};

// Our GPU connection.
use crate::gpu::Gpu;
// Our error type.
use crate::error::EngineError;
// The RGBA bytes of the image.
use crate::image::image_to_rgba;
// The stages, the scheduler step, and the source of weights that are on the GPU already.
use crate::pipeline::{Loaded, StageEvent, decoder_stage, euler_step, scale_input, text_stage, unet_stage};
// The UNet inputs.
use crate::unet::UnetInputs;
// The timing per kernel.
use crate::profile::{ProfileMode, best_mode, start_profile};
// The streaming loader, what it gives back, and the reader of a small file in memory.
use crate::weights::{GpuWeights, WeightBytes, WeightUpload};

// The result of runDecoder: the report lines and the RGBA bytes.
use super::TaesdResult;

// The image size.
const SIDE: usize = 512;

// What the engine keeps between the calls of the page.
#[derive(Default)]
struct State {
    // The file that streams in now, and its name.
    upload: Option<(String, WeightUpload)>,
    // The finished files and their names, in load order. A stage takes its file out of this list.
    loaded: Vec<(String, GpuWeights)>,
    // The text encoder output, from runTextEncoder for runUnet.
    text: Option<Vec<f32>>,
    // The latent after the scheduler step, from runUnet for runDecoder.
    latent: Option<Vec<f32>>,
}

impl State {
    // Take the loaded file `name` out of the list. Its stage owns it from now on.
    fn take(&mut self, name: &str) -> Result<GpuWeights, JsError> {
        // Find it by name.
        let position = self.loaded.iter().position(|(loaded, _)| loaded == name).ok_or_else(|| JsError::new(&format!("{name} is not loaded")))?;
        // Remove it and give it back.
        Ok(self.loaded.remove(position).1)
    }
}

// One GPU connection that stays open while the page lives, with the weight files loaded so far.
// JavaScript reads each weight file as a stream and passes the chunks in, one at a time.
// The page runs a generation one stage at a time: it loads the file of a stage, runs the stage, and the stage frees it.
#[wasm_bindgen]
pub struct Engine {
    // The GPU. Rc, because a promise can outlive the call that made it.
    gpu: Rc<Gpu>,
    // Everything that changes between calls. Rc, for the same reason.
    state: Rc<RefCell<State>>,
}

#[wasm_bindgen]
impl Engine {
    // JavaScript: `const engine = await Engine.open()`.
    pub async fn open() -> Result<Engine, JsError> {
        // Open the GPU once for the whole page.
        let gpu = Gpu::new().await?;
        // Nothing is loaded yet.
        Ok(Self { gpu: Rc::new(gpu), state: Rc::default() })
    }

    // JavaScript: `engine.beginWeights("unet_packed", indexText)`. Start a new file. An unfinished file is dropped, and its buffers are freed.
    #[wasm_bindgen(js_name = beginWeights)]
    pub fn begin_weights(&self, name: String, index_json: &str) -> Result<(), JsError> {
        // Parse the index and check the layout. No buffer exists yet.
        let upload = WeightUpload::begin(&self.gpu, index_json)?;
        // Keep it until finishWeights.
        self.state.borrow_mut().upload = Some((name, upload));
        Ok(())
    }

    // JavaScript: `if (engine.writeWeights(chunk)) await engine.settle()`. Write one chunk (a Uint8Array) into the GPU buffers.
    // wasm-bindgen copies the chunk into WASM memory for this call and frees it after, so WASM memory holds one chunk at most.
    #[wasm_bindgen(js_name = writeWeights)]
    pub fn write_weights(&self, chunk: &[u8]) -> Result<bool, JsError> {
        // The state, for this call only.
        let mut state = self.state.borrow_mut();
        // A chunk without a file means the page called the functions in the wrong order.
        let (_, upload) = state.upload.as_mut().ok_or_else(|| JsError::new("writeWeights before beginWeights"))?;
        // True when the page must wait for the GPU before the next chunk.
        Ok(upload.write(&self.gpu, chunk))
    }

    // JavaScript: `const line = engine.finishWeights()`. Check that every tensor arrived, and keep the file. Returns one summary line.
    #[wasm_bindgen(js_name = finishWeights)]
    pub fn finish_weights(&self) -> Result<String, JsError> {
        // The state, for this call only.
        let mut state = self.state.borrow_mut();
        // Take the file out. On an error below, its buffers are freed.
        let (name, upload) = state.upload.take().ok_or_else(|| JsError::new("finishWeights before beginWeights"))?;
        // Write the last bytes and check the length.
        let weights = upload.finish(&self.gpu)?;
        // The line for the page, before the weights move into the list.
        let line = format!("{name}: {weights}");
        // Keep the buffers alive.
        state.loaded.push((name, weights));
        Ok(line)
    }

    // JavaScript: `await engine.settle()`. Send the queued writes to the GPU and wait until they are done.
    pub fn settle(&self) -> Promise {
        // A handle to the GPU for the promise. The promise can run after this call returns.
        let gpu = Rc::clone(&self.gpu);
        // Wait for the GPU. An error becomes a rejected promise.
        future_to_promise(async move { gpu.finish_uploads().await.map(|()| JsValue::UNDEFINED).map_err(to_js) })
    }

    // JavaScript: `engine.memoryReport()`. The live bytes of the engine's GPU buffers per category, one line each.
    #[wasm_bindgen(js_name = memoryReport)]
    pub fn memory_report(&self) -> String {
        self.gpu.memory_report().to_string()
    }

    // JavaScript: `engine.liveBytes()`. The bytes of every engine GPU buffer that exists now.
    // A Number, not a BigInt: a Number holds whole numbers up to 2^53 exactly, far above the 1920 MB budget.
    #[wasm_bindgen(js_name = liveBytes)]
    pub fn live_bytes(&self) -> f64 {
        self.gpu.memory_report().total() as f64
    }

    // JavaScript: `engine.takeLongestSubmit()`. The seconds of the longest submit since the last call. The next call starts from 0.
    #[wasm_bindgen(js_name = takeLongestSubmit)]
    pub fn take_longest_submit(&self) -> f64 {
        self.gpu.take_longest_submit()
    }

    // JavaScript: `const line = await engine.startProfile(false)`. Measure the door number with the copy benchmark of module 3,
    // then time every kernel of the next stages. `wall_clock` true uses the CPU clock even on a device with timestamps.
    // The promise gives the copy line and the timing mode.
    #[wasm_bindgen(js_name = startProfile)]
    pub fn start_profile(&self, wall_clock: bool) -> Promise {
        // A handle to the GPU for the promise.
        let gpu = Rc::clone(&self.gpu);
        // The benchmark runs after this call returns.
        future_to_promise(async move {
            // Timestamps if the device has them, unless the page asks for the CPU clock.
            let mode = if wall_clock { ProfileMode::WallClock } else { best_mode(&gpu) };
            // The copy benchmark, then the profile starts.
            let door = start_profile(&gpu, mode).await.map_err(to_js)?;
            // The line for the page.
            Ok(JsValue::from(format!("{door}\ntimestamp-query on this device: {}", gpu.has_timestamps())))
        })
    }

    // JavaScript: `engine.profileTable()`. End the profile and give its table. An empty string if no profile ran.
    #[wasm_bindgen(js_name = profileTable)]
    pub fn profile_table(&self) -> String {
        self.gpu.take_profile().map(|profile| profile.table().to_string()).unwrap_or_default()
    }

    // JavaScript: `engine.abortWeights()`. Drop the unfinished file after a failed stream. Its GPU buffers are freed.
    #[wasm_bindgen(js_name = abortWeights)]
    pub fn abort_weights(&self) {
        self.state.borrow_mut().upload = None;
    }

    // JavaScript: `engine.freeWeights()`. Drop every loaded file and the unfinished one. Their GPU buffers are freed.
    #[wasm_bindgen(js_name = freeWeights)]
    pub fn free_weights(&self) {
        // The state, for this call only.
        let mut state = self.state.borrow_mut();
        // Dropping a GpuWeights drops its buffers, and each buffer leaves the counter.
        state.loaded.clear();
        state.upload = None;
    }

    // JavaScript: `const lines = await engine.runTextEncoder("text_encoder", ids)`. Stage 1. `ids` is a Uint32Array of 77 token ids.
    // The stage takes the loaded file `name`, runs, and is dropped, which frees the file. The promise gives the report lines.
    #[wasm_bindgen(js_name = runTextEncoder)]
    pub fn run_text_encoder(&self, name: String, ids: Vec<u32>) -> Promise {
        // Handles for the promise.
        let (gpu, state) = (Rc::clone(&self.gpu), Rc::clone(&self.state));
        // The stage runs after this call returns.
        future_to_promise(async move {
            // The file moves into the stage.
            let weights = state.borrow_mut().take(&name)?;
            // Collect the report lines.
            let mut lines = Vec::new();
            // Prepare, run, free.
            let text = text_stage(&gpu, Loaded(weights), &ids, &mut |event: &StageEvent| lines.push(event.to_string())).await.map_err(to_js)?;
            // Keep the output for the UNet stage.
            state.borrow_mut().text = Some(text);
            // The lines for the page.
            Ok(JsValue::from(lines.join("\n")))
        })
    }

    // JavaScript: `const lines = await engine.runUnet("unet_packed", noise, sigma, timestep)`. Stage 2, after runTextEncoder.
    // `noise` is a Float32Array [1, 4, 64, 64]. The stage takes the loaded file `name`. Then the scheduler step runs on the CPU.
    #[wasm_bindgen(js_name = runUnet)]
    pub fn run_unet(&self, name: String, noise: Vec<f32>, sigma: f32, timestep: f32) -> Promise {
        // Handles for the promise.
        let (gpu, state) = (Rc::clone(&self.gpu), Rc::clone(&self.state));
        // The stage runs after this call returns.
        future_to_promise(async move {
            // The file moves into the stage. The text output must exist.
            let (weights, text) = {
                let mut state = state.borrow_mut();
                let text = state.text.take().ok_or_else(|| JsError::new("runUnet before runTextEncoder"))?;
                (state.take(&name)?, text)
            };
            // The UNet input.
            let sample = scale_input(&noise, sigma);
            // Collect the report lines.
            let mut lines = Vec::new();
            // Prepare, run, free.
            let inputs = UnetInputs { sample: &sample, timestep, text: &text };
            let noise_prediction = unet_stage(&gpu, Loaded(weights), &inputs, &mut |event: &StageEvent| lines.push(event.to_string())).await.map_err(to_js)?;
            // The scheduler step, and keep the latent for the decoder.
            state.borrow_mut().latent = Some(euler_step(&noise, &noise_prediction, sigma));
            // The lines for the page.
            Ok(JsValue::from(lines.join("\n")))
        })
    }

    // JavaScript: `const result = await engine.runDecoder(indexText, bytes)`. Stage 3, after runUnet. `bytes` is weights/taesd.bin
    // as a Uint8Array (9.8 MB, small enough for WASM memory). The promise gives {report, rgba}.
    #[wasm_bindgen(js_name = runDecoder)]
    pub fn run_decoder(&self, index_json: String, bytes: Vec<u8>) -> Promise {
        // Handles for the promise.
        let (gpu, state) = (Rc::clone(&self.gpu), Rc::clone(&self.state));
        // The stage runs after this call returns.
        future_to_promise(async move {
            // The latent must exist.
            let latent = state.borrow_mut().latent.take().ok_or_else(|| JsError::new("runDecoder before runUnet"))?;
            // The reader over the bytes.
            let weights = WeightBytes::new(&index_json, bytes).map_err(to_js)?;
            // Collect the report lines.
            let mut lines = Vec::new();
            // Decode.
            let image = decoder_stage(&gpu, &weights, &latent, &mut |event: &StageEvent| lines.push(event.to_string())).await.map_err(to_js)?;
            // The image as RGBA bytes for a canvas.
            let rgba = image_to_rgba(&image, SIDE, SIDE).map_err(to_js)?;
            // The result for the page.
            Ok(JsValue::from(TaesdResult { report: lines.join("\n"), rgba }))
        })
    }
}

// An engine error as a JavaScript Error, for a rejected promise.
fn to_js(error: EngineError) -> JsValue {
    JsError::from(error).into()
}
