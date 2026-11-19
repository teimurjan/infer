// engine.rs is the Engine object: one GPU connection for the page, and the weight loader.
mod engine;

// wasm_bindgen marks Rust functions that JavaScript can call. JsError is a JavaScript Error object.
use wasm_bindgen::prelude::{JsError, wasm_bindgen};

// Our GPU connection.
use crate::gpu::Gpu;
// The three tensors a layer reads.
use crate::layers::Tensors;
// The three tensors of an attention call.
use crate::attention::AttentionTensors;
// The weight reader over bytes that the page fetched.
use crate::weights::WeightBytes;

// JavaScript calls this as `await siluMaxError(input, expected)` with two Float32Arrays.
// wasm-bindgen copies each Float32Array into a Vec<f32> and turns the result into a Promise.
#[wasm_bindgen(js_name = siluMaxError)]
pub async fn silu_max_error(input: Vec<f32>, expected: Vec<f32>) -> Result<f32, JsError> {
    // Run the same check as the native test. `?` turns an EngineError into a JavaScript Error.
    Ok(crate::silu_max_error(&input, &expected).await?)
}

// JavaScript calls this as `await runBenchmarks()`. It returns one line per benchmark.
#[wasm_bindgen(js_name = runBenchmarks)]
pub async fn run_benchmarks() -> Result<String, JsError> {
    // Open the GPU.
    let gpu = Gpu::new().await?;
    // Run the same benchmarks as the native example.
    let reports = crate::run_benchmarks(&gpu).await?;
    // One line per report, the same text the native example prints.
    Ok(reports.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n"))
}

// JavaScript calls this as `caseNames()` to learn which testdata folders to download.
#[wasm_bindgen(js_name = caseNames)]
pub fn case_names() -> Vec<String> {
    // The name of every case, in order.
    crate::CASES.iter().map(|case| case.name.to_string()).collect()
}

// JavaScript calls this as `await checkCase(name, input, weight, bias, expected)` with four Float32Arrays.
#[wasm_bindgen(js_name = checkCase)]
pub async fn check_case(name: String, input: Vec<f32>, weight: Vec<f32>, bias: Vec<f32>, expected: Vec<f32>) -> Result<String, JsError> {
    // Find the shape for this name.
    let case = crate::find_case(&name)?;
    // Open the GPU.
    let gpu = Gpu::new().await?;
    // The three layer tensors.
    let tensors = Tensors { input: &input, weight: &weight, bias: &bias };
    // Run the same check as the native test, and return its line.
    Ok(crate::check_case(&gpu, case, &tensors, &expected).await?.to_string())
}

// JavaScript calls this as `attentionCaseNames()` to learn which testdata folders to download.
#[wasm_bindgen(js_name = attentionCaseNames)]
pub fn attention_case_names() -> Vec<String> {
    // The name of every attention case, in order.
    crate::ATTENTION_CASES.iter().map(|case| case.name.to_string()).collect()
}

// JavaScript calls this as `await checkAttention(name, chunkRows, query, key, value, expected)` with four Float32Arrays.
#[wasm_bindgen(js_name = checkAttention)]
pub async fn check_attention(name: String, chunk_rows: u32, query: Vec<f32>, key: Vec<f32>, value: Vec<f32>, expected: Vec<f32>) -> Result<String, JsError> {
    // Find the shape for this name.
    let case = crate::find_attention_case(&name)?;
    // Open the GPU.
    let gpu = Gpu::new().await?;
    // The three attention tensors.
    let tensors = AttentionTensors { query: &query, key: &key, value: &value };
    // Run the same check as the native test, and return its line.
    Ok(crate::check_attention(&gpu, case, chunk_rows, &tensors, &expected).await?.to_string())
}

// The result of decodeTaesd, as a JavaScript object with two read-only fields.
#[wasm_bindgen]
pub struct TaesdResult {
    // One line per part, joined with newlines.
    report: String,
    // The image as RGBA bytes, 512 x 512 x 4.
    rgba: Vec<u8>,
}

// The fields that JavaScript can read.
#[wasm_bindgen]
impl TaesdResult {
    // `result.report` in JavaScript.
    #[wasm_bindgen(getter)]
    pub fn report(&self) -> String {
        // wasm-bindgen needs an owned String to copy into JavaScript.
        self.report.clone()
    }

    // `result.rgba` in JavaScript, a Uint8Array.
    #[wasm_bindgen(getter)]
    pub fn rgba(&self) -> Vec<u8> {
        // wasm-bindgen needs an owned Vec to copy into JavaScript.
        self.rgba.clone()
    }
}

// JavaScript calls this as `await decodeTaesd(indexJson, weights, latent, expected)`.
// indexJson is the text of weights/taesd.json, weights is a Uint8Array of weights/taesd.bin, latent and expected are Float32Arrays.
#[wasm_bindgen(js_name = decodeTaesd)]
pub async fn decode_taesd(index_json: String, weights: Vec<u8>, latent: Vec<f32>, expected: Vec<f32>) -> Result<TaesdResult, JsError> {
    // Parse the index. The bytes move into the reader, so they are not copied again.
    let weights = WeightBytes::new(&index_json, weights)?;
    // Open the GPU.
    let gpu = Gpu::new().await?;
    // Decode and report, the same as the native example.
    let report = crate::decode_and_report(&gpu, &weights, &latent, &expected).await?;
    // One string for the page, and the image bytes.
    Ok(TaesdResult { report: report.lines.join("\n"), rgba: report.rgba })
}


// The prompt tokenizer for the page. JavaScript: `const tokenizer = new PromptTokenizer(vocabText, mergesText)`,
// with the text of weights/tokenizer_vocab.json and weights/tokenizer_merges.txt, then `tokenizer.encode(prompt)`.
#[wasm_bindgen]
pub struct PromptTokenizer(crate::Tokenizer);

#[wasm_bindgen]
impl PromptTokenizer {
    // Read the two files. A wrong file becomes a JavaScript Error.
    #[wasm_bindgen(constructor)]
    pub fn new(vocab: &str, merges: &str) -> Result<PromptTokenizer, JsError> {
        Ok(Self(crate::Tokenizer::new(vocab, merges)?))
    }

    // The 77 token ids of `prompt`, as a Uint32Array.
    pub fn encode(&self, prompt: &str) -> Vec<u32> {
        self.0.encode(prompt)
    }
}

// JavaScript: `startingNoise(seed)`. The starting noise of a seed (a whole number from 0 to 4294967295), as a Float32Array
// of 4 x 64 x 64 values. The same values as torch.randn with torch.Generator("cpu").manual_seed(seed).
#[wasm_bindgen(js_name = startingNoise)]
pub fn starting_noise(seed: u32) -> Vec<f32> {
    crate::starting_noise(seed)
}
