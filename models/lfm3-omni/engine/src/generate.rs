// The word-by-word loop: the whole prompt in one run, then one run per new token, with greedy choice (the token with
// the best score), as generation_config.json asks (do_sample false, repetition_penalty 1.0).

// Our GPU connection, kernels, our error type, buffers, and the weights.
use engine_core::dispatch::encode_runs;
use engine_core::error::EngineError;
use engine_core::gpu::Gpu;
use engine_core::memory::{GpuBuffer, MemoryCategory};
use engine_core::weights::{GpuWeights, WeightSource};

// The model, its memory and its output layer.
use crate::language_model::{LanguageModel, token_rows};
use crate::memory::Memory;
use crate::output_layer::OutputLayer;

// The two tokens that end an answer (eos_token_id in generation_config.json): <|im_end|> and <|text_end|>.
pub const STOP_IDS: [u32; 2] = [124900, 125014];

// Everything a conversation needs: the weights, the output layer, the memory, and the model bound for one new token.
pub struct Generator<'a, S: WeightSource> {
    layers: &'a GpuWeights,
    table: &'a S,
    output: OutputLayer,
    memory: Memory,
    // The token row of each new word, and the model bound to it.
    word_row: GpuBuffer,
    word: LanguageModel,
}

impl<'a, S: WeightSource> Generator<'a, S> {
    // Upload the output layer and make the memory for up to `capacity` positions.
    pub fn new(gpu: &Gpu, layers: &'a GpuWeights, table: &'a S, capacity: u32) -> Result<Self, EngineError> {
        let output = OutputLayer::new(gpu, table)?;
        let memory = Memory::new(gpu, capacity);
        let word_row = gpu.empty_storage_buffer("word row", output.width() as usize);
        let word = LanguageModel::bind(gpu, layers, &output, &memory, 1, &word_row)?;
        Ok(Self { layers, table, output, memory, word_row, word })
    }

    // Answer `prompt` with up to `max_new` tokens. `on_step` sees the scores of each step before its token is chosen.
    // Returns the answer, its stop token included.
    pub async fn generate(&self, gpu: &Gpu, prompt: &[u32], max_new: usize, mut on_step: impl FnMut(usize, &[f32])) -> Result<Vec<u32>, EngineError> {
        // The prompt: one run of every prompt row, bound for this length.
        let rows = prompt.len() as u32;
        let hidden = gpu.storage_buffer_from("prompt rows", &token_rows(self.table, prompt)?, MemoryCategory::Activations);
        let model = LanguageModel::bind(gpu, self.layers, &self.output, &self.memory, rows, &hidden)?;
        self.memory.set_step(gpu, 0, rows)?;
        gpu.queue.submit([encode_runs(gpu, model.dispatches(), 1)]);
        let mut scores = model.scores(gpu).await?;
        // One run per new token.
        let mut answer = Vec::new();
        for step in 0..max_new {
            on_step(step, &scores);
            let token = best(&scores);
            answer.push(token);
            if STOP_IDS.contains(&token) || step + 1 == max_new {
                break;
            }
            // The new token is at position rows + step.
            self.memory.set_step(gpu, rows + step as u32, 1)?;
            gpu.write_f32(&self.word_row, &token_rows(self.table, &[token])?);
            gpu.queue.submit([encode_runs(gpu, self.word.dispatches(), 1)]);
            scores = self.word.scores(gpu).await?;
        }
        Ok(answer)
    }
}

// The token with the best score. The first one wins a tie, as torch.argmax does.
fn best(scores: &[f32]) -> u32 {
    let (index, _) = scores.iter().enumerate().fold((0, f32::MIN), |best, (index, &score)| if score > best.1 { (index, score) } else { best });
    index as u32
}
