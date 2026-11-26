// The LFM3-Omni-2B engine: Rust and hand-written WGSL kernels on top of packages/engine-core.

// The kernels that engine-core does not have: RMSNorm, the gated short convolution, SwiGLU, rotary positions,
// attention and row copies.
mod kernels;
// nn.Linear without bias, with the linear kernel of engine-core.
mod linear;
// The cos and sin tables of the rotary positions.
mod rope_table;
// The sizes and the layer kinds of the language model.
mod config;
// The memory of earlier tokens, and the step buffer.
mod memory;
// The feed-forward half that every layer shares.
mod feed_forward;
// One short convolution layer of the language model.
mod conv_layer;
// One attention layer of the language model, full or sliding.
mod attention_layer;
// The final norm and the output layer.
mod output_layer;
// The language model for one run of new tokens.
mod language_model;
// The word-by-word loop.
mod generate;

pub use attention_layer::AttentionLayer;
pub use config::{LAYER_KINDS, LayerKind};
pub use conv_layer::ConvLayer;
pub use generate::{Generator, STOP_IDS};
pub use language_model::{LanguageModel, token_rows};
// nn.Linear without bias, float or packed. tests/grouped_linear.rs checks the packed kernel with it.
pub use linear::bind_linear;
pub use memory::Memory;
pub use output_layer::OutputLayer;
