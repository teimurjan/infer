// The LFM3-Omni-2B engine: Rust and hand-written WGSL kernels on top of packages/engine-core.

// The kernels that engine-core does not have: RMSNorm, the gated short convolution, SwiGLU, rotary positions, attention.
mod kernels;
// nn.Linear without bias, with the linear kernel of engine-core.
mod linear;
// The cos and sin tables of the rotary positions.
mod rope_table;
// The feed-forward half that every layer shares.
mod feed_forward;
// One short convolution layer of the language model.
mod conv_layer;
// One attention layer of the language model, full or sliding.
mod attention_layer;

pub use attention_layer::{AttentionKind, AttentionLayer};
pub use conv_layer::ConvLayer;
