// The LFM3-Omni-2B engine: Rust and hand-written WGSL kernels on top of packages/engine-core.

// The kernels that engine-core does not have: RMSNorm, the gated short convolution and SwiGLU.
mod kernels;
// nn.Linear without bias, with the linear kernel of engine-core.
mod linear;
// One short convolution layer of the language model.
mod conv_layer;

pub use conv_layer::ConvLayer;
