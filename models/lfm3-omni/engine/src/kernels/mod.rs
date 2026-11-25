// The kernels of LFM3 that engine-core does not have. Each binds its buffers once and gives a kernel ready to record.

// rms_norm.rs: RMSNorm of each row, or RMSNorm added to the residual stream.
mod rms_norm;
// short_conv.rs: the gated causal convolution of the short convolution layers.
mod short_conv;
// swiglu.rs: silu(gate) * up of the feed-forward layer.
mod swiglu;
// rope.rs: rotary positions of the queries and keys.
mod rope;
// attention.rs: causal attention with grouped keys and a gate per head.
mod attention;

pub use rms_norm::{NormMode, bind_rms_norm};
pub use short_conv::bind_short_conv;
pub use swiglu::bind_swiglu;
pub use rope::{RopeShape, bind_rope};
pub use attention::{AttentionBuffers, AttentionShape, HEAD_DIM, bind_attention};

// Invocations per workgroup of the element kernels. Their .wgsl files have @workgroup_size(256).
const WORKGROUP_SIZE: u32 = 256;
