// The kernels of LFM3 that engine-core does not have. Each binds its buffers once and gives a kernel ready to record.
// The kernels that depend on the position read it from the step buffer of memory.rs, so one bound kernel serves
// every word of an answer.

// rms_norm.rs: RMSNorm of each row, or RMSNorm added to the residual stream.
mod rms_norm;
// short_conv.rs: the gated causal convolution of the short convolution layers, over their history.
mod short_conv;
// swiglu.rs: silu(gate) * up of the feed-forward layer.
mod swiglu;
// rope.rs: rotary positions of the queries and keys.
mod rope;
// attention.rs: causal attention with grouped keys and a gate per head, over the memory of keys and values.
mod attention;
// copy_rows.rs: rows of one buffer into another, for the memory of keys and values and for the last row.
mod copy_rows;

pub use attention::{AttentionBuffers, AttentionShape, HEAD_DIM, bind_attention};
pub use copy_rows::{Target, bind_copy_rows};
pub use rms_norm::{NormMode, bind_rms_norm};
pub use rope::{RopeShape, RopeTables, bind_rope};
pub use short_conv::{ConvBuffers, ConvShape, bind_short_conv};
pub use swiglu::bind_swiglu;

// Invocations per workgroup of the element kernels. Their .wgsl files have @workgroup_size(256).
const WORKGROUP_SIZE: u32 = 256;
