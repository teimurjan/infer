// The sizes and the layer kinds of the LFM3-Omni-2B language model, from text_config in config.json
// (commit c01718a568e53ae936e6b5e1ff5ab2de7d7f388b). The weights say whether a layer is a convolution or an attention
// layer, but not whether its attention is full or sliding, so the list is written here.

// Values in one token row (hidden_size).
pub const WIDTH: u32 = 2048;
// Key and value heads (num_key_value_heads), of 128 values each.
pub const KV_HEADS: u32 = 4;

// The three kinds of layer.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LayerKind {
    // A short convolution layer.
    Conv,
    // Attention over every earlier token.
    Full,
    // Attention over the last 256 tokens.
    Sliding,
}

// Shorter names for the list below.
use LayerKind::{Conv, Full, Sliding};

// layer_types, in order.
pub const LAYER_KINDS: [LayerKind; 24] = [
    Conv, Conv, Full, Conv, Sliding, Conv, Full, Conv, Sliding, Conv, Full, Conv,
    Sliding, Conv, Full, Conv, Sliding, Conv, Full, Conv, Sliding, Conv, Full, Conv,
];

// The rotary settings of an attention layer, from rope_parameters and sliding_window.
#[derive(Clone, Copy)]
pub struct RopeSettings {
    // rope_theta.
    pub theta: f32,
    // Rotated pairs out of 64: partial_rotary_factor x 128 / 2.
    pub angles: u32,
    // The keys that a query sees, its own included. 0: every earlier key.
    pub window: u32,
}

// The settings of the full and the sliding attention layers. A convolution layer has none.
pub fn rope_settings(kind: LayerKind) -> Option<RopeSettings> {
    match kind {
        Conv => None,
        // Theta 1,000,000, partial_rotary_factor 0.5: 32 of 64 pairs turn.
        Full => Some(RopeSettings { theta: 1_000_000.0, angles: 32, window: 0 }),
        // Theta 10,000, every pair turns, a window of 256 keys.
        Sliding => Some(RopeSettings { theta: 10_000.0, angles: 64, window: 256 }),
    }
}
