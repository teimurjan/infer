// The kind of each of the 24 layers of the LFM3-Omni-2B language model, from text_config.layer_types in config.json
// (commit c01718a568e53ae936e6b5e1ff5ab2de7d7f388b). The weights say whether a layer is a convolution or an attention
// layer, but not whether its attention is full or sliding, so the list is written here.

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
