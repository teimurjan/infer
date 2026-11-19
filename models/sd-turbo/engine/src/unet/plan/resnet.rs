// The graph that records the ops.
use crate::plan::{Graph, TensorId};

// One ResnetBlock2D of diffusers, as the ops that the engine will run:
// GroupNorm + SiLU, conv1, add the time vector, GroupNorm + SiLU, conv2, then add the input (through a 1x1 convolution if the channels change).
// `time` is the time embedding, [1280]. time_emb_proj applies SiLU to it as it reads it. Returns the output, [out_channels, height, width].
pub(super) fn resnet(graph: &mut Graph, prefix: &str, input: TensorId, time: TensorId, out_channels: u32) -> TensorId {
    // The input size.
    let [in_channels, height, width] = graph.dims(input);
    // The size of every tensor after conv1.
    let out = [out_channels, height, width];
    // norm1 (GroupNorm with 32 groups) and SiLU in one kernel.
    let norm1 = graph.op(format!("{prefix}.norm1"), &[input], [in_channels, height, width]);
    // time_emb_proj: SiLU, then a linear layer from 1280 to out_channels on the time vector.
    let time_vector = graph.op(format!("{prefix}.time_emb_proj"), &[time], [out_channels, 1, 1]);
    // conv1: 3x3 convolution.
    let hidden = graph.op(format!("{prefix}.conv1"), &[norm1], out);
    // Add the time vector to every pixel of its channel, in place.
    graph.update(hidden, &[time_vector]);
    // norm2 and SiLU in one kernel.
    let norm2 = graph.op(format!("{prefix}.norm2"), &[hidden], out);
    // conv2: 3x3 convolution.
    let output = graph.op(format!("{prefix}.conv2"), &[norm2], out);
    // The skip path: the input itself, or a 1x1 convolution when the channel count changes.
    let shortcut = if in_channels == out_channels { input } else { graph.op(format!("{prefix}.conv_shortcut"), &[input], out) };
    // Add the skip path to the output, in place.
    graph.update(output, &[shortcut]);
    // The block output.
    output
}
