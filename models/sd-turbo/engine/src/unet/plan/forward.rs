// The graph that records the ops.
use crate::plan::{Graph, TensorId};

// The ResNet block and the transformer block.
use super::resnet::resnet;
use super::transformer::transformer;

// The numbers below come from unet/config.json of stabilityai/sd-turbo.
// Channels of the four levels (block_out_channels).
pub(crate) const LEVEL_CHANNELS: [u32; 4] = [320, 640, 1280, 1280];
// Attention heads of the four levels. diffusers calls this attention_head_dim, but for SD 2 it is the head count.
const LEVEL_HEADS: [u32; 4] = [5, 10, 20, 20];
// ResNet blocks per down level (layers_per_block). An up level has one more.
pub(crate) const DOWN_LAYERS: usize = 2;
// The input has 4 latent channels at 64 x 64 (in_channels, sample_size).
const LATENT: [u32; 3] = [4, 64, 64];
// The text encoder output: 77 tokens of 1024 values (cross_attention_dim).
const TEXT: [u32; 3] = [77, 1024, 1];
// The sinusoidal time features, which the CPU computes.
use crate::unet::time::TIME_FEATURES;
// The time embedding is 4 times the first level (time_embed_dim).
const TIME_CHANNELS: u32 = 1280;
// The golden names of the down and up block outputs, in order.
pub(crate) const DOWN_NAMES: [&str; 4] = ["down_block_0", "down_block_1", "down_block_2", "down_block_3"];
pub(crate) const UP_NAMES: [&str; 4] = ["up_block_0", "up_block_1", "up_block_2", "up_block_3"];

// The UNet forward pass as ops, and the tensors that have a golden file.
pub(crate) struct UnetGraph {
    // Every op and tensor.
    pub(crate) graph: Graph,
    // (golden name, tensor) for each block output, in run order. Only the golden shape test reads it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) blocks: Vec<(&'static str, TensorId)>,
}

// Record the full UNet forward pass of one denoising step, op by op, in the order of diffusers' UNet2DConditionModel.forward.
pub(crate) fn unet_graph() -> UnetGraph {
    // An empty graph.
    let mut graph = Graph::default();
    // The scaled latent. The scheduler step after the UNet reads it again, so it lives to the end.
    let sample = graph.op("sample", &[], LATENT);
    // Keep it for the scheduler.
    graph.keep(sample);
    // The text encoder output from stage 1. The last cross attention is its last reader.
    let text = graph.op("encoder_hidden_states", &[], TEXT);
    // The time embedding, which every ResNet block reads.
    let time = time_embedding(&mut graph);
    // conv_in: 3x3 convolution, 4 to 320 channels.
    let conv_in = graph.op("conv_in", &[sample], [LEVEL_CHANNELS[0], LATENT[1], LATENT[2]]);
    // The golden blocks so far.
    let mut blocks = vec![("time_embedding", time), ("conv_in", conv_in)];
    // The down path keeps a skip tensor after each ResNet or transformer pair and after each downsample.
    let (hidden, skips) = down_path(&mut graph, conv_in, time, text, &mut blocks);
    // The middle block.
    let hidden = mid_block(&mut graph, hidden, time, text);
    // Its golden name.
    blocks.push(("mid_block", hidden));
    // The up path reads the skip tensors in reverse order.
    let hidden = up_path(&mut graph, hidden, &skips, time, text, &mut blocks);
    // conv_norm_out: GroupNorm and SiLU in one kernel.
    let norm = graph.op("conv_norm_out", &[hidden], graph.dims(hidden));
    // conv_out: 3x3 convolution, 320 to 4 channels.
    let output = graph.op("conv_out", &[norm], LATENT);
    // The scheduler reads the output after the UNet.
    graph.keep(output);
    // Its golden name.
    blocks.push(("unet_output", output));
    // The full pass.
    UnetGraph { graph, blocks }
}

// Sinusoidal features, linear_1, SiLU, linear_2. Returns the time embedding.
// The kernel of linear_2 applies SiLU to its input as it reads it, and so does each time_emb_proj. So no op writes a SiLU tensor.
fn time_embedding(graph: &mut Graph) -> TensorId {
    // The 320 features. The CPU computes them from the timestep and writes them into this slot.
    let features = graph.op("time_proj", &[], [TIME_FEATURES as u32, 1, 1]);
    // linear_1: 320 to 1280.
    let first = graph.op("time_embedding.linear_1", &[features], [TIME_CHANNELS, 1, 1]);
    // SiLU, then linear_2: 1280 to 1280. This is the time embedding.
    graph.op("time_embedding.linear_2", &[first], [TIME_CHANNELS, 1, 1])
}

// The four down levels. Returns the last output and the 12 skip tensors, in the order they were made.
fn down_path(graph: &mut Graph, conv_in: TensorId, time: TensorId, text: TensorId, blocks: &mut Vec<(&'static str, TensorId)>) -> (TensorId, Vec<TensorId>) {
    // The conv_in output is the first skip tensor.
    let mut skips = vec![conv_in];
    // The tensor that flows down.
    let mut hidden = conv_in;
    // Levels 0 to 3.
    for (level, &channels) in LEVEL_CHANNELS.iter().enumerate() {
        // For example "down_blocks.0".
        let prefix = format!("down_blocks.{level}");
        // The last level has no attention and no downsample (DownBlock2D).
        let last = level == LEVEL_CHANNELS.len() - 1;
        // Two ResNet blocks, each followed by a transformer on levels 0 to 2.
        for layer in 0..DOWN_LAYERS {
            // The ResNet block.
            hidden = resnet(graph, &format!("{prefix}.resnets.{layer}"), hidden, time, channels);
            // The transformer, except on the last level.
            if !last {
                hidden = transformer(graph, &format!("{prefix}.attentions.{layer}"), hidden, text, LEVEL_HEADS[level]);
            }
            // The up path reads this tensor later.
            skips.push(hidden);
        }
        // Levels 0 to 2 end with a 3x3 convolution of stride 2.
        if !last {
            // Half the height and the width.
            let [_, height, width] = graph.dims(hidden);
            // The downsample.
            hidden = graph.op(format!("{prefix}.downsamplers.0.conv"), &[hidden], [channels, height / 2, width / 2]);
            // It is a skip tensor too.
            skips.push(hidden);
        }
        // The golden name of this level.
        blocks.push((DOWN_NAMES[level], hidden));
    }
    // The bottom output and every skip tensor.
    (hidden, skips)
}

// UNetMidBlock2DCrossAttn: ResNet, transformer, ResNet, at 1280 channels.
fn mid_block(graph: &mut Graph, input: TensorId, time: TensorId, text: TensorId) -> TensorId {
    // The channels and heads of the last level.
    let (channels, heads) = (LEVEL_CHANNELS[3], LEVEL_HEADS[3]);
    // The first ResNet block.
    let hidden = resnet(graph, "mid_block.resnets.0", input, time, channels);
    // The transformer.
    let hidden = transformer(graph, "mid_block.attentions.0", hidden, text, heads);
    // The second ResNet block.
    resnet(graph, "mid_block.resnets.1", hidden, time, channels)
}

// The four up levels. Each ResNet block reads the up tensor joined with one skip tensor along the channels.
fn up_path(graph: &mut Graph, input: TensorId, skips: &[TensorId], time: TensorId, text: TensorId, blocks: &mut Vec<(&'static str, TensorId)>) -> TensorId {
    // The newest skip tensor comes first.
    let reversed: Vec<TensorId> = skips.iter().rev().copied().collect();
    // The tensor that flows up.
    let mut hidden = input;
    // Each up level reads 3 skip tensors: DOWN_LAYERS + 1.
    for (level, level_skips) in reversed.chunks(DOWN_LAYERS + 1).enumerate() {
        // The up levels use the down levels in reverse: 1280, 1280, 640, 320.
        let down_level = LEVEL_CHANNELS.len() - 1 - level;
        // For example "up_blocks.0".
        let prefix = format!("up_blocks.{level}");
        // One ResNet block per skip tensor.
        for (layer, &skip) in level_skips.iter().enumerate() {
            // The ResNet name.
            let resnet_name = format!("{prefix}.resnets.{layer}");
            // The two sizes.
            let ([up_channels, height, width], [skip_channels, _, _]) = (graph.dims(hidden), graph.dims(skip));
            // torch.cat along the channels, into a new tensor.
            let joined = graph.op(format!("{resnet_name}.concat"), &[hidden, skip], [up_channels + skip_channels, height, width]);
            // The ResNet block.
            hidden = resnet(graph, &resnet_name, joined, time, LEVEL_CHANNELS[down_level]);
            // Up level 0 (UpBlock2D) has no attention. The others do.
            if level > 0 {
                hidden = transformer(graph, &format!("{prefix}.attentions.{layer}"), hidden, text, LEVEL_HEADS[down_level]);
            }
        }
        // Levels 0 to 2 end with a 2x nearest upsample and a 3x3 convolution.
        if down_level > 0 {
            // Twice the height and the width.
            let [channels, height, width] = graph.dims(hidden);
            // The upsampled copy.
            let upsampled = graph.op(format!("{prefix}.upsamplers.0.interpolate"), &[hidden], [channels, height * 2, width * 2]);
            // The convolution.
            hidden = graph.op(format!("{prefix}.upsamplers.0.conv"), &[upsampled], [channels, height * 2, width * 2]);
        }
        // The golden name of this level.
        blocks.push((UP_NAMES[level], hidden));
    }
    // The top output.
    hidden
}
