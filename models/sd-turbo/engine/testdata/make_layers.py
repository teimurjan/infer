# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# Path builds the folder paths.
from pathlib import Path

# torch makes the layers, the random data and the expected outputs.
import torch

# Each case gets a folder next to this script.
OUT_DIR = Path(__file__).parent
# A fixed seed, so every run makes the same data.
SEED = 0

# name: (SD-Turbo layer, a function that makes the layer, input shape).
# The names and shapes must match CASES in src/cases.rs.
CASES = {
    # Attention input projection: 4096 tokens of 320 values.
    "linear": ("down_blocks.0.attentions.0.proj_in", lambda: torch.nn.Linear(320, 320), (4096, 320)),
    # ResNet shortcut, 320 to 640 channels at 32 x 32.
    "conv1x1": ("down_blocks.1.resnets.0.conv_shortcut", lambda: torch.nn.Conv2d(320, 640, 1), (1, 320, 32, 32)),
    # First ResNet convolution, 320 channels at 64 x 64.
    "conv3x3": ("down_blocks.0.resnets.0.conv1", lambda: torch.nn.Conv2d(320, 320, 3, padding=1), (1, 320, 64, 64)),
    # Downsampler, 64 x 64 in, 32 x 32 out.
    "conv3x3_stride2": ("down_blocks.0.downsamplers.0.conv", lambda: torch.nn.Conv2d(320, 320, 3, stride=2, padding=1), (1, 320, 64, 64)),
}


# Write a tensor as raw float32 bytes, lowest byte first ("<f4"), with no header.
def write_f32(tensor, path):
    # Detach from autograd, move to numpy, and write.
    tensor.detach().float().numpy().astype("<f4").tofile(path)


# Make the four files of one case.
def make_case(name, layer_name, make_layer, input_shape):
    # Start every case from the same seed, so one case does not change another.
    torch.manual_seed(SEED)
    # PyTorch's normal random weights and bias for this layer kind.
    layer = make_layer()
    # Random input with values around -3..3.
    values = torch.randn(input_shape)
    # Compute the expected output in float64, then round to float32. This is the closest answer to the exact math,
    # so the test measures only the error of our GPU kernel.
    with torch.no_grad():
        expected = layer.double()(values.double())
    # The folder of this case.
    folder = OUT_DIR / name
    # Make it if it does not exist yet.
    folder.mkdir(exist_ok=True)
    # The input, in PyTorch's layout.
    write_f32(values, folder / "input.bin")
    # The weights, in PyTorch's layout: [out, in] or [out, in, k, k].
    write_f32(layer.weight, folder / "weight.bin")
    # The bias.
    write_f32(layer.bias, folder / "bias.bin")
    # The expected output.
    write_f32(expected, folder / "expected.bin")
    # Say what was written.
    print(f"{name} ({layer_name}): input {tuple(values.shape)} -> output {tuple(expected.shape)}")


# Make every case, only when this file runs as a script. make_packed_layers.py imports CASES without making them.
if __name__ == "__main__":
    # One folder per case.
    for case_name, (layer_name, make_layer, input_shape) in CASES.items():
        make_case(case_name, layer_name, make_layer, input_shape)
