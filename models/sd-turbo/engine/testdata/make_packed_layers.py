# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# sys lets us import pack_weights.py from the python folder of the model, two levels up.
import sys
# Path builds the folder paths.
from pathlib import Path

# numpy writes the packed words and the float16 scales.
import numpy as np
# torch makes the layers and computes the expected outputs.
import torch

# The project folder holds pack_weights.py, the same packer that convert_weights.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
# The packed format: rounding, packing, and the weight the engine computes with.
from pack_weights import dequantize, pack_rows, quantize_rows  # noqa: E402
# The float cases, their seed and the f32 writer, so both scripts make the same layers and inputs.
from make_layers import CASES, OUT_DIR, SEED, write_f32  # noqa: E402

# The bit widths to test. The names in src/packed_cases.rs end in _q4 and _q8.
BITS = (4, 8)


# Make the five files of one packed case.
def make_packed_case(name: str, layer_name: str, make_layer, input_shape: tuple, bits: int) -> None:
    # Start from the same seed as the float case, so the layer and the input are the same as in testdata/<name>/.
    torch.manual_seed(SEED)
    # PyTorch's normal random weights and bias for this layer kind.
    layer = make_layer()
    # Random input with values around -3..3.
    values = torch.randn(input_shape)
    # Round the weights with the same code as the converter.
    levels, scale = quantize_rows(layer.weight.detach().numpy(), bits)
    # PyTorch runs the rounded weights: integer x float16 scale, in float64.
    with torch.no_grad():
        # Put the rounded weights into the layer.
        layer = layer.double()
        # The weight that the engine kernel computes with.
        layer.weight.copy_(torch.from_numpy(dequantize(levels, scale, tuple(layer.weight.shape))))
        # The closest answer to the exact math, so the test measures only the error of our GPU kernel.
        expected = layer(values.double())
    # For example testdata/linear_q4.
    folder = OUT_DIR / f"{name}_q{bits}"
    # Make it if it does not exist yet.
    folder.mkdir(exist_ok=True)
    # The input, in PyTorch's layout.
    write_f32(values, folder / "input.bin")
    # The packed words, 4 bytes each, lowest byte first.
    pack_rows(levels, bits).astype("<u4").tofile(folder / "weight.bin")
    # One float16 scale per output channel, 2 bytes each.
    scale.astype("<f2").tofile(folder / "scale.bin")
    # The bias stays f32.
    write_f32(layer.bias, folder / "bias.bin")
    # The expected output.
    write_f32(expected, folder / "expected.bin")
    # Say what was written.
    print(f"{folder.name} ({layer_name}): input {tuple(values.shape)} -> output {tuple(expected.shape)}, levels {levels.min()} to {levels.max()}")


# Every float case at every bit width.
for case_name, (layer_name, make_layer, input_shape) in CASES.items():
    # 4 bits, then 8 bits.
    for bits in BITS:
        # Write one folder.
        make_packed_case(case_name, layer_name, make_layer, input_shape, bits)
