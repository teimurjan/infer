# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "numpy", "safetensors"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# Test data for the packed linear kernel of engine-core with one float16 scale per 32 values, the 4-bit format of LFM3.
# It rounds two real weights of layer 0 with group_quant.py, writes them as a small packed weight file, and computes
# the expected output of random inputs in float64 with the rounded weights. tests/grouped_linear.rs reads the files.
# conv.in_proj has rows near 4e-36, whose scale is 0. feed_forward.w2 reads rows of 7680 values, so 240 scales per row.
# Run it in the model folder, after `uv run python/export_weights.py layer0 layers.0.`:
# `uv run engine/testdata/make_grouped_linear.py`.

# json reads the index of the layer 0 weights. sys lets us import the modules of the python folder.
import json
import sys
# Path builds the file paths.
from pathlib import Path

# numpy reads the float32 weights from the file.
import numpy as np
# torch makes the inputs and computes the expected outputs.
import torch
# save_file writes the inputs and the expected outputs.
from safetensors.torch import save_file

# The python folder of the model, two levels up, holds the weight format and the file writer.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from group_quant import dequantize_groups, pack_rows, quantize_groups  # noqa: E402
from weight_file import write_weight_file  # noqa: E402

# The float32 weights of layer 0, from export_weights.py.
SOURCE = Path("weights/layer0")
# The folder of this test data. Git ignores its .bin and .safetensors files.
OUT_DIR = Path("engine/testdata/grouped_linear")
# The two weights of the test, by their name in the weight file.
WEIGHTS = ("layers.0.conv.in_proj.weight", "layers.0.feed_forward.w2.weight")
# One row, as a word step runs it, and 9 rows, which is more than the 8 rows of one invocation and not a multiple of 8.
ROW_COUNTS = (1, 9)
# The format: 4 bits, one scale per 32 values.
BITS, GROUP = 4, 32
# The same random inputs on every run.
SEED = 0


# The float32 tensor `name` of the weight file `stem`.
def read_float32(stem: Path, name: str) -> torch.Tensor:
    entry = next(entry for entry in json.loads(stem.with_suffix(".json").read_text()) if entry["name"] == name)
    values = np.fromfile(stem.with_suffix(".bin"), dtype="<f4", count=int(np.prod(entry["shape"])), offset=entry["offset"])
    return torch.from_numpy(values.reshape(entry["shape"]))


OUT_DIR.mkdir(exist_ok=True)
torch.manual_seed(SEED)
packed = []
for name in WEIGHTS:
    weight = read_float32(SOURCE, name)
    levels, scale = quantize_groups(weight, BITS, GROUP)
    # The packed words and the scales, as the engine-core loader reads them: "<name>" and "<name>_scale".
    packed.append((name, pack_rows(levels, BITS).astype("<u4"), list(weight.shape), f"int{BITS}"))
    packed.append((f"{name}_scale", scale.numpy().astype("<f2"), list(scale.shape), "float16"))
    # The weight that the kernel computes with, in float64, so the expected output has almost no rounding of its own.
    rounded = dequantize_groups(levels, scale).double()
    for rows in ROW_COUNTS:
        # Random input with values around -3..3, as a hidden state after a norm.
        values = torch.randn(rows, weight.shape[1])
        expected = (values.double() @ rounded.T).float()
        case = f"{name.removeprefix('layers.0.').removesuffix('.weight')}_{rows}"
        save_file({"input": values, "expected": expected}, OUT_DIR / f"{case}.safetensors")
        print(f"{case}: input {tuple(values.shape)} -> output {tuple(expected.shape)}, largest value {expected.abs().max():.4f}")
    print(f"{name}: {scale.shape[1]} scales per row, {int((scale == 0).sum())} scales of 0")
size = write_weight_file(OUT_DIR / "weights", packed)
print(f"{OUT_DIR / 'weights.bin'}: {len(packed)} tensors, {size} bytes")
