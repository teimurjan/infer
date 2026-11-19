# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "scipy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# argparse reads the size limit from the command line.
import argparse
# json reads the weight manifest and writes the plan.
import json
# math.prod multiplies the numbers of a shape.
import math

# numpy holds the per-layer numbers as arrays.
import numpy as np
# milp solves the yes-or-no choice per layer exactly. Bounds and LinearConstraint describe the limits.
from scipy.optimize import Bounds, LinearConstraint, milp

# The output of quant_sensitivity.py, with one table per bit width.
TABLE_PATH = "quant/quant_sensitivity.txt"
# The list of every UNet tensor, with its shape, from convert_weights.py.
MANIFEST_PATH = "weights/unet.json"
# The plan this script writes: layer name to 4 or 8.
PLAN_PATH = "quant/quant_plan.json"
# One scale per output channel, stored as float16.
SCALE_BYTES = 2
# Biases and norm weights stay float16.
FLOAT16_BYTES = 2


# Read one table of quant_sensitivity.txt and return layer name to (weight count, rms error).
def read_table(lines: list[str], bits: int) -> dict[str, tuple[int, float]]:
    # The table starts after its title line, for example "4-bit, one layer at a time, ...".
    start = next(index for index, line in enumerate(lines) if line.startswith(f"{bits}-bit")) + 2
    # The table rows: rank, layer, weights, rms, max. A blank line or the end of the file ends the table.
    rows = []
    # Go through the lines after the column names.
    for line in lines[start:]:
        # Split the row into its five columns.
        columns = line.split()
        # A row starts with its rank. Anything else ends the table.
        if len(columns) != 5 or not columns[0].isdigit():
            # Stop at the end of this table.
            break
        # Keep the layer name, the weight count and the rms error.
        rows.append((columns[1], int(columns[2]), float(columns[3])))
    # Give back a lookup by layer name.
    return {name: (count, rms) for name, count, rms in rows}


# Count the bytes that do not depend on the plan: the float16 scales, biases and norm weights.
def fixed_bytes(layer_names: set[str]) -> int:
    # Every UNet tensor, with its name and shape.
    tensors = json.load(open(MANIFEST_PATH))
    # The weights of the layers we round. Their size depends on the plan, so they are not counted here.
    rounded = {f"{name}.weight" for name in layer_names}
    # One float16 scale per output channel. Dim 0 of each rounded weight is its output channel count.
    scales = sum(tensor["shape"][0] for tensor in tensors if tensor["name"] in rounded) * SCALE_BYTES
    # Every other tensor stays float16.
    others = sum(math.prod(tensor["shape"]) for tensor in tensors if tensor["name"] not in rounded) * FLOAT16_BYTES
    # Both parts together.
    return scales + others


# The size limit in MB (10^6 bytes) comes from the command line.
parser = argparse.ArgumentParser(description="Pick 4 or 8 bits per UNet layer under a size limit.")
# For example `uv run quant_plan.py 700`.
parser.add_argument("limit_mb", type=float, help="the most the UNet weights may take, in MB (10^6 bytes)")
# The plan file. A second plan can get its own name, so the plan of the engine stays.
parser.add_argument("--out", default=PLAN_PATH, help=f"the plan file to write (default {PLAN_PATH})")
# Read the arguments.
arguments = parser.parse_args()
limit_bytes = int(arguments.limit_mb * 1e6)
# Every line of the sensitivity output.
lines = open(TABLE_PATH).read().splitlines()
# Layer name to (weight count, rms error), for each bit width.
four, eight = read_table(lines, 4), read_table(lines, 8)
# The layers in a fixed order, so that index i means the same layer in every array.
names = sorted(four)
# The weight count of each layer.
counts = np.array([four[name][0] for name in names])
# Independent errors add in squares, so the squared rms is the cost we add up.
cost_four = np.array([four[name][1] for name in names]) ** 2
# The same cost if the layer is 8-bit.
cost_eight = np.array([eight[name][1] for name in names]) ** 2
# A 4-bit weight takes half a byte, an 8-bit weight one byte. Everything starts at 4 bits.
base_bytes = fixed_bytes(set(names)) + int(counts.sum()) // 2
# Each layer moved to 8 bits adds half a byte per weight.
extra_bytes = counts / 2
# Choice per layer: 1 means 8 bits. Minimize the total cost: the 4-bit total plus the change of each moved layer.
result = milp(
    # The change in cost when a layer moves from 4 to 8 bits.
    c=cost_eight - cost_four,
    # The added bytes must fit in what is left of the limit after the all-4-bit size.
    constraints=LinearConstraint(extra_bytes[None, :], ub=limit_bytes - base_bytes),
    # Every choice is a whole number.
    integrality=np.ones(len(names)),
    # Every choice is 0 or 1.
    bounds=Bounds(0, 1),
)
# Stop with a clear message if the limit is below the all-4-bit size.
if not result.success:
    # The solver message names the problem.
    raise SystemExit(f"no plan fits {limit_bytes} bytes: {result.message}")
# 1 means 8 bits, 0 means 4 bits. Rounding removes tiny solver noise like 0.9999999.
eight_bit = np.round(result.x).astype(bool)
# The bytes of the chosen plan.
used_bytes = base_bytes + int(extra_bytes[eight_bit].sum())
# Write the plan as layer name to bit width.
json.dump({name: 8 if chosen else 4 for name, chosen in zip(names, eight_bit)}, open(arguments.out, "w"), indent=2)
# The limit and the size of the plan.
print(f"limit {limit_bytes} bytes, plan {used_bytes} bytes, all 4-bit {base_bytes} bytes, all 8-bit {base_bytes + int(extra_bytes.sum())} bytes")
# How much of the model the plan keeps at 8 bits.
print(f"8-bit layers {eight_bit.sum()} of {len(names)}, 8-bit weights {counts[eight_bit].sum()} of {counts.sum()}")
# The predicted rms error of the final latent: the square root of the summed squared rms.
print(f"predicted rms: all 4-bit {np.sqrt(cost_four.sum()):.3e}, plan {np.sqrt(np.where(eight_bit, cost_eight, cost_four).sum()):.3e}, all 8-bit {np.sqrt(cost_eight.sum()):.3e}")
# Name the file, so the user knows where the plan is.
print(f"wrote {arguments.out}")
