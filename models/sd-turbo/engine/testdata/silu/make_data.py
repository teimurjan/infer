# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# Path finds the folder of this script.
from pathlib import Path

# torch makes the input and computes the expected SiLU output on the CPU.
import torch

# The files go next to this script.
OUT_DIR = Path(__file__).parent
# A fixed seed, so every run makes the same input.
SEED = 0
# Not a multiple of 64, so the last workgroup is only partly full. This tests the bounds check.
COUNT = 100_003
# Random values are normal(0, 1) times this. Most values land in -12..12.
SPREAD = 4.0
# Values that we always want to test: zero, both signs, and large values.
# 80 is the largest we test, because e^x overflows f32 a little above 88.
EDGES = [0.0, -0.0, 1.0, -1.0, 10.0, -10.0, 80.0, -80.0]

# A random number generator that starts from SEED.
generator = torch.Generator().manual_seed(SEED)
# The random part of the input.
random_values = torch.randn(COUNT - len(EDGES), generator=generator) * SPREAD
# The full input: the edge values first, then the random values, all float32.
values = torch.cat([torch.tensor(EDGES), random_values]).to(torch.float32)
# The reference result: PyTorch's own SiLU in float32, the same function the UNet uses.
expected = torch.nn.functional.silu(values)

# Write raw float32 bytes, lowest byte first ("<f4"), with no header. The file size is COUNT * 4.
values.numpy().astype("<f4").tofile(OUT_DIR / "input.bin")
# Same format for the expected output.
expected.numpy().astype("<f4").tofile(OUT_DIR / "expected.bin")
# Say what was written, so the user can check the count.
print(f"wrote {COUNT} values to {OUT_DIR / 'input.bin'} and {OUT_DIR / 'expected.bin'}")
