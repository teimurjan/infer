# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "safetensors", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.
# Writes noise.safetensors: the starting noise of several seeds, exactly as diffusers makes it on the CPU:
# torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(seed)). The engine's noise test compares with it.

# Path finds the folder of this script.
from pathlib import Path
# torch makes the noise.
import torch
# save_file writes the tensors.
from safetensors.torch import save_file

# The seeds. 42 is the seed of golden/starting_noise.safetensors. The largest is the top of the 32-bit range.
SEEDS = [0, 1, 42, 1234, 4294967295]
# The latent shape of one 512 x 512 image.
SHAPE = (1, 4, 64, 64)


def main() -> None:
    # One tensor per seed, named seed_<seed>.
    tensors = {f"seed_{seed}": torch.randn(SHAPE, generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32) for seed in SEEDS}
    save_file(tensors, Path(__file__).parent / "noise.safetensors")
    # The first values of each seed, to read in the log.
    for name, tensor in tensors.items():
        print(f"{name}: {[round(value, 6) for value in tensor.flatten()[:4].tolist()]}")
    print(f"torch {torch.__version__}")


# Run main only when the file runs as a script.
if __name__ == "__main__":
    main()
