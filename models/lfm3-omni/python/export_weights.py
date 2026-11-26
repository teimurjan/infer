# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "safetensors", "huggingface_hub", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# Writes some tensors of the LFM3-Omni language model to one flat float32 file with a JSON index: weights/<stem>.bin
# and weights/<stem>.json. It is the format that packages/engine-core reads: each tensor starts on a multiple of 256 bytes.
# Run it in the model folder, for example `uv run python/export_weights.py layer0 layers.0.` for the weights of layer 0.

# argparse reads the stem and the name prefix from the command line.
import argparse
# Path builds file paths.
from pathlib import Path

# hf_hub_download finds the checkpoint in the local cache. safe_open reads one tensor at a time.
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# The writer of the flat weight format.
from weight_file import write_weight_file

# The checkpoint and its commit, the same as in dump_text_golden.py.
MODEL_ID = "LiquidAI/LFM3-Omni-2B"
REVISION = "c01718a568e53ae936e6b5e1ff5ab2de7d7f388b"
# Every weight of the language model starts with this. The file stores the names without it, so they stay short.
LANGUAGE_MODEL = "model.language_model."
# The folder of the weight files. Git ignores it.
OUT_DIR = Path("weights")


# Read the stem of the output file and the name prefix of the tensors to export.
def read_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export language model tensors of LFM3-Omni-2B as float32.")
    parser.add_argument("stem", help="the output name without extension, for example layer0")
    parser.add_argument("prefix", help="the tensor names to export, after 'model.language_model.', for example 'layers.0.'")
    return parser.parse_args()


# Each tensor of `names` as float32, one at a time, so the whole model is never in memory.
def float_tensors(reader, names: list[str]):
    for name in names:
        # The checkpoint is bfloat16. Every bfloat16 value is exact in float32.
        values = reader.get_tensor(name).float().numpy().astype("<f4")
        yield name.removeprefix(LANGUAGE_MODEL), values, list(values.shape), "float32"


arguments = read_arguments()
OUT_DIR.mkdir(exist_ok=True)
source = hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION)
with safe_open(source, framework="pt") as reader:
    # Sorted names give the same file on every run.
    names = sorted(name for name in reader.keys() if name.startswith(LANGUAGE_MODEL + arguments.prefix))
    if not names:
        raise SystemExit(f"no tensor starts with {LANGUAGE_MODEL + arguments.prefix}")
    size = write_weight_file(OUT_DIR / arguments.stem, float_tensors(reader, names))
print(f"{arguments.stem}: {len(names)} tensors, {size} bytes, float32")
