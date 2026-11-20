# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "safetensors", "huggingface_hub"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# Writes the 8-bit plan of the text encoder: every matrix weight at 8 bits, as text_quant_eval.py tested it.
# The matrices are the linear layers and the token table. The position table, the layer norms and the biases stay float.
# Run it in the model folder: `uv run python/text_plan.py`. Then: `uv run python/convert_weights.py text_encoder --plan quant/text_plan_8bit.json --stem text_encoder_q8`.

# json writes the plan.
import json
# Path builds the output path.
from pathlib import Path

# hf_hub_download finds the published file in the local cache.
from huggingface_hub import hf_hub_download
# safe_open reads the shapes without loading the values.
from safetensors import safe_open

# The plan this script writes: layer name to 8.
PLAN_PATH = Path("quant/text_plan_8bit.json")
# The position table is [77][1024]: small, and added to every token, so it stays float.
KEEP_FLOAT = "text_model.embeddings.position_embedding.weight"

# The text encoder of SD-Turbo, as convert_weights.py reads it.
source = hf_hub_download("stabilityai/sd-turbo", "text_encoder/model.safetensors")
with safe_open(source, framework="np") as reader:
    # Every weight with two dimensions is a matrix: [out, in] for a linear layer, [tokens, 1024] for the token table.
    layers = sorted(name.removesuffix(".weight") for name in reader.keys() if name.endswith(".weight") and len(reader.get_slice(name).get_shape()) == 2 and name != KEEP_FLOAT)
# One line per layer, so the plan is easy to read.
PLAN_PATH.write_text(json.dumps({layer: 8 for layer in layers}, indent=1) + "\n")
# How many layers, and the first and last, so the run shows what it planned.
print(f"{PLAN_PATH}: {len(layers)} layers at 8 bits, from {layers[0]} to {layers[-1]}")
