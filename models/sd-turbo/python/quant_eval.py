# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "lpips", "pillow", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# json reads the mixed plan.
import json

# torch holds the weights.
import torch
# StableDiffusionPipeline wires the text encoder, UNet, scheduler and VAE together.
from diffusers import StableDiffusionPipeline

# The shared parts of the 20-image tests.
from eval_images import DEVICE, OUT_DIR, PROMPTS, generate_images, load_or_make_noise, load_pipeline, print_scores, save_grid
# fake_quantize rounds one weight, and quantizable_layers lists the layers to round.
from quantize import fake_quantize, quantizable_layers

# The picture with one row per prompt and one column per version.
GRID_PATH = OUT_DIR / "grid.png"
# The 4-or-8 choice per layer from quant_plan.py.
PLAN_PATH = "quant/quant_plan.json"
# The 8-bit plan (`uv run quant_plan.py 900 --out quant_plan_8bit.json`): 281 of 282 layers at 8 bits.
PLAN_8BIT_PATH = "quant/quant_plan_8bit.json"


# Run the text encoder once for all prompts. The text encoder is not rounded, so all versions share this.
def encode_prompts(pipe: StableDiffusionPipeline) -> torch.Tensor:
    # encode_prompt returns the prompt embeddings first. No guidance, so no negative prompt.
    return pipe.encode_prompt(list(PROMPTS), DEVICE, 1, do_classifier_free_guidance=False)[0]


# Set every layer to the bit width in `plan`. None means the float16 weight.
def apply_plan(layers, originals: dict[str, torch.Tensor], plan: dict[str, int] | None) -> None:
    # Go through every rounded layer.
    for name, layer in layers:
        # Start from the float16 weight, so plans never stack.
        weight = originals[name]
        # Float16 keeps the weight, any other plan rounds it.
        layer.weight.data.copy_(weight if plan is None else fake_quantize(weight, plan[name]))


# The pipeline in float16 on the Mac GPU.
pipe = load_pipeline()
# Every UNet layer with a weight matrix.
layers = quantizable_layers(pipe.unet)
# Keep each float16 weight, so every version starts from it.
originals = {name: layer.weight.data.clone() for name, layer in layers}
# The four versions: no rounding, every layer at 4 bits, the mixed plan, and the 8-bit plan.
plans = {"float16": None, "all 4-bit": {name: 4 for name, _ in layers}, "mixed plan": json.load(open(PLAN_PATH)), "8-bit plan": json.load(open(PLAN_8BIT_PATH))}
# The saved noise and the shared text embeddings.
noise, text = load_or_make_noise(), encode_prompts(pipe)
# Version name to its 20 images.
versions = {}
# Make the images of each version.
for name, plan in plans.items():
    # Set the UNet weights for this version.
    apply_plan(layers, originals, plan)
    # Make and keep the 20 images.
    versions[name] = generate_images(pipe, noise, text)
# Compare every version with float16.
print_scores(versions, "float16")
# Write the grid.
save_grid(versions, GRID_PATH)
# Name the file, so the user knows where the grid is.
print(f"wrote {GRID_PATH}")
