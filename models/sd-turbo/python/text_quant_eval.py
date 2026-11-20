# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "lpips", "pillow", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# The 20-image test of the text encoder: float32 (what the engine runs now), float16, and 8-bit weights.
# Only the text encoder changes. Every version uses the same 8-bit UNet that the engine ships (quant_plan_8bit.json).
# Run it in the model folder: `uv run python/text_quant_eval.py`.

# json reads the UNet plan.
import json
# math.prod multiplies the numbers of a shape.
import math
# Path builds file paths.
from pathlib import Path

# torch runs the models.
import torch
# CLIPTextModel is the text encoder, and CLIPTokenizer turns a prompt into token ids.
from transformers import CLIPTextModel, CLIPTokenizer

# The shared parts of the 20-image tests.
from eval_images import DEVICE, DTYPE, MODEL_ID, PROMPTS, generate_images, load_or_make_noise, load_pipeline, print_scores, save_grid
# fake_quantize rounds one weight with one scale per output row, and quantizable_layers lists the UNet layers.
from quantize import fake_quantize, quantizable_layers

# The folder for the grid of this test.
OUT_DIR = Path("quant/text_quant_eval")
# The picture with one row per prompt and one column per text encoder version.
GRID_PATH = OUT_DIR / "grid.png"
# The UNet plan that the engine ships as unet_q8.
UNET_PLAN_PATH = "quant/quant_plan_8bit.json"
# Tokens per prompt, as in the engine.
TOKENS = 77
# Bytes of one float32 value, of one float16 scale, and of one 8-bit level.
FLOAT32_BYTES, SCALE_BYTES, LEVEL_BYTES = 4, 2, 1


# The text encoder weights that the 8-bit version rounds: every linear layer, and the token table.
# The token table is [49408 tokens][1024], so one scale per row is one scale per token.
# Layer norms, biases and the position table stay float32. They are small.
def rounded_weights(model: CLIPTextModel) -> list[tuple[str, torch.nn.Module]]:
    # Keep the name, so the report can count the bytes.
    return [(name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) or name.endswith("token_embedding")]


# Bytes of the 8-bit file: one level per rounded weight, one float16 scale per row, and float32 for everything else.
def eight_bit_bytes(model: CLIPTextModel) -> int:
    # The rounded tensors, by the name of their weight.
    rounded = {f"{name}.weight": module.weight for name, module in rounded_weights(model)}
    # Levels and scales of the rounded tensors.
    packed = sum(weight.numel() * LEVEL_BYTES + weight.shape[0] * SCALE_BYTES for weight in rounded.values())
    # Every other tensor stays float32.
    other = sum(math.prod(tensor.shape) * FLOAT32_BYTES for name, tensor in model.state_dict().items() if name not in rounded)
    # The whole file, without the small header.
    return packed + other


# The text encoder in float32 on the Mac GPU, from the published weights.
def float32_encoder() -> CLIPTextModel:
    # A fresh copy for every version, so the versions never stack.
    return CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", dtype=torch.float32).to(DEVICE).eval()


# The text embeddings of the 20 prompts, [20, 77, 1024] in float32, from `model`.
def encode(model: CLIPTextModel, tokenizer: CLIPTokenizer) -> torch.Tensor:
    # Pad every prompt to 77 tokens, as the pipeline and the engine do.
    ids = tokenizer(list(PROMPTS), padding="max_length", max_length=TOKENS, truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    # The last hidden state, after the final layer norm. It is what the UNet reads.
    return model(ids)[0].float()


# The three versions of the text encoder, each as its 20 embeddings.
def text_versions(tokenizer: CLIPTokenizer) -> tuple[dict[str, torch.Tensor], int, int]:
    # What the engine runs now.
    model = float32_encoder()
    # The bytes of the float32 file and of the 8-bit file.
    float32_size = sum(tensor.numel() * FLOAT32_BYTES for tensor in model.state_dict().values())
    eight_bit_size = eight_bit_bytes(model)
    # Version name to embeddings.
    versions = {"float32": encode(model, tokenizer)}
    # Every weight in float16. The math also runs in float16, as it would in a float16 engine.
    versions["float16"] = encode(model.half(), tokenizer)
    # A fresh float32 copy, with the linear layers and the token table rounded to 8 bits. The math stays float32.
    model = float32_encoder()
    for _, module in rounded_weights(model):
        module.weight.data.copy_(fake_quantize(module.weight.data, 8))
    versions["8-bit"] = encode(model, tokenizer)
    # The embeddings and the two sizes.
    return versions, float32_size, eight_bit_size


# Print how far each version's embeddings are from the float32 embeddings.
def print_embedding_errors(embeddings: dict[str, torch.Tensor]) -> None:
    # The reference.
    reference = embeddings["float32"]
    # The root mean square of the reference, so the error can be read as a fraction of it.
    reference_rms = reference.pow(2).mean().sqrt().item()
    for name, values in embeddings.items():
        # Skip the reference itself.
        if name == "float32":
            continue
        # The difference from float32.
        difference = values - reference
        # The largest difference, and the root mean square difference relative to the reference.
        print(f"{name} embeddings: largest difference {difference.abs().max().item():.4e}, rms difference {difference.pow(2).mean().sqrt().item() / reference_rms:.4e} of the float32 rms")


# The pipeline in float16 on the Mac GPU.
pipe = load_pipeline()
# The 8-bit UNet of the engine, in every version.
plan = json.load(open(UNET_PLAN_PATH))
for name, layer in quantizable_layers(pipe.unet):
    layer.weight.data.copy_(fake_quantize(layer.weight.data, plan[name]))
# The embeddings of each text encoder version, and the file sizes.
embeddings, float32_size, eight_bit_size = text_versions(pipe.tokenizer)
# The file sizes in MB. MB is 10^6 bytes, as in PROGRESS.md.
print(f"text encoder file: float32 {float32_size / 1e6:.1f} MB, float16 {float32_size / 2e6:.1f} MB, 8-bit {eight_bit_size / 1e6:.1f} MB")
# How far the embeddings move.
print_embedding_errors(embeddings)
# The saved noise of quant_eval.py, so the images match that test.
noise = load_or_make_noise()
# Version name to its 20 images. The UNet reads the embeddings in float16.
versions = {name: generate_images(pipe, noise, values.to(DTYPE)) for name, values in embeddings.items()}
# Compare with the float32 text encoder.
print_scores(versions, "float32")
# Write the grid.
OUT_DIR.mkdir(parents=True, exist_ok=True)
save_grid(versions, GRID_PATH)
# Name the file, so the user knows where the grid is.
print(f"wrote {GRID_PATH}")
