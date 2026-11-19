# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "lpips", "pillow", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# json reads the mixed plan.
import json
# Path builds file paths and creates folders.
from pathlib import Path

# lpips measures how different two images look to a neural network trained to match human judgment.
import lpips
# numpy holds the 8-bit images.
import numpy as np
# torch runs the models.
import torch
# StableDiffusionPipeline wires the text encoder, UNet, scheduler and VAE together.
from diffusers import StableDiffusionPipeline
# Image builds the grid, and ImageDraw writes the column titles.
from PIL import Image, ImageDraw
# load_file and save_file read and write .safetensors files.
from safetensors.torch import load_file, save_file

# fake_quantize rounds one weight, and quantizable_layers lists the layers to round.
from quantize import fake_quantize, quantizable_layers

# The Hugging Face name of the model.
MODEL_ID = "stabilityai/sd-turbo"
# The Mac GPU.
DEVICE = "mps"
# The UNet and the text encoder run in float16, like the engine.
DTYPE = torch.float16
# The folder for the noise and the grid.
OUT_DIR = Path("quant/quant_eval")
# One noise per prompt, made once and then always read from disk.
NOISE_PATH = OUT_DIR / "noise.safetensors"
# The picture with one row per prompt and one column per version.
GRID_PATH = OUT_DIR / "grid.png"
# The 4-or-8 choice per layer from quant_plan.py.
PLAN_PATH = "quant/quant_plan.json"
# The 8-bit plan (`uv run quant_plan.py 900 --out quant_plan_8bit.json`): 281 of 282 layers at 8 bits.
PLAN_8BIT_PATH = "quant/quant_plan_8bit.json"
# The size of each image in pixels.
IMAGE_SIZE = 512
# The height of the title strip above the grid, in pixels.
TITLE_HEIGHT = 32
# The 20 fixed prompts. They cover animals, people, rooms, landscapes, objects, text and styles.
PROMPTS = (
    "a photo of a cat sitting on a wooden table",
    "a portrait of an old fisherman with a grey beard, natural light",
    "a red sports car parked on a wet city street at night",
    "a bowl of ramen with an egg and green onions, top view",
    "a snowy mountain lake at sunrise, reflections on the water",
    "a golden retriever puppy running on green grass",
    "a cozy living room with a fireplace and bookshelves",
    "a watercolor painting of a lighthouse on a cliff",
    "a close-up photo of a hummingbird drinking from a flower",
    "a busy market street in Marrakech with colorful spices",
    "a futuristic city skyline with flying cars, digital art",
    "a sign that says OPEN on a coffee shop door",
    "a pile of fresh oranges and lemons in a wooden crate",
    "an astronaut riding a horse on the moon",
    "a young woman reading a book in a sunny cafe",
    "a pencil sketch of an owl on a branch",
    "a tropical beach with palm trees and turquoise water",
    "a plate of chocolate chip cookies next to a glass of milk",
    "an old castle in the fog, oil painting",
    "a macro photo of dew drops on a spider web",
)


# Return one noise per prompt. Make it with seeds 0 to 19 and save it only on the very first run.
def load_or_make_noise() -> torch.Tensor:
    # Make the noise file only if it is not on disk yet.
    if not NOISE_PATH.exists():
        # The latent is 4 channels of 64 x 64 for a 512 x 512 image.
        shape = (1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8)
        # One CPU generator per image, so image i always gets seed i.
        noise = torch.cat([torch.randn(shape, generator=torch.Generator("cpu").manual_seed(seed)) for seed in range(len(PROMPTS))])
        # Create the output folder if it is missing.
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        # Write the noise to disk.
        save_file({"noise": noise}, NOISE_PATH)
    # Always read the noise from disk, so every run uses the same values.
    return load_file(NOISE_PATH)["noise"]


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


# Make one 8-bit RGB image per prompt with the UNet as it is now.
def generate_images(pipe: StableDiffusionPipeline, noise: torch.Tensor, text: torch.Tensor) -> list[np.ndarray]:
    # One image per prompt.
    images = []
    # Go through the prompts one at a time.
    for index in range(len(PROMPTS)):
        # One step, no guidance, this prompt and this noise, and stop before the VAE.
        latent = pipe(prompt_embeds=text[index : index + 1], num_inference_steps=1, guidance_scale=0.0,
                      latents=noise[index : index + 1].to(DEVICE, DTYPE), output_type="latent").images
        # Decode in float32, so the decoder adds no float16 error of its own.
        decoded = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor, return_dict=False)[0]
        # The pipeline turns the decoded tensor into a picture, the same way it saves PNGs.
        images.append(np.array(pipe.image_processor.postprocess(decoded, output_type="pil")[0]))
    # Give every image back.
    return images


# Peak signal-to-noise ratio in dB between two 8-bit images. Higher is closer. Equal images give infinity.
def psnr(image: np.ndarray, reference: np.ndarray) -> float:
    # The mean squared difference over every pixel and channel.
    mse = np.mean((image.astype(np.float64) - reference.astype(np.float64)) ** 2)
    # 255 is the largest 8-bit value.
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


# Turn an 8-bit image into the (1, 3, H, W) tensor in -1 to 1 that LPIPS reads.
def lpips_input(image: np.ndarray) -> torch.Tensor:
    # Height, width, channel becomes channel, height, width, then add the batch dim and rescale.
    return torch.from_numpy(image).permute(2, 0, 1)[None].float() / 127.5 - 1


# Put the versions side by side, one row per prompt, with the version names on top.
def save_grid(versions: dict[str, list[np.ndarray]]) -> None:
    # One white canvas with a column per version and a row per prompt.
    grid = Image.new("RGB", (IMAGE_SIZE * len(versions), TITLE_HEIGHT + IMAGE_SIZE * len(PROMPTS)), "white")
    # The pen for the titles.
    pen = ImageDraw.Draw(grid)
    # One column per version.
    for column, (name, images) in enumerate(versions.items()):
        # The title of this column.
        pen.text((column * IMAGE_SIZE + 8, 8), name, fill="black")
        # One row per prompt.
        for row, image in enumerate(images):
            # Paste the image into its cell.
            grid.paste(Image.fromarray(image), (column * IMAGE_SIZE, TITLE_HEIGHT + row * IMAGE_SIZE))
    # Write the grid.
    grid.save(GRID_PATH)


# Load the pipeline in float16 on the Mac GPU.
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
# Only the VAE runs in float32.
pipe.vae.to(torch.float32)
# Hide the per-run progress bar.
pipe.set_progress_bar_config(disable=True)
# Turn off gradient tracking, because we only run forward.
torch.set_grad_enabled(False)
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
# The default AlexNet LPIPS model, on the CPU. Lower LPIPS is closer, 0 is equal.
judge = lpips.LPIPS(net="alex", verbose=False)
# The versions that we compare with float16.
compared = [name for name in versions if name != "float16"]
# The column names: two numbers per compared version.
print(f"{'image':>5}  " + "  ".join(f"{name + ' PSNR':>17} {name + ' LPIPS':>17}" for name in compared) + "  prompt")
# One score list per version and metric, for the summary lines.
scores = {name: {"psnr": [], "lpips": []} for name in compared}
# One line per image.
for index, prompt in enumerate(PROMPTS):
    # The float16 image is the reference.
    reference = versions["float16"][index]
    # The numbers of this line.
    cells = []
    # Compare each version with float16.
    for name in compared:
        # PSNR of this image.
        scores[name]["psnr"].append(psnr(versions[name][index], reference))
        # LPIPS of this image.
        scores[name]["lpips"].append(judge(lpips_input(versions[name][index]), lpips_input(reference)).item())
        # Format both numbers.
        cells.append(f"{scores[name]['psnr'][-1]:>17.2f} {scores[name]['lpips'][-1]:>17.4f}")
    # Print the line.
    print(f"{index:>5}  " + "  ".join(cells) + f"  {prompt}")
# The mean, and the worst image, of each metric.
for label, psnr_of, lpips_of in (("mean", np.mean, np.mean), ("worst", np.min, np.max)):
    # Worst PSNR is the lowest, worst LPIPS is the highest.
    print(f"{label:>5}  " + "  ".join(f"{psnr_of(scores[name]['psnr']):>17.2f} {lpips_of(scores[name]['lpips']):>17.4f}" for name in compared))
# Write the grid.
save_grid(versions)
# Name the file, so the user knows where the grid is.
print(f"wrote {GRID_PATH}")
