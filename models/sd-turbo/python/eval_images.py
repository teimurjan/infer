# The shared parts of the 20-image tests: the prompts, the noise, the image loop, the scores and the grid.
# quant_eval.py rounds the UNet, and text_quant_eval.py rounds the text encoder. Both import this file.

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

# The Hugging Face name of the model.
MODEL_ID = "stabilityai/sd-turbo"
# The Mac GPU.
DEVICE = "mps"
# The UNet runs in float16, like the engine.
DTYPE = torch.float16
# The folder for the noise and the grid of quant_eval.py.
OUT_DIR = Path("quant/quant_eval")
# One noise per prompt, made once and then always read from disk.
NOISE_PATH = OUT_DIR / "noise.safetensors"

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
def save_grid(versions: dict[str, list[np.ndarray]], path: Path) -> None:
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
    grid.save(path)


# Load the pipeline in float16 on the Mac GPU, with the VAE in float32, ready for forward runs only.
def load_pipeline() -> StableDiffusionPipeline:
    # The whole pipeline in float16.
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
    # Only the VAE runs in float32.
    pipe.vae.to(torch.float32)
    # Hide the per-run progress bar.
    pipe.set_progress_bar_config(disable=True)
    # Turn off gradient tracking, because we only run forward.
    torch.set_grad_enabled(False)
    # Ready.
    return pipe


# Print PSNR and LPIPS of every version against the version named reference: one line per image, then the mean and the worst.
def print_scores(versions: dict[str, list[np.ndarray]], reference: str) -> None:
    # The default AlexNet LPIPS model, on the CPU. Lower LPIPS is closer, 0 is equal.
    judge = lpips.LPIPS(net="alex", verbose=False)
    # The versions that we compare with the reference.
    compared = [name for name in versions if name != reference]
    # The column names: two numbers per compared version.
    print(f"{'image':>5}  " + "  ".join(f"{name + ' PSNR':>17} {name + ' LPIPS':>17}" for name in compared) + "  prompt")
    # One score list per version and metric, for the summary lines.
    scores = {name: {"psnr": [], "lpips": []} for name in compared}
    # One line per image.
    for index, prompt in enumerate(PROMPTS):
        # The image of the reference version.
        target = versions[reference][index]
        # The numbers of this line.
        cells = []
        # Compare each version with the reference.
        for name in compared:
            # PSNR of this image.
            scores[name]["psnr"].append(psnr(versions[name][index], target))
            # LPIPS of this image.
            scores[name]["lpips"].append(judge(lpips_input(versions[name][index]), lpips_input(target)).item())
            # Format both numbers.
            cells.append(f"{scores[name]['psnr'][-1]:>17.2f} {scores[name]['lpips'][-1]:>17.4f}")
        # Print the line.
        print(f"{index:>5}  " + "  ".join(cells) + f"  {prompt}")
    # The mean, and the worst image, of each metric.
    for label, psnr_of, lpips_of in (("mean", np.mean, np.mean), ("worst", np.min, np.max)):
        # Worst PSNR is the lowest, worst LPIPS is the highest.
        print(f"{label:>5}  " + "  ".join(f"{psnr_of(scores[name]['psnr']):>17.2f} {lpips_of(scores[name]['lpips']):>17.4f}" for name in compared))
