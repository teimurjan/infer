# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "diffusers", "transformers", "accelerate", "safetensors"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# json writes the manifest file.
import json
# Path builds file paths and creates folders.
from pathlib import Path

# torch holds the tensors and runs the math on the CPU.
import torch
# StableDiffusionPipeline wires the tokenizer, text encoder, UNet, scheduler and VAE together.
from diffusers import StableDiffusionPipeline
# load_file and save_file read and write .safetensors files.
from safetensors.torch import load_file, save_file

# The Hugging Face name of the model we use as the reference.
MODEL_ID = "stabilityai/sd-turbo"
# The text that describes the image we want.
PROMPT = "a photo of a cat sitting on a wooden table"
# The seed for the noise. It is used only when the noise file does not exist yet.
SEED = 42
# The width and height of the final image, in pixels.
IMAGE_SIZE = 512
# SD-Turbo makes an image in one denoising step.
NUM_STEPS = 1
# 0.0 turns off classifier-free guidance, so the UNet runs on one batch item only.
GUIDANCE = 0.0
# The folder that holds every golden file.
GOLDEN_DIR = Path("golden")
# The one noise file that both sets read.
NOISE_PATH = GOLDEN_DIR / "starting_noise.safetensors"
# The name of the noise tensor inside its file.
NOISE_KEY = "starting_noise"
# The two precisions we dump, in this order. float32 must come first, because float16 loses detail.
DTYPES = (torch.float32, torch.float16)


# Return the starting noise. Make it and save it only on the very first run.
def load_or_make_noise(pipe: StableDiffusionPipeline) -> torch.Tensor:
    # Make the noise file only if it is not on disk yet.
    if not NOISE_PATH.exists():
        # The VAE shrinks each side of the image by this factor (8 for this model).
        latent_size = IMAGE_SIZE // pipe.vae_scale_factor
        # The UNet works on this many channels (4 for this model).
        latent_channels = pipe.unet.config.in_channels
        # A CPU random generator that starts from our fixed seed.
        generator = torch.Generator("cpu").manual_seed(SEED)
        # One image, 4 channels, 64 x 64 latent pixels, in float32.
        noise = torch.randn((1, latent_channels, latent_size, latent_size), generator=generator, dtype=torch.float32)
        # Create the golden folder if it is missing.
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        # Write the noise to disk.
        save_file({NOISE_KEY: noise}, NOISE_PATH)
    # Always read the noise from disk, so every run and every set uses the same values.
    return load_file(NOISE_PATH)[NOISE_KEY]


# Some blocks return a tensor, others return a tuple or an output object. Return the main tensor.
def main_tensor(output) -> torch.Tensor:
    # A plain tensor is already the answer.
    if isinstance(output, torch.Tensor):
        # Return it unchanged.
        return output
    # For tuples and output objects, item 0 is the main result (down blocks put skip tensors in item 1).
    return output[0]


# Put a tensor into the record under a name, and stop if that name is already there.
def store(records: dict[str, torch.Tensor], name: str, tensor: torch.Tensor) -> None:
    # A second write means a block ran twice, so the golden file would be wrong.
    if name in records:
        # Stop the script with a clear message.
        raise RuntimeError(f"{name} was recorded twice")
    # Copy the tensor, so later code cannot change our saved values. safetensors needs contiguous memory.
    records[name] = tensor.detach().contiguous().clone()


# Make a forward hook that stores the main output of a module under one name.
def recorder(records: dict[str, torch.Tensor], name: str):
    # PyTorch calls this function after the module finishes its forward pass.
    def hook(module, args, output) -> None:
        # Store the main tensor of this module.
        store(records, name, main_tensor(output))
    # Give the hook back to the caller.
    return hook


# Make a forward pre-hook that stores the UNet inputs: the scaled latent and the timestep.
def unet_input_recorder(records: dict[str, torch.Tensor]):
    # PyTorch calls this function before the UNet runs. The pipeline passes sample and timestep as the first two arguments.
    def hook(module, args) -> None:
        # The latent after the scheduler scaled it, (1, 4, 64, 64). This is what conv_in reads.
        store(records, "unet_input", args[0])
        # The timestep is a 0-d tensor. Shape (1,) and float32 keep it readable for the compare tool.
        store(records, "timestep", args[1].reshape(1).float())
    # Give the hook back to the caller.
    return hook


# Attach one recorder to each module we want to save, and return the handles so we can remove them.
def attach_recorders(pipe: StableDiffusionPipeline, records: dict[str, torch.Tensor]) -> list:
    # Each pair is (name in the golden set, module to watch).
    watched = [("text_encoder", pipe.text_encoder), ("time_embedding", pipe.unet.time_embedding)]
    # Add the first UNet convolution, which turns 4 latent channels into 320.
    watched += [("conv_in", pipe.unet.conv_in)]
    # Add every down block, numbered from 0.
    watched += [(f"down_block_{index}", block) for index, block in enumerate(pipe.unet.down_blocks)]
    # Add the mid block.
    watched += [("mid_block", pipe.unet.mid_block)]
    # Add every up block, numbered from 0.
    watched += [(f"up_block_{index}", block) for index, block in enumerate(pipe.unet.up_blocks)]
    # Add the whole UNet, whose main output is the predicted noise.
    watched += [("unet_output", pipe.unet)]
    # The UNet inputs need a hook that runs before the UNet, not after it.
    input_handle = pipe.unet.register_forward_pre_hook(unet_input_recorder(records))
    # Register the output hooks and keep every handle.
    return [input_handle] + [module.register_forward_hook(recorder(records, name)) for name, module in watched]


# Run text encoder, UNet and scheduler, then the VAE. Return the final latent and the decoded image.
def generate(pipe: StableDiffusionPipeline, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Run the pipeline, but stop before the VAE, so we can see the final latent.
    final_latent = pipe(
        # The text to draw.
        prompt=PROMPT,
        # Output height in pixels.
        height=IMAGE_SIZE,
        # Output width in pixels.
        width=IMAGE_SIZE,
        # One denoising step.
        num_inference_steps=NUM_STEPS,
        # No classifier-free guidance.
        guidance_scale=GUIDANCE,
        # Use our noise. The pipeline multiplies it by sigma (about 14.6) before the UNet sees it.
        latents=noise,
        # Return the latent instead of a decoded picture.
        output_type="latent",
    # The pipeline puts the latent in the `images` field when output_type is "latent".
    ).images
    # Turn off gradient tracking, because we only run the model forward.
    with torch.inference_mode():
        # Undo the latent scaling that SD uses, then decode the latent to RGB with the VAE.
        decoded_image = pipe.vae.decode(final_latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
    # Give both tensors back.
    return final_latent, decoded_image


# Make sure that the one-line Euler step gives the pipeline's latent. Only exact in float32.
def check_euler_step(pipe: StableDiffusionPipeline, noise: torch.Tensor, records: dict[str, torch.Tensor]) -> None:
    # Sigma at timestep 999, the only step (about 14.6).
    sigma = pipe.scheduler.sigmas[0]
    # One Euler step from sigma to 0: scale the noise up by sigma, then remove sigma times the predicted noise.
    manual_latent = sigma * (noise - records["unet_output"])
    # Stop the script if the one line and the pipeline disagree beyond float32 rounding.
    torch.testing.assert_close(manual_latent, records["final_latent"])


# Write each tensor to its own .safetensors file, plus manifest.json with name, shape and dtype.
def save_set(records: dict[str, torch.Tensor], set_dir: Path) -> None:
    # Create the folder for this set if it is missing.
    set_dir.mkdir(parents=True, exist_ok=True)
    # Write one file per tensor. The key inside the file is the same as the file name.
    for name, tensor in records.items():
        # For example golden/float32/mid_block.safetensors.
        save_file({name: tensor}, set_dir / f"{name}.safetensors")
    # One manifest entry per tensor, in the order the data flows through the model.
    manifest = [
        # str(torch.float32) is "torch.float32"; we keep only "float32".
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
        # Go through the tensors in the order they were recorded.
        for name, tensor in records.items()
    ]
    # Write the manifest as readable JSON.
    (set_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Print the manifest, one tensor per line, so the user can read it in the terminal.
    for entry in manifest:
        # Name, shape and dtype on one line.
        print(f"  {entry['name']:<16} shape={tuple(entry['shape'])} dtype={entry['dtype']}")


# Convert the pipeline to one dtype, run it once with hooks, and save every recorded tensor.
def dump_set(pipe: StableDiffusionPipeline, noise: torch.Tensor, dtype: torch.dtype) -> None:
    # Convert every weight to this dtype. The warning about float16 on CPU is expected, so we hide it.
    pipe.to(dtype=dtype, silence_dtype_warnings=True)
    # This dict collects the outputs of the watched modules, in the order they run.
    records: dict[str, torch.Tensor] = {}
    # Attach the recorders for this run.
    handles = attach_recorders(pipe, records)
    # Run the model with noise in the same dtype as the weights.
    final_latent, decoded_image = generate(pipe, noise.to(dtype))
    # Remove the recorders, so the next set starts clean.
    for handle in handles:
        # Detach this hook from its module.
        handle.remove()
    # Add the final latent after the UNet outputs.
    store(records, "final_latent", final_latent)
    # Add the decoded image last.
    store(records, "decoded_image", decoded_image)
    # The one-line Euler check matches only in float32. float16 rounds the scaled noise first.
    if dtype == torch.float32:
        # Stop the script if the check fails.
        check_euler_step(pipe, noise, records)
    # The folder name is the dtype name, for example golden/float16.
    set_dir = GOLDEN_DIR / str(dtype).removeprefix("torch.")
    # Tell the user which set is next.
    print(f"{set_dir}:")
    # Write the tensors and the manifest.
    save_set(records, set_dir)
    # Also save a PNG, so you can look at the picture of this set.
    pipe.image_processor.postprocess(decoded_image.float(), output_type="pil")[0].save(set_dir / "image.png")


# Download the model (first run only) and load every part in float32 on the CPU.
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, dtype=torch.float32).to("cpu")
# Read the noise from disk, or make it first if this is the first run.
noise = load_or_make_noise(pipe)
# Dump the float32 set, then the float16 set.
for dtype in DTYPES:
    # Run and save one set.
    dump_set(pipe, noise, dtype)
