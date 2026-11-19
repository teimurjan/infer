# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "diffusers", "accelerate", "safetensors"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# json writes the manifest file.
import json
# Path builds file paths.
from pathlib import Path

# torch holds the tensors and runs the math on the CPU.
import torch
# AutoencoderTiny is the diffusers version of TAESD, a small and fast replacement for the SD VAE.
from diffusers import AutoencoderTiny
# Image writes the PNG. Pillow comes with diffusers.
from PIL import Image
# load_file and save_file read and write .safetensors files.
from safetensors.torch import load_file, save_file

# The Hugging Face name of TAESD. It decodes the latents of SD 1.x and SD 2.x, so also of SD-Turbo.
MODEL_ID = "madebyollin/taesd"
# The float32 golden set. dump_golden.py wrote final_latent there.
SET_DIR = Path("golden/float32")
# The latent that the SD-Turbo pipeline gives to the VAE.
LATENT_NAME = "final_latent"
# The full SD VAE output of the same latent, to see how far TAESD is from it.
FULL_VAE_NAME = "decoded_image"
# The manifest of the TAESD tensors. The main manifest.json stays unchanged.
MANIFEST_NAME = "taesd_manifest.json"
# The layers of decoder.layers whose output we save, and the golden name of each.
# DecoderTiny builds: 0 conv, 1 ReLU, then per resolution 3 blocks, an upsample and a conv, then 1 block and the last conv.
CHECKPOINTS = {
    1: "taesd_conv_in",
    4: "taesd_blocks_0",
    6: "taesd_upsample_0",
    9: "taesd_blocks_1",
    11: "taesd_upsample_1",
    14: "taesd_blocks_2",
    16: "taesd_upsample_2",
    17: "taesd_blocks_3",
}
# The golden name of the decoder output, in [-1, 1], the same range as the full VAE output.
IMAGE_NAME = "taesd_image"
# The type of each checkpoint layer. A different diffusers version could number the layers differently.
EXPECTED_TYPES = {1: "ReLU", 4: "AutoencoderTinyBlock", 6: "Conv2d", 9: "AutoencoderTinyBlock", 11: "Conv2d", 14: "AutoencoderTinyBlock", 16: "Conv2d", 17: "AutoencoderTinyBlock"}


# Read one tensor from the float32 set. The key inside the file is the same as the file name.
def read_golden(name: str) -> torch.Tensor:
    # For example golden/float32/final_latent.safetensors.
    return load_file(SET_DIR / f"{name}.safetensors")[name]


# Stop if the decoder layers are not the ones that CHECKPOINTS was written for.
def check_layer_types(decoder: torch.nn.Module) -> None:
    # Look at every checkpoint layer.
    for index, expected in EXPECTED_TYPES.items():
        # The class name of the layer at this position.
        actual = type(decoder.layers[index]).__name__
        # A mismatch means the golden names would point at the wrong layers.
        if actual != expected:
            # Stop with both names.
            raise RuntimeError(f"decoder.layers[{index}] is {actual}, expected {expected}")


# Make a forward hook that copies the output of one layer into `records`.
def recorder(records: dict[str, torch.Tensor], name: str):
    # PyTorch calls this function after the layer finishes.
    def hook(module, args, output) -> None:
        # A second write means a layer ran twice, so the golden file would be wrong.
        if name in records:
            # Stop with a clear message.
            raise RuntimeError(f"{name} was recorded twice")
        # Copy the tensor, so later code cannot change it. safetensors needs contiguous memory.
        records[name] = output.detach().contiguous().clone()
    # Give the hook back to the caller.
    return hook


# Decode the latent with TAESD, with hooks on the checkpoint layers. Return every recorded tensor in data-flow order.
def decode(model: AutoencoderTiny, latent: torch.Tensor) -> dict[str, torch.Tensor]:
    # The outputs of the hooked layers, in the order they run.
    records: dict[str, torch.Tensor] = {}
    # One hook per checkpoint layer. Keep the handles, so we can remove the hooks.
    handles = [model.decoder.layers[index].register_forward_hook(recorder(records, name)) for index, name in CHECKPOINTS.items()]
    # Turn off gradient tracking, because we only run the model forward.
    with torch.inference_mode():
        # The pipeline divides by scaling_factor before the VAE. For TAESD it is 1.0, so the latent does not change.
        image = model.decode(latent / model.config.scaling_factor, return_dict=False)[0]
    # Remove the hooks.
    for handle in handles:
        # Detach this hook from its layer.
        handle.remove()
    # The decoder output comes last.
    records[IMAGE_NAME] = image.contiguous().clone()
    # Give everything back.
    return records


# Write each tensor to its own file, write the TAESD manifest, and print one line per tensor.
def save(records: dict[str, torch.Tensor]) -> None:
    # One file per tensor, for example golden/float32/taesd_conv_in.safetensors.
    for name, tensor in records.items():
        # The key inside the file is the same as the file name.
        save_file({name: tensor}, SET_DIR / f"{name}.safetensors")
    # One entry per tensor: name, shape, dtype.
    manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")} for name, tensor in records.items()]
    # Write the manifest as readable JSON.
    (SET_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    # Print the manifest, one tensor per line.
    for entry in manifest:
        # Name, shape and dtype on one line.
        print(f"  {entry['name']:<16} shape={tuple(entry['shape'])} dtype={entry['dtype']}")


# Save the image as a PNG, so the user can look at it.
def save_png(image: torch.Tensor, path: Path) -> None:
    # [-1, 1] to [0, 1], then clamp, as the diffusers image processor does.
    unit = (image[0] / 2 + 0.5).clamp(0, 1)
    # [3, 512, 512] to [512, 512, 3] bytes. round() matches the diffusers rounding.
    pixels = (unit.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
    # Write the PNG.
    Image.fromarray(pixels).save(path)


# Download TAESD (first run only) and load it in float32 on the CPU.
model = AutoencoderTiny.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to("cpu")
# Stop early if the layer numbers are not the expected ones.
check_layer_types(model.decoder)
# The latent from the float32 SD-Turbo run.
latent = read_golden(LATENT_NAME)
# Run the decoder with hooks.
records = decode(model, latent)
# Tell the user which set gets the files.
print(f"{SET_DIR}:")
# Write the tensors and the manifest.
save(records)
# Save the picture.
save_png(records[IMAGE_NAME], SET_DIR / "taesd_image.png")
# The full VAE image of the same latent.
full_vae = read_golden(FULL_VAE_NAME)
# How far TAESD is from the full VAE, on the [-1, 1] scale. TAESD is an approximation, so this is not 0.
difference = (records[IMAGE_NAME] - full_vae).abs()
# One line with the largest and the mean difference.
print(f"taesd_image vs decoded_image: largest difference {difference.max().item():.4f}, mean difference {difference.mean().item():.4f}")
