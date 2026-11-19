# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "diffusers", "accelerate", "safetensors", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# The golden set of the packed UNet: the PyTorch UNet with exactly the weights that the engine reads from weights/unet_packed.bin.
# A packed weight is integer x float16 scale. Every other tensor is the float16 value of the file. So the engine and this run
# differ only by float32 rounding, and the golden compare checks the packed kernels, not the rounding of the weights.
# It also runs the same model in float64 and prints the difference, which is the size of the float32 rounding.

# json reads the weight index and writes the manifest.
import json
# sys reads the name of the packed file from the command line.
import sys
# Path builds file paths.
from pathlib import Path

# numpy reads the weight file and unpacks the integers.
import numpy as np
# torch holds the tensors and runs the math on the CPU.
import torch
# UNet2DConditionModel is the SD-Turbo UNet. AutoencoderTiny is TAESD.
from diffusers import AutoencoderTiny, UNet2DConditionModel
# Image writes the PNG. Pillow comes with diffusers.
from PIL import Image
# load_file and save_file read and write .safetensors files.
from safetensors.torch import load_file, save_file

# The reader of the packed format. It uses the same unpacker as the check in convert_weights.py.
from pack_weights import read_packed_file

# The Hugging Face name of the model, as in dump_golden.py.
MODEL_ID = "stabilityai/sd-turbo"
# The packed weight file and its index.
# The packed file: `uv run dump_packed_golden.py` reads weights/unet_packed, `uv run dump_packed_golden.py unet_q8` reads weights/unet_q8.
STEM = sys.argv[1] if len(sys.argv) > 1 else "unet_packed"
PACKED_BIN = Path(f"weights/{STEM}.bin")
PACKED_INDEX = Path(f"weights/{STEM}.json")
# The float32 golden set, which has the UNet inputs.
INPUT_DIR = Path("golden/float32")
# The new golden set of the packed UNet.
# unet_packed keeps its old folder golden/packed. Another file gets golden/<stem>.
SET_DIR = Path("golden/packed" if STEM == "unet_packed" else f"golden/{STEM}")
# The modules whose output we save, and the golden name of each. The same names as the float32 set.
PARTS = {"time_embedding": "time_embedding", "conv_in": "conv_in", "down_block_0": "down_blocks.0", "down_block_1": "down_blocks.1", "down_block_2": "down_blocks.2", "down_block_3": "down_blocks.3", "mid_block": "mid_block", "up_block_0": "up_blocks.0", "up_block_1": "up_blocks.1", "up_block_2": "up_blocks.2", "up_block_3": "up_blocks.3"}


# Read one tensor from the float32 set. The key inside the file is the same as the file name.
def read_input(name: str) -> torch.Tensor:
    # For example golden/float32/unet_input.safetensors.
    return load_file(INPUT_DIR / f"{name}.safetensors")[name]


# Every tensor of the packed file as the float values that the engine computes with, keyed by its PyTorch name.
def packed_state(data: np.ndarray, index: list[dict]) -> dict[str, torch.Tensor]:
    # The reader of the packed format gives float64 numpy arrays. Make each one a torch tensor.
    return {name: torch.from_numpy(values) for name, values in read_packed_file(data, index).items()}


# Run the UNet in `dtype` with hooks on PARTS. Return the recorded tensors and the output.
def run(unet: UNet2DConditionModel, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    # The outputs of the hooked modules.
    records: dict[str, torch.Tensor] = {}
    # A hook that copies the main tensor of a module output.
    def recorder(name: str):
        def hook(module, args, output) -> None:
            # Blocks return a tuple whose item 0 is the main tensor.
            tensor = output if isinstance(output, torch.Tensor) else output[0]
            records[name] = tensor.detach().contiguous().clone()
        return hook
    # One hook per part.
    handles = [unet.get_submodule(module).register_forward_hook(recorder(name)) for name, module in PARTS.items()]
    # Turn off gradient tracking, because we only run the model forward.
    with torch.inference_mode():
        # The same three inputs as the float32 golden run.
        output = unet(read_input("unet_input").to(dtype), read_input("timestep")[0].to(dtype), encoder_hidden_states=read_input("text_encoder").to(dtype), return_dict=False)[0]
    # Remove the hooks.
    for handle in handles:
        handle.remove()
    # The output last.
    records["unet_output"] = output
    # Everything.
    return records


# Load the UNet, put the packed values into it, and run it.
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=torch.float32).to("cpu")
# The packed file and its index.
index = json.loads(PACKED_INDEX.read_text())
data = np.fromfile(PACKED_BIN, dtype=np.uint8)
# The float values of every tensor.
state = packed_state(data, index)
# Every parameter of the model must come from the file, and every tensor of the file must be a parameter.
missing, unexpected = set(unet.state_dict()) - set(state), set(state) - set(unet.state_dict())
if missing or unexpected:
    raise RuntimeError(f"missing {sorted(missing)[:3]}, unexpected {sorted(unexpected)[:3]}")
# The float32 run: the golden set.
unet.load_state_dict({name: value.float() for name, value in state.items()})
golden = run(unet, torch.float32)
# The float64 run: the size of the float32 rounding.
unet.to(torch.float64).load_state_dict(state)
reference = run(unet, torch.float64)
# Print the difference of each part of the UNet.
for name, tensor in reference.items():
    tensor = golden[name]
    print(f"{name}: float32 vs float64 largest difference {(tensor.double() - reference[name]).abs().max().item():.3e}, largest value {reference[name].abs().max().item():.3f}")
# The Euler step of the pipeline, with the same f32 operations as engine/src/pipeline/schedule.rs: from sigma to 0.
noise, sigma = load_file(Path("golden/starting_noise.safetensors"))["starting_noise"], read_input("sigma")[0]
sample = noise * sigma
original = sample - sigma * golden["unet_output"]
golden["final_latent"] = sample + (sample - original) / sigma * (0.0 - sigma)
# TAESD decodes the latent, as the decoder stage does. Its scaling_factor is 1.0.
taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).to("cpu")
with torch.inference_mode():
    golden["taesd_image"] = taesd.decode(golden["final_latent"] / taesd.config.scaling_factor, return_dict=False)[0].contiguous()
# The packed output against the float32 golden output: the effect of the rounding of the weights.
print(f"unet_output: packed vs float32 weights largest difference {(golden['unet_output'] - read_input('unet_output')).abs().max().item():.3e}")
# Write each tensor to its own file, and a manifest.
SET_DIR.mkdir(parents=True, exist_ok=True)
for name, tensor in golden.items():
    save_file({name: tensor}, SET_DIR / f"{name}.safetensors")
manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")} for name, tensor in golden.items()]
(SET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
# The image as a PNG, so the user can look at it: [-1, 1] to bytes, as the diffusers image processor does.
pixels = ((golden["taesd_image"][0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
Image.fromarray(pixels).save(SET_DIR / "taesd_image.png")
# Print the manifest.
print(f"{SET_DIR}:")
for entry in manifest:
    print(f"  {entry['name']:<16} shape={tuple(entry['shape'])} dtype={entry['dtype']}")
