# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "diffusers", "accelerate", "safetensors"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# Save the outputs of parts inside the UNet, which dump_golden.py does not save.
# It runs only the UNet, on the inputs that dump_golden.py saved, so it takes minutes, not an hour.

# json writes the manifest file.
import json
# Path builds file paths.
from pathlib import Path

# torch holds the tensors and runs the math on the CPU.
import torch
# UNet2DConditionModel is the SD-Turbo UNet.
from diffusers import UNet2DConditionModel
# load_file and save_file read and write .safetensors files.
from safetensors.torch import load_file, save_file

# The Hugging Face name of the model, as in dump_golden.py.
MODEL_ID = "stabilityai/sd-turbo"
# The float32 golden set that dump_golden.py wrote.
SET_DIR = Path("golden/float32")
# The manifest of these tensors. The main manifest.json stays unchanged.
MANIFEST_NAME = "unet_parts_manifest.json"
# The first transformer, and the block inside it.
TRANSFORMER = "down_blocks.0.attentions.0"
BLOCK = f"{TRANSFORMER}.transformer_blocks.0"
# The modules whose output we save, in run order. The golden name is the module name, for example "down_blocks.0.resnets.0".
# attn1, attn2 and ff are saved before their residual add, because that is what each module returns.
PARTS = ["down_blocks.0.resnets.0", f"{TRANSFORMER}.proj_in", f"{BLOCK}.attn1", f"{BLOCK}.attn2", f"{BLOCK}.ff", TRANSFORMER, "down_blocks.0.resnets.1", "down_blocks.0.attentions.1"]


# Read one tensor from the float32 set. The key inside the file is the same as the file name.
def read_golden(name: str) -> torch.Tensor:
    # For example golden/float32/unet_input.safetensors.
    return load_file(SET_DIR / f"{name}.safetensors")[name]


# Make a forward hook that copies the output of one module into `records`.
def recorder(records: dict[str, torch.Tensor], name: str):
    # PyTorch calls this function after the module finishes.
    def hook(module, args, output) -> None:
        # A second write means a module ran twice, so the golden file would be wrong.
        if name in records:
            # Stop with a clear message.
            raise RuntimeError(f"{name} was recorded twice")
        # A transformer returns a tuple whose item 0 is the tensor. Other modules return the tensor itself.
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        # Copy the tensor, so later code cannot change it. safetensors needs contiguous memory.
        records[name] = tensor.detach().contiguous().clone()
    # Give the hook back to the caller.
    return hook


# Run the UNet once on the golden inputs, with hooks on PARTS. Return the recorded tensors and the UNet output.
def run(unet: UNet2DConditionModel) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    # The outputs of the hooked modules, in the order they run.
    records: dict[str, torch.Tensor] = {}
    # One hook per part. Keep the handles, so we can remove the hooks.
    handles = [unet.get_submodule(name).register_forward_hook(recorder(records, name)) for name in PARTS]
    # Turn off gradient tracking, because we only run the model forward.
    with torch.inference_mode():
        # The same three inputs that the pipeline gave the UNet. The timestep was saved as (1,), the pipeline passed a 0-d tensor.
        output = unet(read_golden("unet_input"), read_golden("timestep")[0], encoder_hidden_states=read_golden("text_encoder"), return_dict=False)[0]
    # Remove the hooks.
    for handle in handles:
        # Detach this hook from its module.
        handle.remove()
    # Everything that ran.
    return records, output


# Stop if this UNet run does not give the saved UNet output. Then the new tensors would not belong to the old set.
def check_same_run(output: torch.Tensor) -> None:
    # The output that dump_golden.py saved from the full pipeline.
    expected = read_golden("unet_output")
    # The largest difference.
    largest = (output - expected).abs().max().item()
    # Print it, so the log has the number.
    print(f"unet_output: largest difference from the saved file {largest:e}")
    # The same weights, inputs and CPU code must give the same bits.
    if largest != 0.0:
        # Stop with the number.
        raise RuntimeError(f"this run differs from golden/float32/unet_output by {largest:e}")


# Write each tensor to its own file, write the manifest, and print one line per tensor.
def save(records: dict[str, torch.Tensor]) -> None:
    # One file per tensor, for example golden/float32/down_blocks.0.resnets.0.safetensors.
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
        print(f"  {entry['name']:<24} shape={tuple(entry['shape'])} dtype={entry['dtype']}")


# Download the UNet (first run only, it is in the Hugging Face cache already) and load it in float32 on the CPU.
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=torch.float32).to("cpu")
# Run it with hooks.
records, output = run(unet)
# Stop if the run is not the one of the golden set.
check_same_run(output)
# Tell the user which set gets the files.
print(f"{SET_DIR}:")
# Write the tensors and the manifest.
save(records)
