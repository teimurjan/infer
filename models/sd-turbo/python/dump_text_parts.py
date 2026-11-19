# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "diffusers", "transformers", "accelerate", "safetensors"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# Save what the engine needs to run and check the text encoder and the scheduler step:
# the token ids of the golden prompt, the sigma of the one step, and two tensors inside the text encoder.
# It runs only the tokenizer, the text encoder and the scheduler, so it takes seconds.

# json writes the manifest file.
import json
# Path builds file paths.
from pathlib import Path

# torch holds the tensors and runs the math on the CPU.
import torch
# The scheduler of SD-Turbo.
from diffusers import EulerDiscreteScheduler
# The tokenizer and the text encoder of SD-Turbo.
from transformers import CLIPTextModel, CLIPTokenizer
# load_file and save_file read and write .safetensors files.
from safetensors.torch import load_file, save_file

# The Hugging Face name of the model, as in dump_golden.py.
MODEL_ID = "stabilityai/sd-turbo"
# The prompt of dump_golden.py.
PROMPT = "a photo of a cat sitting on a wooden table"
# SD-Turbo makes an image in one step, as in dump_golden.py.
NUM_STEPS = 1
# The float32 golden set that dump_golden.py wrote.
SET_DIR = Path("golden/float32")
# The manifest of these tensors. The main manifest.json stays unchanged.
MANIFEST_NAME = "text_parts_manifest.json"


# Read one tensor from the float32 set. The key inside the file is the same as the file name.
def read_golden(name: str) -> torch.Tensor:
    # For example golden/float32/text_encoder.safetensors.
    return load_file(SET_DIR / f"{name}.safetensors")[name]


# Tokenize the prompt as the pipeline does: pad to 77 tokens, cut longer prompts.
def token_ids(tokenizer: CLIPTokenizer) -> torch.Tensor:
    # The same call as StableDiffusionPipeline.encode_prompt.
    return tokenizer(PROMPT, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids


# Run the text encoder and return the embeddings, the output of layer 0, and the final output.
def encode(model: CLIPTextModel, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Turn off gradient tracking, because we only run the model forward.
    with torch.inference_mode():
        # hidden_states[0] is the input of layer 0, hidden_states[1] its output. Item 0 is last_hidden_state, after final_layer_norm.
        output = model(ids, output_hidden_states=True)
    # The three tensors.
    return output.hidden_states[0], output.hidden_states[1], output.last_hidden_state


# Stop if a value differs from the one that dump_golden.py saved. Then the new tensors would not belong to the old set.
def check_same(name: str, value: torch.Tensor) -> None:
    # The largest difference from the saved file.
    largest = (value.float() - read_golden(name)).abs().max().item()
    # Print it, so the log has the number.
    print(f"{name}: largest difference from the saved file {largest:e}")
    # The same weights, inputs and CPU code must give the same bits.
    if largest != 0.0:
        # Stop with the number.
        raise RuntimeError(f"this run differs from golden/float32/{name} by {largest:e}")


# Write each tensor to its own file, write the manifest, and print one line per tensor.
def save(records: dict[str, torch.Tensor]) -> None:
    # One file per tensor. The key inside the file is the same as the file name.
    for name, tensor in records.items():
        save_file({name: tensor.contiguous()}, SET_DIR / f"{name}.safetensors")
    # One entry per tensor: name, shape, dtype.
    manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")} for name, tensor in records.items()]
    # Write the manifest as readable JSON.
    (SET_DIR / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    # Print the manifest, one tensor per line.
    for entry in manifest:
        print(f"  {entry['name']:<20} shape={tuple(entry['shape'])} dtype={entry['dtype']}")


# Load the parts from the Hugging Face cache in float32 on the CPU.
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
model = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float32).to("cpu")
scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
# The token ids. The golden reader of the engine reads float32, and every id is below 2^24, so float32 keeps them exact.
ids = token_ids(tokenizer)
# Run the encoder.
embeddings, layer_0, output = encode(model, ids)
# The output must be the saved text_encoder tensor.
check_same("text_encoder", output)
# The scheduler for one step, as the pipeline sets it.
scheduler.set_timesteps(NUM_STEPS)
# The timestep must be the saved one.
check_same("timestep", scheduler.timesteps[:1].float())
# Tell the user which set gets the files.
print(f"{SET_DIR}:")
# Write the tensors and the manifest. sigma is the noise level of the one step.
save({"text_input_ids": ids.float(), "sigma": scheduler.sigmas[:1].float(), "text_embeddings": embeddings, "text_layer_0": layer_0})
# Print the sigma with all its digits.
print(f"sigma {scheduler.sigmas[0].item()!r}")
