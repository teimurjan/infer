# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "transformers==5.24.0", "accelerate", "safetensors", "huggingface_hub"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# The golden set of one short text exchange with LFM3-Omni-2B, from the official PyTorch model on the CPU in float32.
# It saves the prompt ids, the output of every layer of the language model for the prompt, the scores of the next token,
# and the greedy answer. The engine checks itself against these files later.
# Run it in the model folder: `uv run python/dump_text_golden.py`.

# json writes the manifest.
import json
# Path builds file paths and creates folders.
from pathlib import Path

# torch runs the model.
import torch
# safe_open reads the shapes of a weight file without loading the values. save_file writes .safetensors files.
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
# AutoModelForMultimodalLM loads LFM3-Omni with its own code. AutoTokenizer turns the chat into token ids.
from transformers import AutoModelForMultimodalLM, AutoTokenizer

# The Hugging Face name of the model.
MODEL_ID = "LiquidAI/LFM3-Omni-2B"
# The commit of the weights and of the model's own code. A pinned commit means a later upload cannot change either.
REVISION = "c01718a568e53ae936e6b5e1ff5ab2de7d7f388b"
# The folder of this golden set.
SET_DIR = Path("golden/text")
# The chat of the golden exchange. "Respond in text." asks for a text answer, as the model card says.
CHAT = [
    {"role": "system", "content": "Respond in text."},
    {"role": "user", "content": "What is the capital of France? Answer in one word."},
]
# The longest answer, in tokens. The answer stops earlier at <|im_end|>.
MAX_NEW_TOKENS = 16


# The parameters of each part of the model, by the first two parts of each parameter name, for the memory plan.
def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        # For example "model.language_model" or "model.depth_decoder".
        part = ".".join(name.split(".")[:2])
        counts[part] = counts.get(part, 0) + parameter.numel()
    return counts


# The parameters of the speech codec, which the processor loads from its own file, read from the file header only.
def codec_parameter_count() -> int:
    path = hf_hub_download(MODEL_ID, "audio_tokenizer/model.safetensors", revision=REVISION)
    with safe_open(path, framework="pt") as reader:
        # The product of each shape is the value count of that tensor.
        return sum(torch.Size(reader.get_slice(name).get_shape()).numel() for name in reader.keys())


# Save each tensor in its own file, with a manifest of names, shapes and dtypes, as the SD-Turbo golden sets do.
def save(records: dict[str, torch.Tensor]) -> None:
    SET_DIR.mkdir(parents=True, exist_ok=True)
    for name, tensor in records.items():
        save_file({name: tensor.contiguous()}, SET_DIR / f"{name}.safetensors")
    manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")} for name, tensor in records.items()]
    (SET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{SET_DIR}: {len(records)} tensors")


# Turn off gradient tracking, because we only run forward.
torch.set_grad_enabled(False)
# The tokenizer with the chat template of the checkpoint.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
# The official model with its own code, in float32 on the CPU, so the numbers do not depend on a GPU.
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32, device_map="cpu", trust_remote_code=True).eval()

# The size of each part, and of the codec.
for part, count in sorted(parameter_counts(model).items()):
    print(f"parameters {part}: {count}")
print(f"parameters audio_tokenizer (codec file): {codec_parameter_count()}")

# The prompt as text and as ids. add_generation_prompt opens the assistant turn.
prompt = tokenizer.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)
ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
print(f"prompt: {json.dumps(prompt)}")
print(f"prompt ids ({ids.shape[1]}): {ids[0].tolist()}")

# One pass of the language model over the prompt: the input of layer 0 and the output of each of the 24 layers.
language_model = model.model.language_model
outputs = language_model(input_ids=ids, output_hidden_states=True)
# The scores of the next token, from the last position of the prompt. lm_head shares its weights with the token table.
logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
next_id = int(logits[0, -1].argmax())
print(f"next token: {next_id} {json.dumps(tokenizer.decode([next_id]))}")

# The official greedy answer in text mode.
generated = model.generate(input_ids=ids, generation_mode="text", do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
sequences = generated.sequences if hasattr(generated, "sequences") else generated
answer_ids = sequences[0, ids.shape[1]:]
print(f"answer ids ({answer_ids.shape[0]}): {answer_ids.tolist()}")
print(f"answer: {json.dumps(tokenizer.decode(answer_ids))}")
# The first answer token must be the best score of the single pass above, or the two paths differ.
print(f"first answer token equals the best score of the prompt pass: {int(answer_ids[0]) == next_id}")

# The golden tensors. hidden_states[0] is the input of layer 0. hidden_states[i] is the output of layer i - 1.
records = {"prompt_ids": ids[0].to(torch.int64), "answer_ids": answer_ids.to(torch.int64)}
for index, hidden in enumerate(outputs.hidden_states):
    records[f"hidden_{index:02d}"] = hidden[0]
records["last_hidden_state"] = outputs.last_hidden_state[0]
records["next_logits"] = logits[0, -1]
save(records)
