# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "transformers==5.24.0", "accelerate", "safetensors", "huggingface_hub"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# The golden set of a word-by-word answer, from the official PyTorch model on the CPU in float32.
# The prompt is longer than 256 tokens, so the sliding attention layers must drop their oldest keys.
# It saves the prompt ids, the greedy answer, and the scores of every step: step 0 comes from the prompt,
# step i from answer token i - 1. The official generate must give the same answer, or the script stops.
# Run it in the model folder: `uv run python/dump_loop_golden.py`.

# json writes the manifest.
import json
# Path builds file paths and creates folders.
from pathlib import Path

# torch runs the model.
import torch
# save_file writes .safetensors files.
from safetensors.torch import save_file
# The official model with its own code, its token cache, and the tokenizer.
from transformers import AutoModelForMultimodalLM, AutoTokenizer, DynamicCache

# The checkpoint and its commit, the same as in dump_text_golden.py.
MODEL_ID = "LiquidAI/LFM3-Omni-2B"
REVISION = "c01718a568e53ae936e6b5e1ff5ab2de7d7f388b"
# The folder of this golden set.
SET_DIR = Path("golden/loop")
# The two tokens that end an answer (eos_token_id in generation_config.json): <|im_end|> and <|text_end|>.
STOP_IDS = {124900, 125014}
# The longest answer, in tokens.
MAX_NEW_TOKENS = 40
# A passage written for this test, long enough to pass 256 tokens with the question.
PASSAGE = (
    "Bread is one of the oldest foods that people prepare. A simple loaf needs only flour, water, salt and yeast. "
    "The baker mixes the flour with water and lets the dough rest, so that the flour can drink the water. "
    "Then the salt and the yeast go in, and the baker kneads the dough for several minutes. Kneading builds a "
    "network of gluten, a stretchy protein that will hold the gas that the yeast makes. The dough then rises in a "
    "warm place for one or two hours. During this time the yeast eats the sugars in the flour and breathes out "
    "carbon dioxide, which fills the dough with small bubbles. The baker shapes the loaf, lets it rise a second "
    "time, and cuts a few lines into its top, so that it can open in a controlled way in the heat. In a hot oven, "
    "the water in the dough turns to steam and the bubbles grow, which is called oven spring. The outside dries "
    "and browns into a crust, while the inside sets into a soft crumb. A good loaf sounds hollow when you tap its "
    "bottom. Many bakers say that time is the most important ingredient: a slow, cool rise gives the dough more "
    "flavor than a fast, warm one, because the yeast and the bacteria in the flour have longer to work. Some "
    "breads, like sourdough, use no bought yeast at all. They rise with a starter, a mix of flour and water that "
    "holds wild yeast and lactic acid bacteria, kept alive by feeding it fresh flour every day."
)
# The chat: a system turn for text answers, and the passage with one question.
CHAT = [
    {"role": "system", "content": "Respond in text."},
    {"role": "user", "content": PASSAGE + "\n\nIn two sentences, explain why a slow rise gives bread more flavor."},
]


# Save each tensor in its own file, with a manifest, as the other golden sets do.
def save(records: dict[str, torch.Tensor]) -> None:
    SET_DIR.mkdir(parents=True, exist_ok=True)
    for name, tensor in records.items():
        save_file({name: tensor.contiguous()}, SET_DIR / f"{name}.safetensors")
    manifest = [{"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")} for name, tensor in records.items()]
    (SET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{SET_DIR}: {len(records)} tensors")


# Turn off gradient tracking, because we only run forward.
torch.set_grad_enabled(False)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32, device_map="cpu", trust_remote_code=True).eval()
language_model = model.model.language_model

# The prompt ids.
prompt = tokenizer.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)
ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
print(f"prompt tokens: {ids.shape[1]}")

# The greedy loop: the whole prompt in one pass, then one token per pass, with the memory of earlier tokens.
cache = DynamicCache(config=language_model.config)
step_scores = []
answer = []
inputs = ids
for _ in range(MAX_NEW_TOKENS):
    hidden = language_model(input_ids=inputs, past_key_values=cache, use_cache=True).last_hidden_state
    # The scores of the next token, from the last row.
    scores = model.lm_head(hidden[:, -1, :])[0]
    step_scores.append(scores)
    token = int(scores.argmax())
    answer.append(token)
    if token in STOP_IDS:
        break
    inputs = torch.tensor([[token]])
print(f"answer ids ({len(answer)}): {answer}")
print(f"answer: {json.dumps(tokenizer.decode(answer))}")

# The official generate in text mode must give the same tokens.
generated = model.generate(input_ids=ids, generation_mode="text", do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
sequences = generated.sequences if hasattr(generated, "sequences") else generated
official = sequences[0, ids.shape[1]:].tolist()
print(f"official generate gives the same tokens: {official == answer}")
if official != answer:
    raise SystemExit(f"the official generate gave {official}")

save({"prompt_ids": ids[0].to(torch.int64), "answer_ids": torch.tensor(answer, dtype=torch.int64), "step_scores": torch.stack(step_scores)})
