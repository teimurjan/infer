# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "transformers==5.24.0", "accelerate", "safetensors", "huggingface_hub"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# How much each weight format changes the language model of LFM3-Omni-2B, before the engine gets packed kernels.
# Each format rounds every linear weight of the 24 layers and the token table, then puts the rounded values back as float32.
# The token table is also the output layer, so rounding it changes both ends of the model.
# For each format it prints the size of the rounded weights, how far its scores are from float32 over the bread exchange
# of golden/loop, and whether the greedy answers of golden/text and golden/loop stay the same.
# Run it in the model folder, after dump_text_golden.py and dump_loop_golden.py: `uv run python/quant_eval.py`.

# dataclass makes the small record of one format.
from dataclasses import dataclass

# torch runs the model.
import torch
# load_file reads the golden token ids.
from safetensors.torch import load_file
# The official model with its own code, and the tokenizer that prints an answer as text.
from transformers import AutoModelForMultimodalLM, AutoTokenizer

# The rounding of the weight format.
from group_quant import fake_quantize

# The checkpoint and its commit, the same as in dump_text_golden.py.
MODEL_ID = "LiquidAI/LFM3-Omni-2B"
REVISION = "c01718a568e53ae936e6b5e1ff5ab2de7d7f388b"
# The two golden exchanges whose prompts and answers this script reuses.
GOLDEN_SETS = ("golden/text", "golden/loop")
# The longest answer, in tokens, as in dump_loop_golden.py.
MAX_NEW_TOKENS = 40
# Bytes of one float16 scale.
SCALE_BYTES = 2
# Bytes in one megabyte, as the other logs count them.
MEGABYTE = 1_000_000


# One weight format: the bits of each integer, and how many neighbor values of a row share one float16 scale.
@dataclass(frozen=True)
class Format:
    # The name in the printed table.
    name: str
    # 4 or 8. None keeps float32.
    bits: int | None
    # Values per scale. None means one scale for the whole row, as the packed kernels of engine-core read today.
    group: int | None


# The formats to compare. Float32 is the reference.
FORMATS = (
    Format("float32", None, None),
    Format("8-bit, one scale per row", 8, None),
    Format("4-bit, one scale per row", 4, None),
    Format("4-bit, one scale per 128", 4, 128),
    Format("4-bit, one scale per 64", 4, 64),
    Format("4-bit, one scale per 32", 4, 32),
)


# The names of the weights that a format rounds: every 2-D weight of the language model. The 1-D norms and the
# convolution taps [2048, 1, 7] stay float32, because they are small.
def rounded_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.model.language_model.named_parameters() if parameter.dim() == 2]


# The bytes that the rounded weights take in a format: the integers and the float16 scales, or float32.
def format_bytes(shapes: list[torch.Size], form: Format) -> int:
    total = 0
    for shape in shapes:
        values = shape.numel()
        if form.bits is None:
            # 4 bytes per float32 value.
            total += values * 4
        else:
            # The integers, then one scale per group, or per row.
            total += values * form.bits // 8 + values // (form.group or shape[1]) * SCALE_BYTES
    return total


# The scores of the next token at every position of `ids`, in one pass. Row i scores the token after token i.
def all_scores(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    hidden = model.model.language_model(input_ids=ids[None]).last_hidden_state
    return model.lm_head(hidden)[0]


# The greedy answer to `prompt`, with the official generate in text mode, as dump_loop_golden.py checks it.
def greedy_answer(model: torch.nn.Module, prompt: torch.Tensor) -> list[int]:
    generated = model.generate(input_ids=prompt[None], generation_mode="text", do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    return sequences[0, prompt.shape[0]:].tolist()


# Turn off gradient tracking, because we only run forward.
torch.set_grad_enabled(False)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32, device_map="cpu", trust_remote_code=True).eval()
parameters = dict(model.model.language_model.named_parameters())
names = rounded_names(model)
# The output layer must be the token table itself, or rounding the table would leave the output layer in float32.
print(f"output layer shares the token table: {model.lm_head.weight.data_ptr() == parameters['embed_tokens.weight'].data_ptr()}")
# The original weights, kept so that each format rounds them and not the last format's result. bfloat16 is the checkpoint
# dtype, so this copy loses nothing and takes half the memory.
originals = {name: parameters[name].detach().to(torch.bfloat16).clone() for name in names}
print(f"rounded weights: {len(names)} tensors, {sum(weight.numel() for weight in originals.values())} values")

# The prompt and the golden answer of each exchange.
exchanges = [load_file(f"{folder}/prompt_ids.safetensors")["prompt_ids"] for folder in GOLDEN_SETS]
golden_answers = [load_file(f"{folder}/answer_ids.safetensors")["answer_ids"].tolist() for folder in GOLDEN_SETS]
# The bread exchange with its whole answer, read in one pass with the answer given, so every format sees the same tokens.
bread_ids = torch.cat([exchanges[1], torch.tensor(golden_answers[1])])
reference = None
for form in FORMATS:
    # Put each rounded weight back into the model, from the original values.
    for name in names:
        original = originals[name].float()
        parameters[name].copy_(original if form.bits is None else fake_quantize(original, form.bits, form.group))
    # The log probabilities at every position of the bread exchange.
    log_probabilities = torch.log_softmax(all_scores(model, bread_ids), dim=-1)
    if reference is None:
        reference = log_probabilities
    # KL(float32 || this format) at each position: how much information the rounding loses about the next token.
    divergence = (reference.exp() * (reference - log_probabilities)).sum(dim=-1)
    # The positions where this format picks the same best token as float32.
    agreement = (log_probabilities.argmax(dim=-1) == reference.argmax(dim=-1)).float().mean()
    # The greedy answers, compared with the golden answers.
    answers = [greedy_answer(model, prompt) for prompt in exchanges]
    same = [answer == golden for answer, golden in zip(answers, golden_answers)]
    size = format_bytes([parameters[name].shape for name in names], form) / MEGABYTE
    print(
        f"{form.name}: {size:.1f} MB, mean KL {divergence.mean():.4f}, largest KL {divergence.max():.4f}, "
        f"same best token {agreement * 100:.1f}% of {bread_ids.shape[0]} positions, same answers (Paris, bread): {same}"
    )
    # The bread answer as text, so a changed answer can be read.
    print(f"  bread answer: {tokenizer.decode(answers[1])!r}")
