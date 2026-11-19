# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# Path builds the folder paths.
from pathlib import Path

# torch makes the random data and the expected outputs.
import torch
# F.scaled_dot_product_attention is PyTorch's attention: softmax(q k^T / sqrt(64)) v.
import torch.nn.functional as F

# Each case gets a folder next to this script.
OUT_DIR = Path(__file__).parent
# A fixed seed, so every run makes the same data.
SEED = 0
# 320 channels = 5 heads of 64, as in the first UNet block of SD-Turbo.
HEADS = 5
HEAD_SIZE = 64

# name: (SD-Turbo layer, query tokens, key tokens). The names and sizes must match ATTENTION_CASES in src/attention/cases.rs.
CASES = {
    # Self attention: 64 x 64 = 4096 image tokens look at each other.
    "attention_self": ("down_blocks.0.attentions.0.transformer_blocks.0.attn1", 4096, 4096),
    # Cross attention: the 4096 image tokens look at the 77 text tokens.
    "attention_cross": ("down_blocks.0.attentions.0.transformer_blocks.0.attn2", 4096, 77),
}


# Write a tensor as raw float32 bytes, lowest byte first ("<f4"), with no header.
def write_f32(tensor, path):
    # Move to numpy and write.
    tensor.float().numpy().astype("<f4").tofile(path)


# [tokens, heads * 64] -> [heads, tokens, 64], the layout that scaled_dot_product_attention wants. float64 for the exact answer.
def split_heads(tensor):
    return tensor.double().view(tensor.shape[0], HEADS, HEAD_SIZE).transpose(0, 1)


# Make the four files of one case.
def make_case(name, layer_name, query_tokens, key_tokens):
    # Start every case from the same seed, so one case does not change another.
    torch.manual_seed(SEED)
    # Random rows in the layout of the UNet before it splits the heads: [tokens, heads * 64].
    query = torch.randn(query_tokens, HEADS * HEAD_SIZE)
    key = torch.randn(key_tokens, HEADS * HEAD_SIZE)
    value = torch.randn(key_tokens, HEADS * HEAD_SIZE)
    # Attention per head in float64, then back to [tokens, heads * 64]. Rounded to float32 only when written,
    # so the test measures only the error of our GPU kernel.
    expected = F.scaled_dot_product_attention(split_heads(query), split_heads(key), split_heads(value))
    expected = expected.transpose(0, 1).reshape(query_tokens, HEADS * HEAD_SIZE)
    # The folder of this case.
    folder = OUT_DIR / name
    # Make it if it does not exist yet.
    folder.mkdir(exist_ok=True)
    # The four files.
    write_f32(query, folder / "query.bin")
    write_f32(key, folder / "key.bin")
    write_f32(value, folder / "value.bin")
    write_f32(expected, folder / "expected.bin")
    # Say what was written.
    print(f"{name} ({layer_name}): {query_tokens} queries x {key_tokens} keys, {HEADS} heads of {HEAD_SIZE}")


# Make every case.
for case_name, (layer_name, query_tokens, key_tokens) in CASES.items():
    make_case(case_name, layer_name, query_tokens, key_tokens)
