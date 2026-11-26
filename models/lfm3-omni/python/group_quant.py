# The 4-bit and 8-bit weight format of LFM3, in the packed format of packages/engine-core (see pack_weights.py of SD-Turbo).
#
# One weight row is one output channel. Each run of `group` neighbor values of a row shares one float16 scale, so a row of
# 2048 values has 64 scales at a group of 32. The scales of a matrix are [out][groups], row after row.
# Each weight becomes a signed integer from -8 to 7 (4 bits) or -128 to 127 (8 bits). The weight is integer x scale.
# The integers of a row are packed into u32 words, lowest bits first: value i of a word sits at bits i*bits to i*bits + bits - 1.
# A row ends with zero bits up to a whole word, so every row starts on a word.
# quant_eval.py, export_weights.py and engine/testdata/make_grouped_linear.py import it.

# numpy packs the bits.
import numpy as np
# torch rounds the weights.
import torch

# Bits in one packed word.
WORD_BITS = 32


# Round `weight` [out, in] to `bits`-bit integers with one float16 scale per `group` values of a row.
# None as the group means one scale for the whole row. It gives the integers [out, in] and the scales [out, groups].
def quantize_groups(weight: torch.Tensor, bits: int, group: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    # The largest integer level: 7 for 4 bits, 127 for 8 bits.
    max_level = 2 ** (bits - 1) - 1
    out_features, in_features = weight.shape
    # One group per row when the format has no group size.
    size = group or in_features
    # [out][groups][size], so each group gets its own scale.
    groups = weight.float().reshape(out_features, in_features // size, size)
    # The scale maps the largest absolute value of a group to max_level. It is rounded to float16, as the engine reads it.
    # The minimum avoids a divide by zero.
    scale = (groups.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / max_level).half()
    # A few rows of conv.in_proj hold only values near 4e-36, so their float16 scale is 0. Those groups get the integer 0,
    # and the weight becomes 0, as the engine computes it from a scale of 0. Dividing by 1 there avoids 0 / 0.
    divisor = torch.where(scale > 0, scale.float(), torch.ones_like(scale, dtype=torch.float32))
    # The nearest integer for the scale as stored, kept inside the range of `bits` bits.
    levels = torch.clamp(torch.round(groups / divisor), -max_level - 1, max_level).to(torch.int8)
    return levels.reshape(out_features, in_features), scale.reshape(out_features, -1)


# The weight that the engine computes with: integer x scale, in float32, [out, in].
def dequantize_groups(levels: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out_features, in_features = levels.shape
    # Each scale covers in / groups neighbor values of its row.
    grouped = levels.float().reshape(out_features, scale.shape[1], -1) * scale.float()[:, :, None]
    return grouped.reshape(out_features, in_features)


# Round and give back integer x scale in float32, the weight that quant_eval.py puts into the PyTorch model.
def fake_quantize(weight: torch.Tensor, bits: int, group: int | None) -> torch.Tensor:
    return dequantize_groups(*quantize_groups(weight, bits, group))


# Pack the integers [out, in] of each row into u32 words: [out, words per row].
def pack_rows(levels: torch.Tensor, bits: int) -> np.ndarray:
    rows, row_values = levels.shape
    # How many values one word holds: 8 at 4 bits, 4 at 8 bits.
    per_word = WORD_BITS // bits
    # The words of one row, rounded up.
    words = -(-row_values * bits // WORD_BITS)
    # Two's complement in `bits` bits: -1 becomes 0b1111 at 4 bits. Pad each row with zeros to whole words.
    fields = np.zeros((rows, words * per_word), dtype=np.uint32)
    # Keep only the low `bits` bits of each integer.
    fields[:, :row_values] = levels.numpy().astype(np.int32) & (2**bits - 1)
    # Shift value i of each word to bits i*bits, then add the values of the word together.
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    return (fields.reshape(rows, words, per_word) << shifts).sum(axis=2, dtype=np.uint32)
