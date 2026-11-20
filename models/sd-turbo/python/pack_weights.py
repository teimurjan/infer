# The packed weight format. convert_weights.py writes it, and engine/testdata/make_packed_layers.py makes test data with it.
#
# One row is one output channel: all weights of that channel in PyTorch order, for example in_channels x 3 x 3.
# Each weight becomes a signed integer from -8 to 7 (4 bits) or -128 to 127 (8 bits).
# The integers of a row are packed into u32 words, lowest bits first: value i of a word sits at bits i*bits to i*bits + bits - 1.
# A row ends with zero bits up to a whole word, so every row starts on a word.
# Each row has one float16 scale. The weight is integer x scale.

# numpy does the rounding and the bit packing.
import numpy as np

# The bit widths that the engine kernels can unpack.
BITS = (4, 8)
# Bits in one packed word.
WORD_BITS = 32


# The number of u32 words that hold one row of `row_values` integers.
def row_words(row_values: int, bits: int) -> int:
    # Round the bits of the row up to whole words.
    return -(-row_values * bits // WORD_BITS)


# Round each row of `weight` to `bits` bits. Return the integers (one row per output channel) and the float16 scales.
# `min_scale` is the smallest scale. 0 keeps the rounding that made the UNet files, so they stay the same bytes.
def quantize_rows(weight: np.ndarray, bits: int, min_scale: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    # The largest integer level. 7 for 4 bits, 127 for 8 bits.
    max_level = 2 ** (bits - 1) - 1
    # One row per output channel, in float32. Dim 0 is the output channel for Linear and Conv2d.
    rows = weight.astype(np.float32).reshape(weight.shape[0], -1)
    # The scale maps the largest absolute value of a row to max_level. The minimum avoids a divide by zero.
    # Then no scale is below min_scale. Rows with a larger scale do not change.
    scale = np.maximum(np.maximum(np.abs(rows).max(axis=1), 1e-12) / max_level, min_scale).astype(np.float16)
    # Divide by the scale as stored, so each integer is the nearest one for the scale that the engine reads.
    levels = np.clip(np.rint(rows / scale.astype(np.float32)[:, None]), -max_level - 1, max_level).astype(np.int8)
    # The integers and the scales.
    return levels, scale


# Pack the integers of each row into u32 words.
def pack_rows(levels: np.ndarray, bits: int) -> np.ndarray:
    # How many values one word holds: 8 at 4 bits, 4 at 8 bits.
    per_word = WORD_BITS // bits
    # The words of one row.
    words = row_words(levels.shape[1], bits)
    # Two's complement in `bits` bits: -1 becomes 0b1111 at 4 bits. Pad each row with zeros to whole words.
    fields = np.zeros((levels.shape[0], words * per_word), dtype=np.uint32)
    # Keep only the low `bits` bits of each integer.
    fields[:, : levels.shape[1]] = levels.astype(np.int32) & (2**bits - 1)
    # Shift value i of each word to bits i*bits, then add the values of the word together.
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    # One u32 per word, rows one after another.
    return (fields.reshape(levels.shape[0], words, per_word) << shifts).sum(axis=2, dtype=np.uint32)


# Undo pack_rows: give back `row_values` signed integers per row.
def unpack_rows(words: np.ndarray, bits: int, row_values: int) -> np.ndarray:
    # How many values one word holds.
    per_word = WORD_BITS // bits
    # Shift each value down to bit 0 and keep its low `bits` bits.
    fields = (words[:, :, None] >> (np.arange(per_word, dtype=np.uint32) * bits)) & (2**bits - 1)
    # Values with the top bit set are negative: subtract 2^bits.
    signed = fields.astype(np.int32) - ((fields >> (bits - 1)) << bits).astype(np.int32)
    # Join the words of each row and drop the padding.
    return signed.reshape(words.shape[0], -1)[:, :row_values].astype(np.int8)


# The weight that the engine computes with: integer x scale, in float64, in the original shape.
def dequantize(levels: np.ndarray, scale: np.ndarray, shape: tuple) -> np.ndarray:
    # float64, so the expected output has no rounding of its own.
    return (levels.astype(np.float64) * scale.astype(np.float64)[:, None]).reshape(shape)


# The packed dtype names of the weight index, and their bit widths.
INDEX_BITS = {"int4": 4, "int8": 8}
# The float dtype names of the weight index. The packed UNet keeps its small tensors in float16, the packed text encoder in float32.
INDEX_FLOATS = {"float16": np.float16, "float32": np.float32}


# The bytes of one index entry, as numpy values of `dtype`.
def entry_values(data: np.ndarray, entry: dict, dtype, count: int) -> np.ndarray:
    # `count` values of `dtype`, starting at the offset of the entry.
    return np.frombuffer(data, dtype=dtype, count=count, offset=entry["offset"])


# Every tensor of a packed weight file (convert_weights.py --plan) as the float64 values that the engine computes with,
# keyed by its PyTorch name. `data` is the whole .bin file as bytes, `index` is its parsed .json index.
def read_packed_file(data: np.ndarray, index: list[dict]) -> dict[str, np.ndarray]:
    # Find the scale of a packed weight by its name.
    by_name = {entry["name"]: entry for entry in index}
    # The float values, one per PyTorch parameter.
    state = {}
    for entry in index:
        # The shape and the value count of this tensor.
        shape, values = tuple(entry["shape"]), int(np.prod(entry["shape"]))
        # Scales belong to their packed weight. They are read there.
        if entry["name"].endswith("_scale"):
            continue
        # A packed weight: unpack the integers, then integer x scale.
        if entry["dtype"] in INDEX_BITS:
            # The bits, the rows and the values of each row.
            bits, rows = INDEX_BITS[entry["dtype"]], shape[0]
            row_values = values // rows
            # The words, one row per output channel.
            words = entry_values(data, entry, np.uint32, rows * row_words(row_values, bits)).reshape(rows, row_words(row_values, bits))
            # The float16 scale of each row.
            scale = entry_values(data, by_name[entry["name"] + "_scale"], np.float16, rows)
            # integer x scale in float64, as the converter checks it.
            state[entry["name"]] = dequantize(unpack_rows(words, bits, row_values), scale, shape)
        else:
            # A float tensor: its value, as float64.
            state[entry["name"]] = entry_values(data, entry, INDEX_FLOATS[entry["dtype"]], values).astype(np.float64).reshape(shape)
    # Every tensor.
    return state
