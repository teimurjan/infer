# Writes the flat weight format that packages/engine-core reads: <stem>.bin with every tensor, and <stem>.json with one
# {name, offset, shape, dtype} entry per tensor. Each tensor starts on a multiple of 256 bytes.
# export_weights.py and engine/testdata/make_grouped_linear.py use it.

# Iterable is the type of the tensor list. json writes the index.
from collections.abc import Iterable
import json
# Path builds the file paths.
from pathlib import Path

# numpy arrays hold the bytes of each tensor.
import numpy as np

# Each tensor starts on a multiple of this many bytes, as packages/engine-core expects.
ALIGNMENT = 256


# Write `tensors` to <stem>.bin and <stem>.json. Each tensor is (name, values, shape, dtype name). The values are written
# as they are, so they must already have the little-endian type of the dtype: <f4 for float32, <f2 for float16, <u4 words
# for int4 and int8. The shape of a packed tensor is the shape of its weight, [out, in]. It gives the size of the .bin file.
def write_weight_file(stem: Path, tensors: Iterable[tuple[str, np.ndarray, list[int], str]]) -> int:
    index = []
    with stem.with_suffix(".bin").open("wb") as writer:
        for name, values, shape, dtype in tensors:
            # Zero bytes up to the next multiple of 256.
            writer.write(b"\0" * (-writer.tell() % ALIGNMENT))
            index.append({"name": name, "offset": writer.tell(), "shape": shape, "dtype": dtype})
            writer.write(np.ascontiguousarray(values).tobytes())
    # One JSON object per line, as the SD-Turbo index files are.
    stem.with_suffix(".json").write_text("[\n" + ",\n".join(json.dumps(entry) for entry in index) + "\n]\n")
    return stem.with_suffix(".bin").stat().st_size
