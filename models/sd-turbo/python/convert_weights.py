# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "safetensors", "huggingface_hub"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# argparse reads the command line options.
import argparse
# json writes the index file.
import json
# Path builds file paths and creates folders.
from pathlib import Path

# numpy holds each tensor while we copy it.
import numpy as np
# hf_hub_download finds one model file in the Hugging Face cache, and downloads it on the first run.
from huggingface_hub import hf_hub_download
# safe_open reads one tensor at a time, so a 3.4 GB file never sits in memory all at once.
from safetensors import safe_open

# The packed 4-bit and 8-bit format: rounding, packing, and unpacking for the check.
from pack_weights import pack_rows, quantize_rows, unpack_rows

# Each network, the Hugging Face model that holds it, and the path of its safetensors file inside that model.
# TAESD is a small replacement for the SD-Turbo VAE, from its own model.
NETWORKS = {
    "text_encoder": ("stabilityai/sd-turbo", "text_encoder/model.safetensors"),
    "unet": ("stabilityai/sd-turbo", "unet/diffusion_pytorch_model.safetensors"),
    "vae": ("stabilityai/sd-turbo", "vae/diffusion_pytorch_model.safetensors"),
    "taesd": ("madebyollin/taesd", "diffusion_pytorch_model.safetensors"),
}
# The folder that receives the .bin and .json files.
OUT_DIR = Path("weights")
# Every tensor starts at a multiple of 256 bytes. WebGPU requires this to bind part of a buffer.
ALIGNMENT = 256
# The dtype names that the index can hold, and the numpy type of each. "<" means lowest byte first.
DTYPES = {"float32": np.dtype("<f4"), "float16": np.dtype("<f2")}
# The index dtype of a packed weight, by bit width. The bytes are u32 words, lowest byte first.
PACKED = {4: "int4", 8: "int8"}
# The largest rounding error of a packed weight, as a part of its scale. Half a step, plus room for float rounding.
MAX_STEP_ERROR = 0.5 + 1e-3


# One tensor as the flat file stores it: index name, index dtype, shape, and the bytes.
Stored = tuple[str, str, list[int], bytes]


# Round a byte count up to the next multiple of ALIGNMENT.
def align(offset: int) -> int:
    # Integer math: add ALIGNMENT - 1, then drop the remainder.
    return (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


# The smallest float16 scale of each network. The text encoder's token table has 229 rows of almost zeros. Their scale
# would round to 0 or to a coarse float16 subnormal, so the value check below fails. The smallest normal float16, 6.1e-5, fixes it.
# The UNet keeps 0, the rounding that made unet_packed and unet_q8, so those files keep their bytes.
MIN_SCALE = {"unet": 0.0, "text_encoder": float(np.finfo(np.float16).tiny)}


# The stored form of one source tensor. A weight in the plan becomes packed words plus a float16 scale per row.
def stored_tensors(name: str, tensor: np.ndarray, dtype_name: str, plan: dict[str, int], min_scale: float) -> list[Stored]:
    # "conv_in.weight" belongs to the layer "conv_in". Other names, such as biases, are never in the plan.
    layer = name.removesuffix(".weight")
    # Every tensor that is not a planned weight keeps its values, in the chosen dtype.
    if not name.endswith(".weight") or layer not in plan:
        # Raw numbers in PyTorch's row-major layout. No transpose.
        return [(name, dtype_name, list(tensor.shape), np.ascontiguousarray(tensor.astype(DTYPES[dtype_name])).tobytes())]
    # 4 or 8, from the plan.
    bits = plan[layer]
    # The integers and the scale of each output channel.
    levels, scale = quantize_rows(tensor, bits, min_scale)
    # The packed words keep the PyTorch shape in the index, so the engine knows the layer sizes.
    words = (name, PACKED[bits], list(tensor.shape), pack_rows(levels, bits).astype("<u4").tobytes())
    # One float16 scale per output channel, under the weight name plus "_scale".
    scales = (f"{name}_scale", "float16", [tensor.shape[0]], scale.astype("<f2").tobytes())
    # The words first, then the scales.
    return [words, scales]


# Copy every tensor of one safetensors file into one flat file. Return the index entries and the bytes of each entry.
def write_flat(source: Path, bin_path: Path, dtype_name: str, plan: dict[str, int], min_scale: float) -> tuple[list[dict], dict[str, int]]:
    # One entry per stored tensor, in file order.
    index = []
    # The byte length of each stored tensor, to find the largest one.
    sizes = {}
    # Open the source for reading and the flat file for writing.
    with safe_open(source, framework="np") as reader, bin_path.open("wb") as writer:
        # Sorted names give the same file every run.
        for name in sorted(reader.keys()):
            # One tensor, or the words and the scales of a planned weight.
            for stored_name, stored_dtype, shape, data in stored_tensors(name, reader.get_tensor(name), dtype_name, plan, min_scale):
                # Write zero bytes until the file position is aligned.
                writer.write(b"\0" * (align(writer.tell()) - writer.tell()))
                # Record where this tensor starts, before we write it.
                index.append({"name": stored_name, "offset": writer.tell(), "shape": shape, "dtype": stored_dtype})
                # Write the bytes.
                writer.write(data)
                # Remember the length.
                sizes[stored_name] = len(data)
    # Give the entries and the lengths back to the caller.
    return index, sizes


# Stop if a packed weight is more than half a rounding step away from its source value.
def check_packed(name: str, source: np.ndarray, words: bytes, scale: bytes, bits: int) -> None:
    # One row of source weights per output channel, in float64.
    rows = source.astype(np.float64).reshape(source.shape[0], -1)
    # The words of each row, read back from the file bytes.
    packed = np.frombuffer(words, dtype="<u4").reshape(rows.shape[0], -1)
    # The float16 scales, in float64.
    scales = np.frombuffer(scale, dtype="<f2").astype(np.float64)
    # The integers of each row, unpacked by a separate function from the one that packed them.
    levels = unpack_rows(packed, bits, rows.shape[1]).astype(np.float64)
    # The largest error of each row, measured in steps of that row's scale.
    steps = (np.abs(levels * scales[:, None] - rows).max(axis=1) / scales).max()
    # More than half a step means a wrong integer, a wrong scale, or a wrong bit position.
    if steps > MAX_STEP_ERROR:
        # Stop with the name and the error.
        raise RuntimeError(f"{name}: a packed value is {steps} steps from its source value")


# Read every tensor back from the flat file and stop if one byte differs from what the converter meant to write.
def check_flat(source: Path, bin_path: Path, index: list[dict], dtype_name: str, plan: dict[str, int], min_scale: float) -> None:
    # Map the flat file into memory without reading it all.
    flat = np.memmap(bin_path, dtype=np.uint8, mode="r")
    # Find the offset of each stored tensor by its name.
    offsets = {entry["name"]: entry["offset"] for entry in index}
    # Open the source again.
    with safe_open(source, framework="np") as reader:
        # Check the source tensors one at a time.
        for name in sorted(reader.keys()):
            # The source values.
            tensor = reader.get_tensor(name)
            # What the converter should have written for this tensor.
            stored = stored_tensors(name, tensor, dtype_name, plan, min_scale)
            # The bytes in the file at each stored offset.
            on_disk = [flat[offsets[stored_name] : offsets[stored_name] + len(data)].tobytes() for stored_name, _, _, data in stored]
            # Every byte must be the same.
            if any(disk != data for disk, (_, _, _, data) in zip(on_disk, stored)):
                # Stop with the name of the bad tensor.
                raise RuntimeError(f"{bin_path}: {name} differs from the source")
            # A planned weight also gets the half-step check on the bytes that are on disk.
            if len(stored) == 2:
                # The words, the scales, and the bit width from the plan.
                check_packed(name, tensor, on_disk[0], on_disk[1], plan[name.removesuffix(".weight")])


# Convert one network and print a summary line. A non-empty plan packs the planned weights.
# `stem` names the output files. None means the default name.
def convert(network: str, dtype_name: str, plan: dict[str, int], stem: str | None) -> None:
    # The model and the file of this network.
    model_id, file_path = NETWORKS[network]
    # Find the file in the cache. Only this file is fetched.
    source = Path(hf_hub_download(model_id, file_path))
    # For example weights/unet.bin, or weights/unet_packed.bin with a plan, so the float file stays.
    # --stem gives another name, so a second packed file does not replace the one that the pages load.
    stem = stem or (f"{network}_packed" if plan else network)
    # The flat file.
    bin_path = OUT_DIR / f"{stem}.bin"
    # Write the flat file.
    # The smallest scale of this network. Networks without a plan are never rounded, so 0 is fine for them.
    min_scale = MIN_SCALE.get(network, 0.0)
    index, sizes = write_flat(source, bin_path, dtype_name, plan, min_scale)
    # Every planned layer must have a weight in the source. A missing one means a wrong plan file.
    missing = sorted(set(plan) - {entry["name"].removesuffix(".weight") for entry in index})
    # Stop with the names.
    if missing:
        raise RuntimeError(f"the plan names layers that {network} does not have: {', '.join(missing)}")
    # One JSON object per line, so the index stays easy to read and to search with grep.
    lines = ",\n".join(json.dumps(entry) for entry in index)
    # Write the index next to the flat file, as one JSON list.
    (OUT_DIR / f"{stem}.json").write_text(f"[\n{lines}\n]\n")
    # Read the flat file back and compare it with the source.
    check_flat(source, bin_path, index, dtype_name, plan, min_scale)
    # The name of the largest tensor. One GPU buffer on the iPhone holds about 1073 MB.
    largest = max(sizes, key=sizes.get)
    # How many weights were packed at each bit width.
    packed = ", ".join(f"{sum(bits == width for bits in plan.values())} {PACKED[width]}" for width in sorted(PACKED))
    # One line per network: count, file size, dtype, packed layers, largest tensor.
    print(f"{stem}: {len(index)} tensors, {bin_path.stat().st_size} bytes, {dtype_name}"
          f"{f', packed {packed}' if plan else ''}, largest {largest} {sizes[largest]} bytes, read back equal")


# Read the options from the command line. Return the network names, the dtype name, the plan and the output name.
def parse_args() -> tuple[list[str], str, dict[str, int], str | None]:
    # The parser and its help text.
    parser = argparse.ArgumentParser(description="Convert SD-Turbo safetensors to one flat .bin plus a .json index per network.")
    # Zero or more network names. argparse rejects any default with nargs="*" and choices, so we check the names below.
    parser.add_argument("networks", nargs="*", help=f"any of {', '.join(NETWORKS)}; none means all")
    # float32 is the default, because every kernel reads float32 today.
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32")
    # A JSON file of layer name to 4 or 8, from quant_plan.py for the UNet or text_plan.py for the text encoder.
    parser.add_argument("--plan", type=Path, help="pack the weights of the layers in this plan, for example quant/quant_plan.json")
    # The output name without .bin or .json. The browser cache knows a file by its name, so a changed file needs a new name.
    parser.add_argument("--stem", help="the output file name without extension, for example unet_q8; only with one network")
    # Parse the command line.
    args = parser.parse_args()
    # Names that are not networks of this model.
    unknown = [name for name in args.networks if name not in NETWORKS]
    # Stop with the usage text if there is one.
    if unknown:
        # parser.error prints the message and exits.
        parser.error(f"unknown networks: {', '.join(unknown)}")
    # A plan holds the layer names of one network, so it cannot apply to two.
    if args.plan and len(args.networks) != 1:
        # parser.error prints the message and exits.
        parser.error("--plan works only with a single network")
    # Two networks cannot share one output name.
    if args.stem and len(args.networks) != 1:
        # parser.error prints the message and exits.
        parser.error("--stem works only with a single network")
    # No plan means no packed weights.
    plan = json.loads(args.plan.read_text()) if args.plan else {}
    # A bit width the kernels cannot unpack would make a file the engine cannot use.
    if any(bits not in PACKED for bits in plan.values()):
        # parser.error prints the message and exits.
        parser.error(f"the plan may only use the bit widths {sorted(PACKED)}")
    # No names on the command line means all networks.
    return args.networks or list(NETWORKS), args.dtype, plan, args.stem


# Read the options.
networks, dtype_name, plan, stem = parse_args()
# Create the output folder if it is missing.
OUT_DIR.mkdir(exist_ok=True)
# Convert the networks one after the other.
for network in networks:
    # Write, index, and check one network.
    convert(network, dtype_name, plan, stem)
