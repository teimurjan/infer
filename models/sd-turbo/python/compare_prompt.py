# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "diffusers", "transformers", "accelerate", "safetensors", "numpy"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.
# Makes the image of one prompt three ways in PyTorch, with the golden noise of the engine, to see where a quality loss comes from:
#   1_original.png: SD-Turbo as published, float32 UNet and the full VAE decoder. This is the base model.
#   2_taesd.png:    the same float32 UNet, decoded with TAESD, the small decoder of the engine.
#   3_engine_math.png: the packed 4-bit and 8-bit UNet of weights/<packed>.bin, decoded with TAESD. This is what the engine computes.
# Usage: uv run compare_prompt.py "a photo of a dog" --out prompt_images/dog [--packed unet_q8] [--seed 1234]
# Without --seed, the noise is golden/starting_noise.safetensors, which is seed 42.

# argparse reads the command line.
import argparse
# json reads the index of the packed weight file.
import json
# Path builds file paths.
from pathlib import Path

# numpy reads the packed weight file.
import numpy as np
# torch runs the models on the CPU.
import torch
# The SD-Turbo pipeline, and TAESD.
from diffusers import AutoencoderTiny, StableDiffusionPipeline
# load_file reads the golden noise.
from safetensors.torch import load_file

# The reader of the packed format, the same as dump_packed_golden.py uses.
from pack_weights import read_packed_file

# The model, as in dump_golden.py.
MODEL_ID = "stabilityai/sd-turbo"
# The noise that the engine pages use, [1, 4, 64, 64].
NOISE_PATH = Path("golden/starting_noise.safetensors")
# The folder of the packed weight files.
WEIGHTS = Path("weights")


# The final latent of one SD-Turbo step for `prompt`, from the golden noise, as dump_golden.py makes it.
def final_latent(pipe: StableDiffusionPipeline, prompt: str, noise: torch.Tensor) -> torch.Tensor:
    # One step, no guidance, 512 x 512, the latent only.
    return pipe(prompt=prompt, height=512, width=512, num_inference_steps=1, guidance_scale=0.0, latents=noise.clone(), output_type="latent").images


# Write an image [1, 3, 512, 512] in [-1, 1] as a PNG, with the same rounding as the engine.
def save_png(pipe: StableDiffusionPipeline, image: torch.Tensor, path: Path) -> None:
    pipe.image_processor.postprocess(image.float(), output_type="pil")[0].save(path)
    print(f"wrote {path}")


def main() -> None:
    # The prompt and the output folder.
    parser = argparse.ArgumentParser(description="Make the image of one prompt three ways in PyTorch.")
    parser.add_argument("prompt")
    parser.add_argument("--out", type=Path, required=True, help="the folder for the three PNG files")
    parser.add_argument("--packed", default="unet_packed", help="the packed UNet file in weights/, without .bin (default unet_packed)")
    parser.add_argument("--seed", type=int, help="the seed of the noise, as on the engine page; without it, the golden noise (seed 42)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # The float32 pipeline on the CPU, as for the golden set.
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, dtype=torch.float32).to("cpu")
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).to("cpu")
    # The noise of the seed, made as diffusers and the engine make it, or the golden noise.
    noise = torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(args.seed), dtype=torch.float32) if args.seed is not None else load_file(NOISE_PATH)["starting_noise"]
    # The ids that the pipeline's tokenizer gives, to compare with the page of the engine.
    ids = pipe.tokenizer(args.prompt, padding="max_length", max_length=77, truncation=True).input_ids
    print(f"ids: {[id for id in ids if id != 0]}")
    with torch.inference_mode():
        # 1 and 2: the published float32 UNet, with each decoder.
        latent = final_latent(pipe, args.prompt, noise)
        save_png(pipe, pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0], args.out / "1_original.png")
        save_png(pipe, taesd.decode(latent / taesd.config.scaling_factor, return_dict=False)[0], args.out / "2_taesd.png")
        # 3: the same pipeline with the packed weights of the engine in the UNet, decoded with TAESD.
        packed = read_packed_file(np.fromfile(WEIGHTS / f"{args.packed}.bin", dtype=np.uint8), json.loads((WEIGHTS / f"{args.packed}.json").read_text()))
        pipe.unet.load_state_dict({name: torch.from_numpy(values).float() for name, values in packed.items()})
        latent = final_latent(pipe, args.prompt, noise)
        save_png(pipe, taesd.decode(latent / taesd.config.scaling_factor, return_dict=False)[0], args.out / "3_engine_math.png")


# Run main only when the file runs as a script.
if __name__ == "__main__":
    main()
