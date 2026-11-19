# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["torch", "diffusers", "transformers", "accelerate", "safetensors"]
# ///
# The block above tells `uv run` which Python and which packages this script needs.

# torch holds the tensors and runs the UNet on the Mac GPU.
import torch
# StableDiffusionPipeline wires the UNet and the scheduler together, the same way as dump_golden.py.
from diffusers import StableDiffusionPipeline
# load_file reads .safetensors files.
from safetensors.torch import load_file
# fake_quantize rounds one weight, and quantizable_layers lists the layers to round. Both live in quantize.py.
from quantize import fake_quantize, quantizable_layers

# The Hugging Face name of the model.
MODEL_ID = "stabilityai/sd-turbo"
# "mps" is the Mac GPU. The CPU needs over an hour for one float16 run, the GPU about 0.2 s.
DEVICE = "mps"
# Every run uses float16 weights and activations, like the engine.
DTYPE = torch.float16
# The same starting noise as the golden set.
NOISE_PATH = "golden/starting_noise.safetensors"
# The float16 text encoder output. It does not change between runs, so we read it instead of running the encoder.
TEXT_PATH = "golden/float16/text_encoder.safetensors"
# The float16 final latent from the CPU, to show how far the GPU baseline is from it.
GOLDEN_LATENT_PATH = "golden/float16/final_latent.safetensors"
# The bit widths to test, in the order the user asked for them.
BIT_WIDTHS = (4, 8)


# Run text-to-latent once from the saved noise and give back the final latent in float32 on the CPU.
def final_latent(pipe: StableDiffusionPipeline, noise: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    # One denoising step, no guidance, our noise and our text, and stop before the VAE.
    latent = pipe(prompt_embeds=text, num_inference_steps=1, guidance_scale=0.0, latents=noise, output_type="latent").images
    # float32 on the CPU makes the error math exact.
    return latent.float().cpu()


# Return the largest absolute difference and the root mean square difference between two latents.
def errors(latent: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    # The difference at each latent value.
    difference = latent - reference
    # Largest error, and the typical error over all 16,384 values.
    return difference.abs().max().item(), difference.pow(2).mean().sqrt().item()


# For each layer, round only that layer, run the pipeline, put the weight back, and record the error.
def sweep(pipe, layers, noise, text, reference, bits: int) -> list[tuple[str, int, float, float]]:
    # One row per layer: name, weight count, largest error, root mean square error.
    rows = []
    # Go through the layers in model order.
    for name, layer in layers:
        # Keep the float16 weight, so we can put it back.
        original = layer.weight.data.clone()
        # Replace the weight with its rounded copy.
        layer.weight.data.copy_(fake_quantize(original, bits))
        # Measure the final latent with only this layer rounded.
        largest, rms = errors(final_latent(pipe, noise, text), reference)
        # Put the float16 weight back before the next layer.
        layer.weight.data.copy_(original)
        # Save the result for this layer.
        rows.append((name, original.numel(), largest, rms))
    # Give every row back.
    return rows


# Print the rows with the largest root mean square error first.
def print_table(rows: list[tuple[str, int, float, float]], bits: int) -> None:
    # A title line that names the bit width.
    print(f"\n{bits}-bit, one layer at a time, sorted by rms error of the final latent:")
    # The column names.
    print(f"{'rank':>4}  {'layer':<72} {'weights':>10} {'rms':>10} {'max':>10}")
    # Sort by the rms error, largest first.
    for rank, (name, count, largest, rms) in enumerate(sorted(rows, key=lambda row: -row[3]), start=1):
        # One line per layer.
        print(f"{rank:>4}  {name:<72} {count:>10} {rms:>10.3e} {largest:>10.3e}")


# Load the pipeline in float16 on the Mac GPU. The text encoder and the VAE load too, but they do not run.
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
# Hide the per-run progress bar, because we run the pipeline hundreds of times.
pipe.set_progress_bar_config(disable=True)
# The saved noise, in float16 on the GPU, like dump_golden.py passes it.
noise = load_file(NOISE_PATH)["starting_noise"].to(DEVICE, DTYPE)
# The saved text encoder output, on the GPU.
text = load_file(TEXT_PATH)["text_encoder"].to(DEVICE, DTYPE)
# Every UNet layer with a weight matrix.
layers = quantizable_layers(pipe.unet)
# Turn off gradient tracking, because we only change weights by hand and run forward.
torch.set_grad_enabled(False)
# The float16 baseline on the GPU, with no layer rounded. Every error below is measured against it.
reference = final_latent(pipe, noise, text)
# A second baseline run. If it differs from the first, the GPU is not repeatable and the errors below are noisy.
repeat_largest, _ = errors(final_latent(pipe, noise, text), reference)
# The CPU float16 golden latent.
golden_largest, golden_rms = errors(reference, load_file(GOLDEN_LATENT_PATH)["final_latent"].float())
# How many layers we test, and how many weights they hold.
print(f"{len(layers)} layers, {sum(layer.weight.numel() for _, layer in layers)} weights")
# The noise floor of the method: zero means the GPU gives the same bits every run.
print(f"baseline repeat: max {repeat_largest:.3e}")
# How far the GPU baseline is from the CPU golden file. This is not part of the errors below.
print(f"baseline vs golden/float16 final_latent: max {golden_largest:.3e}, rms {golden_rms:.3e}")
# One sweep and one table per bit width.
for bits in BIT_WIDTHS:
    # Run every layer, then print the sorted table.
    print_table(sweep(pipe, layers, noise, text, reference, bits), bits)
