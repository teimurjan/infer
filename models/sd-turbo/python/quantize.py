# torch holds the weight tensors.
import torch


# Round a weight to `bits` bits with one scale per output channel, and give it back in the weight's own dtype.
def fake_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    # The largest integer level. 7 for 4 bits, 127 for 8 bits.
    max_level = 2 ** (bits - 1) - 1
    # One row per output channel. Linear is (out, in), Conv2d is (out, in, 3, 3), so dim 0 is the output channel.
    rows = weight.float().reshape(weight.shape[0], -1)
    # The scale maps the largest absolute value in each row to max_level. The minimum avoids a divide by zero.
    scale = rows.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / max_level
    # Divide, round to the nearest integer, and keep it inside the signed range.
    levels = (rows / scale).round().clamp(-max_level - 1, max_level)
    # Multiply back, so the math stays float16 but the values are the rounded ones.
    return (levels * scale).reshape(weight.shape).to(weight.dtype)


# Return every UNet layer with a weight matrix: the convolutions and the linear layers. Norm weights stay float16.
def quantizable_layers(unet: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    # Keep the name, because the plan and the tables use it.
    return [(name, module) for name, module in unet.named_modules() if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
