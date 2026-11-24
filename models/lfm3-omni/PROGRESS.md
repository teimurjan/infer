# Progress log

This file records every step of the LFM3-Omni work, with the exact outputs and the calculations made from them. The rules for new entries are in `AGENTS.md` at the repository root.

## Devices and runtimes

| Name in this log | Device | Runtime |
|---|---|---|
| Mac native | MacBook with Apple M1 Max, 24-core GPU, 34.4 GB of memory | Rust, wgpu on Metal |
| Mac Python | the same Mac | Python through `uv run`, PyTorch on the CPU |
| iPhone | iPhone 16 Pro Max | Safari, WebGPU, the WASM build |

Units: MB is 10^6 bytes. A multiply-add counts as 2 operations.

## 2026-11-24: The choice of LFM3-Omni-2B, and a golden set of one text exchange

The user chose LFM3 as the next model after SD-Turbo, and agreed to start with a reference run and golden tensors.

Why this model. LFM3-Omni-2B (Liquid AI, October 2026) is one model that takes text, images, video and 16 kHz audio, and answers in text and 24 kHz speech. Liquid publishes an official ONNX export. Its smallest set of files for a voice exchange is the 4-bit language model (`decoder_q4f16.onnx_data`, 1413.4 MB), the audio encoder (`audio_encoder_fp16.onnx_data`, 255.2 MB), the speech token decoder (`depth_decoder.onnx_data`, 335.6 MB) and the codec (`codec_decoder_fp16.onnx_data`, 89.5 MB). That is 1413.4 + 255.2 + 335.6 + 89.5 = 2093.7 MB, above the 1920 MB tab budget of the iPhone (`CLAUDE.md`). The sizes come from `HfApi().model_info("LiquidAI/LFM3-Omni-2B-ONNX", files_metadata=True)`. So the ready-made web runtimes are not likely to run a voice exchange on the iPhone. That gap is the reason to build this engine.

Facts from the model card and `config.json` (commit `c01718a568e53ae936e6b5e1ff5ab2de7d7f388b`):

- The language model has 24 layers: 12 short convolution layers, 6 full attention layers and 6 attention layers with a window of 256 tokens. Width 2048, feed-forward width 7680, 16 attention heads with 4 key-value heads of 128 values. The vocabulary has 128,000 tokens, and the output layer shares its weights with the token table.
- Speech out: the LiquidCodec makes 24 kHz audio from frames of 1920 samples, so 24000 / 1920 = 12.5 frames per second. Each frame has 12 codes of 2017 values. The depth decoder makes the 12 codes of a frame. In the official `generate`, one speech frame is one generation slot, so speech in real time needs 12.5 passes of the language model per second.
- License: LFM Open License v1.0. Commercial use is licensed only for an entity below 10 million US dollars of annual revenue.

Files added:

- `python/dump_text_golden.py`: runs the official model in float32 on the CPU with its own code, and saves the golden set `golden/text/`. Git ignores `golden/`.
- `PROGRESS.md`: this log. `plan.md`: the plan of this model.

Design choices:

- The model and its code are pinned to commit `c01718a568e53ae936e6b5e1ff5ab2de7d7f388b`, so a later upload cannot change the golden set.
- The model needs its own code (`trust_remote_code=True`). Before the first run, the agent searched the 14 Python files of the checkpoint for network, process, file and `eval` calls with `grep`. The only such use is `os.path` in `processing_lfm3_omni.py`, to build the folder of the codec.
- The first golden set is text only, so the first engine step needs only the language model. "Respond in text." in the system turn asks for a text answer, as the model card says.
- The set holds the input of layer 0 and the output of each layer for the 32 prompt tokens, the scores of the next token, and the greedy answer.

Command: `uv run python/dump_text_golden.py`, in the model folder. Device: Mac Python. The second run, with the pinned commit (only these lines kept):

```
parameters model.audio_encoder: 109505344
parameters model.audio_projector: 5772800
parameters model.codebook_embedding: 49569792
parameters model.depth_decoder: 83898560
parameters model.language_model: 1728801536
parameters model.multi_modal_projector: 25174016
parameters model.vision_tower: 164206336
parameters audio_tokenizer (codec file): 73559090
prompt ids (32): [124894, 124899, 23630, 207, 89056, 296, 2803, 22, 124900, 207, 124899, 5922, 207, 2992, 355, 278, 5205, 302, 3980, 39, 41774, 296, 734, 1858, 22, 124900, 207, 124899, 63514, 207, 124901, 124902]
next token: 41176 "Paris"
answer ids (2): [41176, 124900]
answer: "Paris<|im_end|>"
first answer token equals the best score of the prompt pass: True
golden/text: 29 tensors
```

The first run, before the pin, printed the same lines. `md5 -q golden/text/*.safetensors | md5 -q` printed `73769b809f92b6a0ff4fb16709556ceb` after both runs, so the set is the same bytes.

A check in Mac Python: `hidden_24` and `last_hidden_state` have a largest difference of `0.0`. They are the same tensor. The set has no separate output of layer 23 before the final norm.

Calculations:

- Parameters of the model: 109505344 + 5772800 + 49569792 + 83898560 + 1728801536 + 25174016 + 164206336 = 2166928384, the 2.17B of the model card. The codec file adds 73559090.
- The token table: 128000 x 2048 = 262144000 parameters, 15.2% of the language model.
- Weights of a voice exchange, without vision. The language model at 4 bits: 1728801536 x 0.5 = 864.4 MB, plus scales. The audio encoder, audio projector, codebook table, depth decoder and codec at 8 bits: 109505344 + 5772800 + 49569792 + 83898560 + 73559090 = 322305586 bytes = 322.3 MB. The codec file also holds its encoder, which speech out does not need. Total about 1186.7 MB, below the 1920 MB budget. These are estimates from parameter counts, not measured.
- The memory of earlier tokens: 6 full attention layers x 2 (keys and values) x 4 heads x 128 values x 2 bytes (float16) = 12288 bytes per token. For 4,096 tokens: 50.3 MB.

Not measured on the iPhone. Next step: the language model in the engine, layer by layer against `golden/text`, starting with one short convolution layer.
