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

## 2026-11-25: Layer 0 of the language model in the engine

The user asked to start the engine. Layer 0 is a short convolution layer, and the golden set has its input (`hidden_00`) and its output (`hidden_01`).

The layer, from `modeling_lfm3.py` of the pinned commit (`Lfm3DecoderLayer` and `Lfm3ShortConv`), with sandwich norms (`use_sandwich_norm = True`):

1. `n = operator_norm(hidden)`, an RMSNorm with eps 1e-5.
2. `in_proj(n)` gives three parts of 2048 values per token: B, C and x. No biases (`conv_bias = False`).
3. `h = B * x`, then a causal depthwise convolution of width 7 over the tokens: token t sees tokens t - 6 to t, and tokens before 0 count as 0.
4. `y = C * conv(h)`, then `out_proj(y)`.
5. `hidden += post_operator_norm(out_proj(y))`.
6. `hidden += post_ffn_norm(w2(silu(w1(m)) * w3(m)))`, with `m = ffn_norm(hidden)`. The feed-forward width is 7680.

Files added:

- `python/export_weights.py`: writes chosen tensors of the language model as float32 in the flat format of engine-core: `weights/<stem>.bin` and `weights/<stem>.json`, each tensor on a multiple of 256 bytes. The names lose `model.language_model.`. Git ignores `weights/`.
- `engine/Cargo.toml`: the crate `lfm3-engine`, on top of `packages/engine-core`.
- `engine/src/kernels/rms_norm.wgsl` and `rms_norm.rs`: RMSNorm of each row, one workgroup of 256 per row. With `NormMode::Add`, the result is added to the output: the residual add after a sandwich norm, in the same kernel.
- `engine/src/kernels/short_conv.wgsl` and `short_conv.rs`: steps 3 and 4 before `out_proj`, in one kernel: one invocation per token and channel.
- `engine/src/kernels/swiglu.wgsl` and `swiglu.rs`: `gate = silu(gate) * up`, in place.
- `engine/src/kernels/mod.rs`, `engine/src/linear.rs` (nn.Linear without bias on the linear kernel of engine-core, with a zero bias), `engine/src/conv_layer.rs` (`ConvLayer`: the 11 kernels of one layer), `engine/src/lib.rs`.
- `engine/tests/conv_layer.rs`: layer 0 against `hidden_01`.

Design choices:

- The sizes come from the weight shapes: the width from `operator_norm`, the feed-forward width from `w1`, the convolution width from `conv.conv`. No size is written twice.
- The layer changes `hidden` in place, as the residual stream of PyTorch does. The two norms after a branch add into it, so no separate add kernel is needed.
- The weights stay float32 for this step. The 4-bit weights come with step 3 of `plan.md`.

Commands and outputs. Device: Mac Python, then Mac native.

`uv run python/export_weights.py layer0 layers.0.`, in the model folder:

```
layer0: 10 tensors, 255942656 bytes, float32
```

`cargo test --release -- --nocapture`, in `engine/`:

```
layer 0: largest difference from PyTorch 2.2649765e-6, largest value 0.9128683
test layer_0_matches_pytorch ... ok
```

The first version of the test had a tolerance of 1e-3, with a comment that the values reach about 30. The run showed a largest value of 0.913, so the tolerance is now 1e-4.

Deliberate-bug check: `taps[channel * shape.kernel + tap]` became `taps[channel * shape.kernel + (shape.kernel - 1u - tap)]` in `short_conv.wgsl`, so the convolution taps run in reverse. Same command:

```
layer 0: largest difference from PyTorch 8.243309e-1, largest value 0.9128683
largest difference 8.243309e-1 is above 1e-4
```

The bug was then removed, and the test passed again with `2.2649765e-6`.

Calculation: the parameters of layer 0 are 2048 x 7 + 6144 x 2048 + 2048 x 2048 + 3 x 7680 x 2048 + 4 x 2048 = 63,985,664. In float32 that is 4 x 63,985,664 = 255,942,656 bytes, exactly the size of the file. Every tensor of this layer is a multiple of 256 bytes long, so the file has no padding.

Not measured on the iPhone. Next step: an attention layer (layer 2 is the first full attention layer), with its rotary positions and the keys and values for the memory of earlier tokens.

## 2026-11-25: The attention layers, full and sliding

The user asked for the next step: the first attention layers. Layer 2 is a full attention layer and layer 4 a sliding one. The golden set has their inputs and outputs.

The attention, from `modeling_lfm3.py` of the pinned commit (`Lfm3Attention`, `Lfm3RotaryEmbedding`, `apply_rotary_pos_emb`):

- `n = operator_norm(hidden)`. Then `q_proj(n)` gives 16 heads of 128 values, and `k_proj(n)` and `v_proj(n)` give 4 heads of 128. Query head h uses key head h / 4 (`repeat_kv`).
- `q_norm` and `k_norm`: an RMSNorm of each head of 128 values.
- Rotary positions in float32, on the pairs (j, j + 64). Full layers: theta 1,000,000, and `partial_rotary_factor` 0.5 gives 32 rotated pairs. The other 32 have a frequency of 0, so they do not turn. Sliding layers: theta 10,000, and all 64 pairs turn.
- Causal attention, scores times 1 / sqrt(128). Sliding layers see only keys with `position - key < 256`.
- The output of each head times `sigmoid(gate_proj(n))`, one gate per head, then `o_proj`, `post_operator_norm` and the residual add. Then the same feed-forward half as layer 0.

Files added:

- `engine/src/kernels/rope.wgsl` and `rope.rs`: the rotation, in place, one invocation per pair. It reads cos and sin from tables.
- `engine/src/rope_table.rs`: the cos and sin tables on the CPU, in float32 in the same order as PyTorch: `1 / theta^(2j / 128)`, then `position * inv_freq`, then cos and sin. A unit test checks that position 0 does not turn.
- `engine/src/kernels/attention.wgsl` and `attention.rs`: one workgroup of 128 lanes per query row and head. The keys go by in chunks of 128, with a running maximum and total (the online softmax of SD-Turbo step 4). The kernel takes the number of keys and the position of the first query row, so the memory of earlier tokens can use it later. A key that is not visible has weight 0, so a chunk with no visible key changes nothing.
- `engine/src/attention_layer.rs`: `AttentionLayer` and `AttentionKind` (Full or Sliding, with the settings of `config.json`).
- `engine/src/feed_forward.rs`: the feed-forward half, moved out of `conv_layer.rs`, because both kinds of layer use it.
- `engine/tests/common/mod.rs`: the helpers of the layer tests. `engine/tests/attention_layer.rs`: layers 2 and 4.

Files changed: `engine/src/conv_layer.rs` (uses `bind_feed_forward`), `engine/src/kernels/mod.rs`, `engine/src/lib.rs`, `engine/tests/conv_layer.rs` (uses the shared helpers).

Design choices:

- The rotary tables come from the CPU, because GPU `sin` and `cos` of angles up to 32 radians are less exact than the CPU ones.
- The head counts come from the shapes of `q_proj` and `k_proj`. The head size of 128 is fixed in the kernel, and the layer divides by it.
- q and k are normed into new buffers, because one dispatch cannot read and write the same buffer through two bindings.

Commands and outputs. Device: Mac Python, then Mac native.

`uv run python/export_weights.py layer2 layers.2.` and the same for layer 4:

```
layer2: 14 tensors, 230851584 bytes, float32
layer4: 14 tensors, 230851584 bytes, float32
```

`cargo test --release -- --nocapture`, in `engine/` (only these lines kept):

```
test rope_table::tests::position_zero_is_not_rotated ... ok
layer 2: largest difference from PyTorch 2.861023e-6, largest value 8.547484
layer 4: largest difference from PyTorch 4.529953e-6, largest value 12.328884
test attention_layers_match_pytorch ... ok
layer 0: largest difference from PyTorch 2.2649765e-6, largest value 0.9128683
test layer_0_matches_pytorch ... ok
```

Layer 0 gives the same number as before the move of the feed-forward half.

Deliberate-bug checks, one at a time, with `cargo test --release --test attention_layer -- --nocapture`:

1. `values[first] = a * c - b * s` became `a * c + b * s` in `rope.wgsl`:

```
layer 2: largest difference from PyTorch 5.8372736e-2, largest value 8.547484
layer 4: largest difference from PyTorch 1.1593775e-1, largest value 12.328884
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s
```

2. `let kv_head = head / (shape.heads / shape.kv_heads);` became `head % shape.kv_heads` in `attention.wgsl`:

```
layer 2: largest difference from PyTorch 1.3567052e0, largest value 8.547484
layer 4: largest difference from PyTorch 4.7960162e-1, largest value 12.328884
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s
```

Both bugs were then removed, and `cargo test --release` passed 3 tests, 0 failed.

Not tested yet: the window of the sliding layers. The prompt has 32 tokens, fewer than 256, so no key is outside the window. A test with more than 256 positions is needed when the memory of earlier tokens comes.

Not measured on the iPhone. Next step: all 24 layers in a row, the token table and the output scores, against `next_logits` and the answer "Paris".

## 2026-11-25: The whole language model over the prompt

The user asked for the next step: all 24 layers in a row, the token table and the scores of the next token, against `next_logits` and the answer "Paris".

The forward pass, from `Lfm3Model.forward` of the pinned commit: `hidden = embed_tokens(ids)` with no scale, then the 24 layers, then `embedding_norm` (an RMSNorm, the final norm), then the output layer, which shares its weights with `embed_tokens` (`tie_word_embeddings`). `hidden_24` and `last_hidden_state` of the golden set are both after the final norm.

Files added:

- `engine/src/config.rs`: `LAYER_KINDS`, the kind of each of the 24 layers from `layer_types` in `config.json`. The weights tell a convolution layer from an attention layer, but not full from sliding, so the list is written out.
- `engine/src/language_model.rs`: `LanguageModel` binds the 24 layers, the final norm and the output layer for a prompt. `token_rows` reads the rows of the prompt ids from the token table on the CPU.
- `engine/tests/language_model.rs`: the whole model against `hidden_00`, `last_hidden_state`, `next_logits`, and the token of "Paris".

Files changed:

- `engine/src/linear.rs`: `bind_linear_matrix` takes a matrix that the caller holds, for the chunks of the output layer. `bind_linear` calls it.
- `engine/src/lib.rs`: the new modules.
- `engine/tests/common/mod.rs`: `#![allow(dead_code)]`, because each test file uses only some of the helpers.
- `packages/compare/src/safetensors.rs`: reads `I64` tensors, for `prompt_ids` and `answer_ids`. The root log has a short entry.

Design choices:

- The float32 token table is 128000 x 2048 x 4 = 1,048,576,000 bytes. With the default WebGPU limits, one buffer holds at most 256 MiB and one storage binding at most 128 MiB. So in this float32 step the token rows come from the CPU, and the output layer runs in 8 chunks of 16000 tokens: 16000 x 2048 x 4 = 131,072,000 bytes each, under 134,217,728. The 4-bit weights of the next steps make the table small enough for the GPU.
- The weights come in two files: `weights/layers.bin` with the 24 layers, streamed into GPU buffers, and `weights/embed.bin` with the token table and the final norm, read on the CPU.
- The whole prompt runs in one submit of 350 dispatches. The Mac has no 250 ms limit in this test. The iPhone will need smaller submits.

Commands and outputs. Device: Mac Python, then Mac native.

`uv run python/export_weights.py layers layers.` and `uv run python/export_weights.py embed embed`:

```
layers: 284 tensors, 5866621952 bytes, float32
embed: 2 tensors, 1048584192 bytes, float32
```

The export of `layers` took 15.6 s (`time`).

`cargo test --release -- --nocapture`, in `engine/`:

```
layer 2: largest difference from PyTorch 2.861023e-6, largest value 8.547484
layer 4: largest difference from PyTorch 4.529953e-6, largest value 12.328884
layer 0: largest difference from PyTorch 2.2649765e-6, largest value 0.9128683
token rows: largest difference from PyTorch 0e0
last hidden state: largest difference from PyTorch 1.487732e-4
next scores: largest difference from PyTorch 5.1498413e-5, best token 41176
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.38s
```

The first run of `language_model` printed the same three lines. Its score tolerance was 1e-2. That is about 200 times the result, so it is now 1e-3. The hidden tolerance stays 1e-3, about 7 times the result.

Deliberate-bug check: layer 2 in `LAYER_KINDS` became `Sliding` in place of `Full`. Command: `cargo test --release --test language_model -- --nocapture`.

```
last hidden state: largest difference from PyTorch 6.024132e0
next scores: largest difference from PyTorch 1.267034e0, best token 41176
last hidden state 6.024132e0 is above 1e-3
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 5.55s
```

With the bug, the best token is still "Paris". So the answer alone is a weak check, and the test also compares the numbers. The bug was then removed, and all 4 tests passed.

Calculations:

- Parameters in `layers.bin`: 5866621952 / 4 = 1,466,655,488. With the token table, 128000 x 2048 = 262,144,000, and the final norm, 2048: 1,466,655,488 + 262,144,000 + 2,048 = 1,728,801,536, the parameter count of `model.language_model`.
- The prompt run: 4.38 s for the whole test, with the load of 5.87 GB. The time of the GPU work alone is not measured yet.

Not measured on the iPhone. Next step: the word-by-word loop with the memory of earlier tokens (the keys and values of the attention layers, and the last 6 inputs of each convolution layer), on the Mac, against `answer_ids`.

## 2026-11-26: The word-by-word loop with the memory of earlier tokens

The user asked for the next step: the word-by-word loop, against a golden answer.

The golden answer of `golden/text` is 2 tokens ("Paris", then the end), so it would test one step only. A new golden set has a longer prompt and a longer answer.

Files added:

- `python/dump_loop_golden.py`: the golden set `golden/loop`. The prompt is a passage about bread, written for this test, and a question, 353 tokens, more than the 256-token window of the sliding layers. A greedy loop over `model.model.language_model` with a `DynamicCache` writes the answer and the scores of every step. The official `generate` in text mode must give the same tokens, or the script stops. `generation_config.json` asks for greedy choice: `do_sample` false, `repetition_penalty` 1.0, stop on 124900 or 125014.
- `engine/src/memory.rs`: `Memory`, the memory of earlier tokens for up to `capacity` positions. An attention layer keeps the keys and values of every position. A convolution layer keeps `h = B * x` of every position. A small uniform buffer, the step, holds `[start, keys]`: the position of the first new token and the positions that exist after the run. It also holds the rotary tables of both attention kinds.
- `engine/src/kernels/conv_gate.wgsl`: `h = B * x` of the new tokens into the history of the layer, at their positions.
- `engine/src/kernels/copy_rows.wgsl` and `copy_rows.rs`: copies rows, to a fixed row or from the start of the run. The attention layers store their new keys and values with it, and the model takes its last row with it.
- `engine/src/output_layer.rs`: the final norm and the output layer, uploaded once. The prompt run and the word run both bind to it.
- `engine/src/generate.rs`: `Generator`. The prompt goes in one run. Then one run per token, with the model bound once for 1 row. The CPU writes the step and the token row before each run.
- `engine/tests/generate.rs`: the loop against `golden/loop`, step by step.

Files changed:

- `engine/src/kernels/short_conv.wgsl`: the convolution reads the history in place of computing h itself. `rope.wgsl` and `attention.wgsl` take the first position and the key count from the step. `attention.wgsl` reads the memory of keys and values.
- `engine/src/config.rs`: `WIDTH`, `KV_HEADS` and `rope_settings`. `AttentionKind` is gone: the kind comes from `LAYER_KINDS`.
- `engine/src/conv_layer.rs`, `attention_layer.rs`, `language_model.rs`: they bind to the memory, and take the layer index in place of a name prefix. The language model computes the scores of the last row only.
- `engine/src/lib.rs`, `engine/tests/common/mod.rs` (`prompt_memory`), and the three earlier tests: the new arguments.
- `packages/engine-core/src/gpu.rs`: `writable_uniform_buffer` and `write_u32`, because the step changes before every run. The root log has an entry.

Design choices:

- One bound model serves every word. Only the step and the token row change between runs, so a word costs no new bind group.
- The history of a convolution layer has a row per position, like the keys. It costs 512 x 2048 x 4 = 4,194,304 bytes per layer for 512 positions. A ring of the last 6 rows would be smaller. That can come with the memory plan of the iPhone.
- The keys go into the memory after their norm and rotation, at the index of their position. The window check of the sliding layers uses positions, so it works the same for the prompt and for each word.

Failed first: the first run stopped with `Shader 'copy_rows' parsing error: name `target` is a reserved keyword`. The buffer is now named `destination`.

Commands and outputs.

`uv run python/dump_loop_golden.py`, in the model folder. Device: Mac Python. It took 1 min 43.9 s (`time`).

```
prompt tokens: 353
answer ids (24): [41, 5373, 20, 6373, 7359, 6101, 278, 32106, 708, 19317, 1428, 278, 25405, 309, 7315, 296, 278, 14725, 551, 4538, 310, 835, 22, 124900]
answer: "A slow, cool rise gives the dough more flavor because the yeast and bacteria in the flour have longer to work.<|im_end|>"
official generate gives the same tokens: True
golden/loop: 3 tensors
```

`cargo test --release -- --nocapture`, in `engine/`. Device: Mac native.

```
layer 2: largest difference from PyTorch 2.861023e-6, largest value 8.547484
layer 4: largest difference from PyTorch 4.529953e-6, largest value 12.328884
layer 0: largest difference from PyTorch 2.2649765e-6, largest value 0.9128683
prompt tokens: 353, answer tokens: 24
largest score difference per step: ["6.7e-5", "3.3e-5", "3.1e-5", "2.7e-5", "3.1e-5", "3.1e-5", "3.2e-5", "2.6e-5", "3.2e-5", "2.5e-5", "3.3e-5", "2.7e-5", "2.8e-5", "3.0e-5", "2.9e-5", "3.6e-5", "2.9e-5", "3.1e-5", "3.8e-5", "3.2e-5", "4.1e-5", "4.1e-5", "2.8e-5", "4.1e-5"]
largest score difference of all steps: 6.723404e-5
same answer as PyTorch: true
test answer_matches_pytorch_word_by_word ... ok
token rows: largest difference from PyTorch 0e0
dispatches: 386
last hidden state: largest difference from PyTorch 1.487732e-4
next scores: largest difference from PyTorch 5.1498413e-5, best token 41176
```

The earlier numbers did not change. The language model has 386 dispatches now, 36 more than before: 2 for the convolution of each of the 12 convolution layers in place of 1 (12 more), 2 copies into the memory in each of the 12 attention layers (24 more), and 1 copy of the last row with 1 chunk run per chunk (the same 8 + 1 as before, plus 1 copy, minus 1).

Deliberate-bug checks, one at a time, with `cargo test --release --test generate -- --nocapture`:

1. The window of the sliding layers became 0 (no window) in `rope_settings`:

```
largest score difference of all steps: 2.5141022e1
same answer as PyTorch: false
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.98s
```

2. Each word one position too late: `set_step(gpu, rows + step as u32 + 1, 1)` in `generate.rs`:

```
largest score difference of all steps: 2.4555809e1
same answer as PyTorch: false
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.83s
```

Both bugs were then removed, and all 5 tests passed.

The first check is the test of the sliding window that "The attention layers, full and sliding" left open: the prompt has 353 tokens, so the first 97 keys are outside the window of the last query.

Not measured: the time per word. Not measured on the iPhone. Next step: 4-bit weights of the language model, with a golden set of the same exchange from the 4-bit weights, and the first time per word.
