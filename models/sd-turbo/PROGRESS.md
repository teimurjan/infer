# Progress log

This file records every step of the engine work, with the exact outputs and the calculations made from them. The rules for new entries are in `AGENTS.md`.

## Devices and runtimes

| Name in this log | Device | Runtime |
|---|---|---|
| Mac native | MacBook with Apple M1 Max, 24-core GPU | Rust, wgpu 30.0.2 on Metal |
| Mac Chromium | the same Mac | Playwright Chromium, WebGPU, the WASM build |
| iPhone | iPhone 16 Pro Max | Safari, WebGPU, the WASM build, opened through an ngrok tunnel |

The project limits for the iPhone are in `CLAUDE.md`, and the evidence is in `crash-record/2026-11-15-iphone16promax.md`. Every submit must stay under 250 ms.

Units: GB/s and G ops/s are 10^9 per second. MB is 10^6 bytes. A multiply-add counts as 2 operations.

## 2026-11-17: Rust crate and the SiLU kernel

The engine moved from TypeScript to Rust with wgpu in `engine/`, because the user chose that. The same crate runs natively and compiles to WASM with `wasm-pack build --release --target web --out-dir web/pkg --no-typescript`.

The first kernel is SiLU, `x * sigmoid(x)`, in `engine/src/silu/silu.wgsl`. One invocation computes one value. A workgroup has 64 invocations. Each invocation finds its value with `global_invocation_id.x`.

The test data comes from `engine/testdata/silu/make_data.py`: 100003 values, `randn * 4`, plus the edge values -80 and 80, with `F.silu` in float32.

Commands: `cargo test --release --test silu -- --nocapture`, and `engine/web/index.html`.

Mac native and Mac Chromium printed the same line:

```
SiLU: 100003 values, largest error 9.536743e-7
```

Mutation check: a wrong bounds check in the shader gave the error below, and the test failed. The shader was restored.

```
SiLU: 100003 values, largest error 1.4729282e1
```

## 2026-11-17: Copy and math benchmarks

Files: `engine/src/bench/`. Native runner: `cargo run --release --example bench`. Web page: `engine/web/bench.html`.

Method:

1. Finish the buffer uploads first, so they are not part of a timed submit.
2. Run one warm-up submit and drop its time.
3. Calibrate: double the dispatches per submit while one submit takes less than 50 ms. The cap is 4096.
4. Time 30 submits. Wait for `onSubmittedWorkDone` after each one.
5. If one submit takes 250 ms or more, stop with an error.
6. Report the median and the best rate.

Copy reads one 64 MiB buffer and writes another, with vec4 loads and 256 invocations per workgroup. Bytes per dispatch = 2 x 64 MiB = 134,217,728.

Math runs 2^20 invocations. Each invocation runs 256 loop steps of 4 `fma` calls. Operations per dispatch = 2^20 x 256 x 4 x 2 = 2,147,483,648.

Mac native, two copy runs and the main math run:

```
copy: 351.84 GB/s median, 363.05 GB/s best (30 runs, 256 dispatches per submit, 97.7 ms median submit, 105.2 ms longest)
copy: 367.15 GB/s median, 371.89 GB/s best (30 runs, 256 dispatches per submit, 93.6 ms median submit, 98.3 ms longest)
math: 1424.69 G ops/s median, 1463.35 G ops/s best (30 runs, 64 dispatches per submit, 96.5 ms median submit, 142.3 ms longest)
```

Mac Chromium:

```
copy: 343.25 GB/s median, 347.07 GB/s best (30 runs, 128 dispatches per submit, 50.0 ms median submit, 52.1 ms longest)
math: 4155.37 G ops/s median, 4177.48 G ops/s best (30 runs, 128 dispatches per submit, 66.2 ms median submit, ...)
```

The native math rate is about 3 times lower than the Chromium rate on the same GPU. Three shader changes did not raise the native rate. Each change was reverted.

| Change | Mac native math |
|---|---|
| 4 separate chains | 940.95 G ops/s median |
| A loop 8 times longer | 1484.42 G ops/s median |
| Plain multiply and add instead of `fma` | 1424.84 G ops/s median |

Conclusion: native wgpu (naga) compiles this loop to slower Metal code than Chromium does. Compare the iPhone only with browser numbers.

## 2026-11-17: The first iPhone benchmark failed

The user reported this from the iPhone: one submit took 256 ms, and the limit is 250 ms.

Change: the error now names the benchmark and the submit phase. `gpu.finish_uploads()` now runs before the warm-up, so the 64 MiB upload is not inside a submit. After the change, the iPhone run passed. The change matches the cause, but this run does not prove the cause.

iPhone:

```
copy: 47.72 GB/s median, 48.26 GB/s best (30 runs, 32 dispatches per submit, 90.0 ms median submit, 91.0 ms longest)
math: 636.29 G ops/s median, 648.30 G ops/s best (30 runs, 16 dispatches per submit, 54.0 ms median submit, 54.0 ms longest)
```

Calculations:

- iPhone copy is 47.72 / 351.84 = 0.136 of Mac native copy, which is about 7.4 times slower.
- iPhone math is 636.29 / 4155.37 = 0.153 of Mac Chromium math, which is about 6.5 times slower.
- A memory-bound kernel on the iPhone costs about bytes / 47.7 GB/s. A math-bound kernel costs about operations / 636 G ops/s.

## 2026-11-17: Linear and convolution kernels

Files: `engine/src/layers/` (linear, conv 1x1, conv 3x3 with stride 1 and 2), `engine/src/cases.rs`, `engine/testdata/make_layers.py`, `engine/tests/layers.rs`, `engine/web/layers.html`. No subgroups.

Block size: each invocation computes a block of 4 x 4 outputs, and a workgroup is 8 x 8 invocations.

- A 4 x 4 block loads 4 input vec4 and 4 weight vec4 and does 16 dot products. That is 4 times fewer loads per output than 1 output per invocation.
- An 8 x 8 block needs more registers, so fewer invocations fit on the GPU, and the grid gets 4 times smaller. For the stride 2 case, the grid falls from 20,480 to 5,120 invocations.
- vec4 gives 16-byte loads. Every SD-Turbo dimension here is a multiple of 4.
- The block size is not tuned on the iPhone.

The cases use real SD-Turbo shapes at a 512 x 512 image (64 x 64 latent). The expected output is computed in float64 and then rounded to float32. The tolerance is 1e-4.

| Case | SD-Turbo layer | Shape |
|---|---|---|
| linear | `down_blocks.0.attentions.0.proj_in` | 4096 x 320 to 320 |
| conv1x1 | `down_blocks.1.resnets.0.conv_shortcut` | 320 to 640 channels at 32 x 32 |
| conv3x3 | `down_blocks.0.resnets.0.conv1` | 320 to 320 channels at 64 x 64, stride 1 |
| conv3x3_stride2 | `down_blocks.0.downsamplers.0.conv` | 320 to 320 channels, 64 x 64 to 32 x 32 |

Bytes per run count every tensor once (input + weight + bias + output, 4 bytes each). Operations per run count a multiply-add as 2.

| Case | Values per run | Bytes per run | Operations per run |
|---|---|---|---|
| linear | 1,310,720 + 102,400 + 320 + 1,310,720 | 10,896,640 | 2 x 4096 x 320 x 320 = 838,860,800 |
| conv1x1 | 327,680 + 204,800 + 640 + 655,360 | 4,753,920 | 2 x 320 x 640 x 1024 = 419,430,400 |
| conv3x3 | 1,310,720 + 921,600 + 320 + 1,310,720 | 14,173,440 | 2 x 320 x 320 x 9 x 4096 = 7,549,747,200 |
| conv3x3_stride2 | 1,310,720 + 921,600 + 320 + 327,680 | 10,241,280 | 2 x 320 x 320 x 9 x 1024 = 1,887,436,800 |

Command: `cargo test --release --test layers -- --nocapture`.

Mac native:

```
linear: 19.13 GB/s median, 19.18 GB/s best (30 runs, 128 dispatches per submit, 72.9 ms median submit, 83.6 ms longest), largest error 1.1920929e-6
conv1x1: 25.80 GB/s median, 26.04 GB/s best (30 runs, 512 dispatches per submit, 94.4 ms median submit, 98.1 ms longest), largest error 1.9073486e-6
conv3x3: 0.83 GB/s median, 1.02 GB/s best (30 runs, 4 dispatches per submit, 68.5 ms median submit, 77.0 ms longest), largest error 6.4373016e-6
conv3x3_stride2: 1.32 GB/s median, 1.32 GB/s best (30 runs, 8 dispatches per submit, 62.2 ms median submit, 63.5 ms longest), largest error 6.198883e-6
```

Mac Chromium:

```
linear: 18.21 GB/s median, 18.40 GB/s best (30 runs, 128 dispatches per submit, 76.6 ms median submit, 87.8 ms longest), largest error 1.1920929e-6
conv1x1: 16.68 GB/s median, 24.15 GB/s best (30 runs, 256 dispatches per submit, 73.0 ms median submit, 144.7 ms longest), largest error 1.9073486e-6
conv3x3: 0.70 GB/s median, 0.75 GB/s best (30 runs, 4 dispatches per submit, 80.7 ms median submit, 89.1 ms longest), largest error 6.4373016e-6
conv3x3_stride2: 1.66 GB/s median, 1.75 GB/s best (30 runs, 16 dispatches per submit, 99.0 ms median submit, 104.0 ms longest), largest error 6.198883e-6
```

Time per run = bytes per run / median rate. Operation rate = operations per run / time per run.

| Case | Mac native time per run | Mac native operation rate |
|---|---|---|
| linear | 0.570 ms | about 1470 G ops/s |
| conv1x1 | 0.184 ms | about 2280 G ops/s |
| conv3x3 | 17.1 ms | about 440 G ops/s |
| conv3x3_stride2 | 7.76 ms | about 240 G ops/s |

The conv3x3 kernel is the slow one. It loads one scalar at a time and checks the padding on every load. A possible fix is to repack the weights so that one vec4 load gets 4 output channels, and to load each input row once.

Mutation check: with the 3x3 padding removed, the errors were about 4.0, and the test failed. The shader was restored.

## 2026-11-17: Layer kernels on the iPhone

Page: `engine/web/layers.html`. The first two runs on the iPhone showed only some of the lines:

```
linear: 8.61 GB/s median, 8.72 GB/s best (30 runs, 64 dispatches per submit, 81.0 ms median submit, 82.0 ms longest), largest error 2.3841858e-6
```

```
linear: 8.35 GB/s median, 8.61 GB/s best (30 runs, 64 dispatches per submit, 83.5 ms median submit, 87.0 ms longest), largest error 2.3841858e-6
conv1x1: 7.70 GB/s median, 8.01 GB/s best (30 runs, 128 dispatches per submit, 79.0 ms median submit, 81.0 ms longest), largest error 1.9073486e-6
```

The full run on the iPhone:

```
linear: 8.40 GB/s median, 8.50 GB/s best (30 runs, 64 dispatches per submit, 83.0 ms median submit, 85.0 ms longest), largest error 2.3841858e-6
conv1x1: 7.80 GB/s median, 8.22 GB/s best (30 runs, 128 dispatches per submit, 78.0 ms median submit, 82.0 ms longest), largest error 1.9073486e-6
conv3x3: 0.65 GB/s median, 0.65 GB/s best (30 runs, 4 dispatches per submit, 87.0 ms median submit, 91.0 ms longest), largest error 6.4373016e-6
conv3x3_stride2: 1.74 GB/s median, 1.76 GB/s best (30 runs, 16 dispatches per submit, 94.0 ms median submit, 98.0 ms longest), largest error 6.198883e-6
```

| Case | iPhone time per run | iPhone operation rate | Share of the 636 G ops/s math rate |
|---|---|---|---|
| linear | 1.30 ms | about 645 G ops/s | about 100% |
| conv1x1 | 0.61 ms | about 690 G ops/s | about 100% |
| conv3x3 | 21.8 ms | about 346 G ops/s | about 54% |
| conv3x3_stride2 | 5.89 ms | about 321 G ops/s | about 50% |

All errors are under 1e-5, and the longest submit is 98 ms. Linear and conv1x1 run at the math limit of the iPhone. The 3x3 kernels reach about half of it, so they are the next kernels to make faster. The iPhone is faster than Mac native on conv3x3_stride2 (1.74 against 1.32 GB/s).

## 2026-11-17: Attention

Files: `engine/src/attention/attention.wgsl`, `engine/src/attention/mod.rs`, `engine/src/attention/cases.rs`, `engine/testdata/make_attention.py`, `engine/tests/attention.rs`, `engine/web/attention.html`.

Shared changes in the same step:

- `engine/src/memory.rs`: a counter of GPU buffer bytes. A new buffer adds its bytes. A dropped buffer removes them. The counter keeps the peak.
- `engine/src/dispatch.rs`: one compiled pipeline can have many bind groups. A run can be several dispatches.
- The report line changed from `(30 runs, N dispatches per submit, ...)` to `(30 submits, N runs per submit, ...)`. One run of attention is one dispatch per chunk.
- `engine/src/checks.rs`: the size checks that the layers and attention share.

Design: one dispatch per chunk of query rows covers all heads. The kernel computes the scores against all keys and writes them to a scratch buffer of chunk x keys x heads values. Then it finds the largest score of each row, computes `exp(score - largest)`, and adds up weight x value. The chunk is a parameter. Every chunk reuses the same scratch buffer, so the full tokens x tokens table never exists. The head size is fixed at 64, because every SD-Turbo attention layer uses 64.

| Case | SD-Turbo layer | Shape |
|---|---|---|
| attention_self | `down_blocks.0.attentions.0.transformer_blocks.0.attn1` | 4096 queries, 4096 keys, 5 heads of 64 |
| attention_cross | `down_blocks.0.attentions.0.transformer_blocks.0.attn2` | 4096 queries, 77 text keys, 5 heads of 64 |

The expected output is `F.scaled_dot_product_attention` in float64, rounded to float32. The tolerance is 1e-4.

Operations per run = 4 x queries x keys x 64 x heads (scores and weighted sum, 2 operations per multiply-add). The softmax is left out.

- Self: 4 x 4096 x 4096 x 64 x 5 = 21,474,836,480.
- Cross: 4 x 4096 x 77 x 64 x 5 = 403,701,760.

Peak memory for self attention at chunk 256:

| Buffer | Bytes |
|---|---|
| query, key, value: 3 x 4096 x 320 x 4 | 15,728,640 |
| output: 4096 x 320 x 4 | 5,242,880 |
| scores: 256 x 4096 x 5 x 4 | 20,971,520 |
| read-back copy of the output | 5,242,880 |
| 16 chunk size buffers of 32 bytes | 512 |
| Total, the same as the printed peak | 47,186,432 |

The full scores table for self attention is 4096 x 4096 x 5 x 4 = 335,544,320 bytes. If the self attention peak is not below it, the test fails. For cross attention, the full table is only 6.3 MB, so the test checks only the error there.

Design steps on Mac native, self attention, chunk 256 unless the table says otherwise:

| Design | Time per run | Rate | Result |
|---|---|---|---|
| 1 invocation per row, 64 rows per workgroup, shared key and value tiles | 543.9 ms (warm-up) | about 39 G ops/s | stopped by the 250 ms limit |
| The same, chunk 1024 | 166.2 ms | 129.18 G ops/s | 3.3 times faster, so too few invocations was the cause |
| 4 lanes per row, 64 rows, tiles | 190.7 ms | 112.62 G ops/s | 2.85 times faster than the first design |
| 4 lanes per row, 64 rows, no tiles | 195.4 ms | 109.90 G ops/s | the barriers were not the limit |
| 8 lanes per row, 32 rows | 115.6 ms | 185.79 G ops/s | |
| 16 lanes per row, 16 rows | 67.1 ms | 320.25 G ops/s | kept |
| 16 lanes, each weight computed once | 66.2 ms | 324.61 G ops/s | 1.5% faster but more code, so not kept |

At chunk 256, 5 heads, and 16 rows per workgroup, one chunk has 16 x 5 = 80 workgroups of 256 invocations. The first design had 4 x 5 = 20 workgroups of 64 invocations.

Bugs found during the step:

- The lanes read the scores of other lanes, and that needs `storageBarrier()`. The first version had only `workgroupBarrier()`, which does not cover storage memory.
- Chromium rejected `const LOWEST: f32 = -3.4028235e38;` because the literal is past the f32 limit. Metal accepted it. The constant is now `-3.4e38`. In the browser, the failed shader did not stop the page. The page printed an error of 1.768633e-1 instead.

Mutation checks. Each bug made the test fail, and the shader was restored:

```
every head reads the keys of head 0: attention_self chunk 256: error 1.7432314e-1, attention_cross chunk 256: error 1.6718882e0
every chunk starts at row 0:          attention_self chunk 256: error 1.768633e-1,  attention_cross chunk 256: error 1.6794683e0
```

Command: `uv run testdata/make_attention.py`, then `cargo test --release --test attention -- --nocapture`, both in `engine/`.

Mac native, final kernel:

```
attention_self: 322.44 G ops/s median, 323.09 G ops/s best (30 submits, 1 runs per submit, 66.6 ms median submit, 66.8 ms longest), chunk 256 rows, largest error 4.8056245e-7, peak 47186432 bytes allocated (47.2 MB, the full scores table alone would be 335.5 MB)
attention_cross: 223.67 G ops/s median, 226.60 G ops/s best (30 submits, 32 runs per submit, 57.8 ms median submit, 58.3 ms longest), chunk 256 rows, largest error 8.34465e-7, peak 16320512 bytes allocated (16.3 MB, the full scores table alone would be 6.3 MB)
attention_self: 368.13 G ops/s median, 369.18 G ops/s best (30 submits, 1 runs per submit, 58.3 ms median submit, 59.0 ms longest), chunk 1000 rows, largest error 4.8056245e-7, peak 108134560 bytes allocated (108.1 MB, the full scores table alone would be 335.5 MB)
```

A rerun after the `LOWEST` fix gave lower and less steady numbers: self 268.48 G ops/s median and 311.68 G ops/s best, cross 138.37 G ops/s median. The cause was not found.

The same run gave these layer lines in the new format. The kernels did not change, but conv3x3 went from 0.83 to 1.08 GB/s. The cause of this difference was not found.

```
linear: 19.12 GB/s median, 19.16 GB/s best (30 submits, 128 runs per submit, 72.9 ms median submit, 73.6 ms longest), largest error 1.1920929e-6
conv1x1: 25.84 GB/s median, 26.11 GB/s best (30 submits, 512 runs per submit, 94.2 ms median submit, 94.4 ms longest), largest error 1.9073486e-6
conv3x3: 1.08 GB/s median, 1.08 GB/s best (30 submits, 4 runs per submit, 52.7 ms median submit, 53.0 ms longest), largest error 6.4373016e-6
conv3x3_stride2: 1.33 GB/s median, 1.33 GB/s best (30 submits, 8 runs per submit, 61.6 ms median submit, 61.9 ms longest), largest error 6.198883e-6
```

Mac Chromium, `engine/web/attention.html`, chunk 256:

```
attention_self: 156.64 G ops/s median, 160.26 G ops/s best (30 submits, 1 runs per submit, 137.1 ms median submit, 137.4 ms longest), chunk 256 rows, largest error 4.7683716e-7, peak 47186432 bytes allocated (47.2 MB, the full scores table alone would be 335.5 MB)
attention_cross: 179.67 G ops/s median, 187.50 G ops/s best (30 submits, 32 runs per submit, 71.9 ms median submit, 72.2 ms longest), chunk 256 rows, largest error 1.5497208e-6, peak 16320512 bytes allocated (16.3 MB, the full scores table alone would be 6.3 MB)
```

Chunk 1000 calculations: scores = 1000 x 4096 x 5 x 4 = 81,920,000 bytes. Peak = 81,920,000 + 15,728,640 + 5,242,880 + 5,242,880 + 5 x 32 = 108,134,560 bytes, the same as the printed peak.

## 2026-11-17: Attention on the iPhone

Page: `engine/web/attention.html`, chunk 256.

```
attention_self: 148.10 G ops/s median, 149.13 G ops/s best (30 submits, 1 runs per submit, 145.0 ms median submit, 147.0 ms longest), chunk 256 rows, largest error 4.917383e-7, peak 47186432 bytes allocated (47.2 MB, the full scores table alone would be 335.5 MB)
attention_cross: 171.11 G ops/s median, 176.97 G ops/s best (30 submits, 32 runs per submit, 75.5 ms median submit, 76.0 ms longest), chunk 256 rows, largest error 8.34465e-7, peak 16320512 bytes allocated (16.3 MB, the full scores table alone would be 6.3 MB)
```

Calculations:

- Self time per run = 21,474,836,480 / 148.10e9 = 145.0 ms. That is under the 250 ms limit for one submit.
- Cross time per run = 403,701,760 / 171.11e9 = 2.36 ms.
- Self rate is 148.10 / 636.29 = 23% of the iPhone math rate.
- The iPhone self run (145.0 ms) is close to the Mac Chromium self run (137.1 ms). The Mac GPU is much faster on the math benchmark. So the kernel is limited by memory loads, not by math. In both passes, each multiply-add needs one load.

Next step: give each invocation a 4 x 4 block, as in the linear kernel. In pass 1 that is 4 query rows x 4 keys. In pass 2 it is 4 query rows x 4 output values. Each loaded value is then used 4 times. Not built or measured yet.

## 2026-11-17: Weight converter

Files added: `convert_weights.py`. The script writes `weights/text_encoder.bin`, `weights/unet.bin`, `weights/vae.bin`, and one `.json` index next to each `.bin`.

Design choices:

- One flat `.bin` file per network holds the raw numbers of every tensor, with no header. The engine reads the file and makes GPU buffers from byte ranges.
- The `.json` index is a list with one object per line: `name`, `offset` in bytes, `shape`, and `dtype`. The engine finds each tensor by its name.
- Each tensor starts at a multiple of 256 bytes. If a shader binds part of a buffer, WebGPU requires an offset that is a multiple of 256 bytes. With this alignment, the engine can use one buffer for many tensors.
- The tensors keep the PyTorch layout. The linear and convolution kernels already read that layout, so the converter does not transpose.
- The tensors are sorted by name, so two runs give the same file.
- The default dtype is float32, because every kernel reads float32. The `--dtype float16` option halves the file for later work on the iPhone.
- The VAE file keeps the encoder tensors. The engine does not need them for text to image, but the index lets it skip them.
- After it writes a file, the script reads every tensor back and compares it bit for bit with the source. A difference stops the script.
- The script reads one tensor at a time with `safe_open`, so the 3.4 GB UNet file never sits in memory all at once.

Device and runtime: Mac, Python through `uv run`, CPU only. This runtime is not in the table, because it does not use the GPU.

Command: `uv run convert_weights.py`, in the project folder.

```
text_encoder: 372 tensors, 1361551360 bytes, float32, largest text_model.embeddings.token_embedding.weight 202375168 bytes, read back equal
unet: 686 tensors, 3463643136 bytes, float32, largest up_blocks.0.resnets.0.conv1.weight 117964800 bytes, read back equal
vae: 248 tensors, 334616576 bytes, float32, largest decoder.mid_block.resnets.0.conv1.weight 9437184 bytes, read back equal
```

Command: `uv run convert_weights.py --dtype float16 vae`. A float32 run then replaced the file.

```
vae: 248 tensors, 167308928 bytes, float16, largest decoder.mid_block.resnets.0.conv1.weight 4718592 bytes, read back equal
```

The first lines of `weights/text_encoder.json`:

```
[
{"name": "text_model.embeddings.position_embedding.weight", "offset": 0, "shape": [77, 1024], "dtype": "float32"},
{"name": "text_model.embeddings.token_embedding.weight", "offset": 315392, "shape": [49408, 1024], "dtype": "float32"},
```

An audit script summed the tensor sizes from each index and compared them with the file:

```
text_encoder: tensor bytes 1361551360, file bytes 1361551360, last tensor ends at 1361551360, padding 0, unaligned offsets 0
unet: tensor bytes 3463642896, file bytes 3463643136, last tensor ends at 3463643136, padding 240, unaligned offsets 0
vae: tensor bytes 334615452, file bytes 334616576, last tensor ends at 334616576, padding 1124, unaligned offsets 0
```

Calculations:

- Text encoder tensors: 2 embeddings + 23 layers x 16 tensors + 2 final LayerNorm tensors = 372. That is the printed count.
- Second offset: 77 x 1024 x 4 = 315,392 bytes. That is a multiple of 256, so the alignment added no padding there.
- Largest text encoder tensor: 49408 x 1024 x 4 = 202,375,168 bytes. It fits in one iPhone GPU buffer (about 1073 MB).
- The whole text encoder in float32 is 1361.6 MB. That is more than one iPhone GPU buffer, so the engine needs several buffers. It is also 1361.6 / 1920 = 71% of the tab budget.
- The UNet in float32 is 3463.6 MB. That is more than the 1920 MB tab budget, so the iPhone must use float16 or smaller. Half of it is 1731.8 MB. This number is a calculation, not a measurement.
- VAE float16 file: 167,308,928 bytes. Half of the float32 file is 167,308,288 bytes. The 640 extra bytes are alignment padding, because smaller tensors leave more gaps.

Failed runs:

- The first version gave the network names `choices` and a list default. argparse checks the default against the choices and stopped: `convert_weights.py: error: argument networks: invalid choice: ['text_encoder', 'unet', 'vae'] (choose from 'text_encoder', 'unet', 'vae')`.
- An empty list default failed in the same way: `convert_weights.py: error: argument networks: invalid choice: [] (choose from 'text_encoder', 'unet', 'vae')`. The script now has no `choices` for the names and checks them itself. `uv run convert_weights.py bogus` prints `convert_weights.py: error: unknown networks: bogus`.

Mutation check. A copy of the script recorded every offset 256 bytes too far. The read-back check stopped it:

```
RuntimeError: weights/vae.bin: decoder.conv_in.bias differs from the source
```

Not measured on the iPhone.

Conclusion: the three flat files and their indexes match the source bit for bit. Next step: add per-layer samples of the text encoder to `dump_golden.py`, so every later engine step has a reference to compare with.

## 2026-11-17: UNet step 1, weight loader and conv_in

The user asked for the full UNet: conv_in, four down levels, the middle block, four up levels with the skip tensors, and conv_out. The user also asked for a golden compare after every block, and for every submit to stay under 0.25 s. The project rule allows one step per prompt. This step builds the parts that every later block needs, and the first block, conv_in.

Files added:

- `engine/src/weights/mod.rs`, `engine/src/weights/index.rs`, `engine/src/weights/file.rs`: read one tensor at a time from `weights/unet.bin`, with the `.json` index.
- `engine/src/unet/mod.rs`, `engine/src/unet/conv.rs`: `run_unet`, the `Activation` type, and a 3x3 convolution whose weights come from the weight file.
- `engine/src/submit.rs`: one timed submit, and the 0.25 s limit, which `bench/measure.rs` also uses now.
- `engine/tests/unet.rs`: runs the UNet and compares each block with `golden/float32`.
- `compare/src/lib.rs`: the compare crate is now a library and a command line tool, so the engine test reads golden files with the same code.

Files changed: `dump_golden.py`, `engine/Cargo.toml`, `engine/src/lib.rs`, `engine/src/error.rs`, `engine/src/gpu.rs`, `engine/src/checks.rs`, `engine/src/layers/*.rs`, `engine/src/bench/measure.rs`, `compare/src/main.rs`.

Design choices:

- `dump_golden.py` now also saves `unet_input` (the scaled latent that the UNet reads), `timestep`, and `conv_in`. A forward pre-hook on the UNet records the first two. The golden set had no tensor before the first down block, so conv_in had nothing to compare with.
- The engine reads weights through a `WeightSource` trait. The Mac reads the file with `read_exact_at`, one tensor at a time. The browser will need another source, so the UNet code does not know about files. The file reader and the index parser are native only for now.
- The UNet code takes the output channel count from the weight shape in the index. It does not repeat 320 in the code. A weight whose shape does not fit the input stops the run with `WeightShape`.
- The layers split into a plan (checked sizes, no data) and `bind_layer`, which connects a plan to buffers that are already on the GPU. The UNet chains layers this way, with no copy back to the CPU between layers. The old test path uploads the data and then calls `bind_layer`, so the layer test also covers the new code.
- `run_unet` takes a `check` function. After each block, the output goes back to the CPU and `check` sees it. An error from `check` stops the run there. That is the "stop at the first mismatch" rule.
- Each block runs in its own submits. `submit_within_limit` measures each submit and stops the run at 0.25 s. The test prints the submit count and the longest submit of each block.
- The weights of a block are freed after its submit. Only the block output stays.
- The submit error messages now print the limit from the constant. Before, the text said "250 ms" in all cases.

Command: `uv run dump_golden.py`, in the project folder. Device: Mac, Python through `uv run`, CPU. The float32 part of the output:

```
golden/float32:
  text_encoder     shape=(1, 77, 1024) dtype=float32
  unet_input       shape=(1, 4, 64, 64) dtype=float32
  timestep         shape=(1,) dtype=float32
  time_embedding   shape=(1, 1280) dtype=float32
  conv_in          shape=(1, 320, 64, 64) dtype=float32
  down_block_0     shape=(1, 320, 32, 32) dtype=float32
  down_block_1     shape=(1, 640, 16, 16) dtype=float32
  down_block_2     shape=(1, 1280, 8, 8) dtype=float32
  down_block_3     shape=(1, 1280, 8, 8) dtype=float32
  mid_block        shape=(1, 1280, 8, 8) dtype=float32
  up_block_0       shape=(1, 1280, 16, 16) dtype=float32
  up_block_1       shape=(1, 1280, 32, 32) dtype=float32
  up_block_2       shape=(1, 640, 64, 64) dtype=float32
  up_block_3       shape=(1, 320, 64, 64) dtype=float32
  unet_output      shape=(1, 4, 64, 64) dtype=float32
  final_latent     shape=(1, 4, 64, 64) dtype=float32
  decoded_image    shape=(1, 3, 512, 512) dtype=float32
```

The old float32 files did not change. Command, for each old file: `compare/target/release/compare <old copy> golden/float32/<name>.safetensors 0`. Every one of the 14 old tensors printed `max=0.000e0`. `cmp` found `golden/starting_noise.safetensors` identical.

A numpy check of the new input printed:

```
timestep [999.] ratio min 0.997667133808136 max 0.9976673722267151
sigma from ratio 14.614697977636695
```

Calculation: the Euler scheduler scales the noise by sigma / sqrt(sigma^2 + 1). The ratio unet_input / noise is 0.99767 for every value, and r / sqrt(1 - r^2) = 14.6147. That is the sigma of timestep 999.

Command: `cargo test --release --test unet -- --nocapture`, in `engine/`. Device: Mac native.

```
conv_in: largest error 9.5367431640625e-7, mean error 2.2269295524637855e-8, worst at [0, 240, 11, 1] (engine -2.2016876, golden -2.2016885), 1 submits, longest 2.0 ms, peak 10551296 bytes
test unet_matches_golden_block_by_block ... ok
```

A second run printed `longest 1.8 ms` and the same errors.

Calculations:

- conv_in: 320 x 64 x 64 outputs x 36 multiply-adds (4 channels x 9) = 47,185,920 multiply-adds.
- conv_in.weight is 320 x 4 x 3 x 3 x 4 = 46,080 bytes, and conv_in.bias is 1,280 bytes.
- Peak = latent + output + read-back copy = 65,536 + 5,242,880 + 5,242,880 = 10,551,296 bytes. The latent is 4 x 64 x 64 x 4 bytes, and the output is 320 x 64 x 64 x 4 bytes. That is the printed peak. The weights are not in it, because they are freed before the read-back.

Command: `cargo test --release --test layers -- --nocapture` after the layer refactor. Device: Mac native. The golden dump ran on the CPU at the same time, so the speeds are not valid. Only the errors count here, and they are the same as in "Linear and convolution kernels".

```
linear: 10.04 GB/s median, 11.18 GB/s best (30 submits, 64 runs per submit, 69.5 ms median submit, 102.9 ms longest), largest error 1.1920929e-6
conv1x1: 6.76 GB/s median, 9.93 GB/s best (30 submits, 128 runs per submit, 90.1 ms median submit, 96.2 ms longest), largest error 1.9073486e-6
conv3x3: 0.47 GB/s median, 0.53 GB/s best (30 submits, 2 runs per submit, 59.7 ms median submit, 65.3 ms longest), largest error 6.4373016e-6
conv3x3_stride2: 0.72 GB/s median, 0.77 GB/s best (30 submits, 4 runs per submit, 57.2 ms median submit, 62.7 ms longest), largest error 6.198883e-6
```

`cargo test --release --lib` printed `test result: ok. 12 passed`. The WASM build, `cargo build --release --target wasm32-unknown-unknown`, gave no warnings.

Mutation checks. Each bug made the test fail, and the code was restored:

```
every weight read 4 bytes too far:
conv_in: largest error 3.966747283935547e0, mean error 3.7506689689694916e-1, worst at [0, 238, 38, 40] (engine 2.4368927, golden -1.5298545), 1 submits, longest 2.8 ms, peak 10551296 bytes
conv_in: first mismatch, largest error 3.966747283935547e0 is above 1e-4

submit limit set to 1 microsecond, old message:
conv_in: one submit took 1.8 ms, the limit is 250 ms

submit limit set to 1 microsecond, new message:
conv_in: one submit took 1.9 ms, the limit is 0.0 ms
```

Failed runs: the first WASM build warned that `WeightIndex`, `WeightEntry` and two constants were unused, because only the native file reader uses them. They are now native only.

Not measured on the iPhone. The 0.25 s check ran on the Mac only. The iPhone GPU is slower, so each later block needs its own iPhone run.

Conclusion: the engine reads UNet weights from the flat file, runs conv_in on the GPU, and matches PyTorch to 9.5e-7. The run stops at the first block that does not match.

Next step: the time embedding (sinusoidal timestep features, then linear, SiLU, linear) and its compare with `time_embedding`. After that, the down blocks need new kernels: GroupNorm, residual add, LayerNorm, GEGLU, a channel transpose, and attention projections without bias. The block types and channel counts will come from `unet/config.json`, which the converter will copy next to `unet.bin`.

## 2026-11-18: Float16 golden set after the UNet hooks

This entry completes "UNet step 1, weight loader and conv_in". That entry has only the float32 part of `uv run dump_golden.py`, because the float16 part was still running.

Device: Mac, Python through `uv run`, CPU. The float16 part of the output:

```
golden/float16:
  text_encoder     shape=(1, 77, 1024) dtype=float16
  unet_input       shape=(1, 4, 64, 64) dtype=float16
  timestep         shape=(1,) dtype=float32
  time_embedding   shape=(1, 1280) dtype=float16
  conv_in          shape=(1, 320, 64, 64) dtype=float16
  down_block_0     shape=(1, 320, 32, 32) dtype=float16
  down_block_1     shape=(1, 640, 16, 16) dtype=float16
  down_block_2     shape=(1, 1280, 8, 8) dtype=float16
  down_block_3     shape=(1, 1280, 8, 8) dtype=float16
  mid_block        shape=(1, 1280, 8, 8) dtype=float16
  up_block_0       shape=(1, 1280, 16, 16) dtype=float16
  up_block_1       shape=(1, 1280, 32, 32) dtype=float16
  up_block_2       shape=(1, 640, 64, 64) dtype=float16
  up_block_3       shape=(1, 320, 64, 64) dtype=float16
  unet_output      shape=(1, 4, 64, 64) dtype=float16
  final_latent     shape=(1, 4, 64, 64) dtype=float16
  decoded_image    shape=(1, 3, 512, 512) dtype=float16
```

`timestep` is float32 in both sets, because the pre-hook converts it with `.float()`. Every other entry is float16.

The old float16 files did not change. Command, for each old file: `compare/target/release/compare <old copy> golden/float16/<name>.safetensors 0`. Every one of the 14 old tensors printed `max=0.000e0`.

`ps` showed the process still running after 64 minutes. The float16 files have the time 2026-11-18 00:30. The total run time was not measured. Nothing in the engine reads the float16 set yet.

Not measured on the iPhone.

## 2026-11-18: TAESD decoder and its golden set

The user asked for three things in one prompt: the TAESD decoder in Rust, its golden output from diffusers `AutoencoderTiny`, and one web page from prompt to image. This step does the first two. The web page is not in this step, because the engine has no tokenizer and no text encoder yet, and the UNet has only conv_in.

Files added:

- `dump_taesd_golden.py`: decodes `golden/float32/final_latent` with `AutoencoderTiny` (`madebyollin/taesd`) and saves 9 tensors, `golden/float32/taesd_manifest.json` and `golden/float32/taesd_image.png`.
- `engine/src/activation.rs`, `engine/src/block.rs`, `engine/src/conv.rs`: the activation type, the block report and the 3x3 convolution binder. They moved out of `engine/src/unet/`, because the UNet and TAESD both use them.
- `engine/src/taesd/`: `mod.rs` (the decoder in order), `steps.rs` (the parts and their submits), `kernels.rs`, and 5 shaders: `clamp.wgsl`, `relu.wgsl`, `add_relu.wgsl`, `upsample.wgsl`, `image.wgsl`.
- `engine/tests/common/mod.rs`: the golden check, moved out of `engine/tests/unet.rs`. `engine/tests/taesd.rs`: the TAESD test.

Files changed: `convert_weights.py` (a `taesd` network from its own model, `hf_hub_download` in place of `snapshot_download`), `engine/src/unet/mod.rs`, `engine/src/unet/conv.rs` (deleted, moved to `engine/src/conv.rs`), `engine/src/weights/mod.rs` (`upload_weight` removed, no caller left), `engine/src/layers/mod.rs`, `engine/src/dispatch.rs` (`Dispatch` is `Clone`), `engine/src/lib.rs`, `engine/tests/unet.rs`.

Design choices:

- The decoder follows diffusers `DecoderTiny`: `tanh(x / 3) * 3`, then `decoder.layers.0` to `decoder.layers.18`, then `x * 2 - 1`. The Rust code names each layer by its position in `decoder.layers`, which is also the weight name.
- A TAESD block is `ReLU(conv(ReLU(conv(ReLU(conv(x))))) + x)`. Every convolution has its own submit, with the ReLU or the add + ReLU in the same submit. The reason: one 64-channel convolution at 512 x 512 is 19.3 G operations, so three in one submit would be the longest submit of the engine.
- ReLU and add + ReLU work in place. The reason: one activation at 512 x 512 x 64 is 67 MB, and a second buffer per kernel would double the peak.
- The convolutions after each upsample have `bias=False`. `conv.rs` takes `Bias::Zero` and gives the kernel a buffer of zeros. The existing conv3x3 kernel did not change.
- The last convolution has 3 output channels, and the kernel computes 4 channels at a time. `conv.rs` pads the weights and the bias with zeros to 4 channels. `image.wgsl` reads only the first 3 channels, which are the first 3 x 512 x 512 values.
- The small kernels use one vec4 per invocation and 256 invocations per workgroup. At 512 x 512 x 64 that is 16,384 workgroups. With 64 invocations and one value each, it would be 262,144, above the limit of 65,535.
- The golden set is float32 only, in a separate script. The reason: `dump_golden.py` took more than 64 minutes, and TAESD only needs `final_latent`.

Command: `uv run dump_taesd_golden.py`, in the project folder. Device: Mac, Python through `uv run`, CPU.

```
golden/float32:
  taesd_conv_in    shape=(1, 64, 64, 64) dtype=float32
  taesd_blocks_0   shape=(1, 64, 64, 64) dtype=float32
  taesd_upsample_0 shape=(1, 64, 128, 128) dtype=float32
  taesd_blocks_1   shape=(1, 64, 128, 128) dtype=float32
  taesd_upsample_1 shape=(1, 64, 256, 256) dtype=float32
  taesd_blocks_2   shape=(1, 64, 256, 256) dtype=float32
  taesd_upsample_2 shape=(1, 64, 512, 512) dtype=float32
  taesd_blocks_3   shape=(1, 64, 512, 512) dtype=float32
  taesd_image      shape=(1, 3, 512, 512) dtype=float32
taesd_image vs decoded_image: largest difference 0.9587, mean difference 0.0862
```

The difference line compares TAESD with the full SD-Turbo VAE on the [-1, 1] scale. TAESD is an approximation, so it is not 0. `taesd_image.png` and `image.png` show the same cat on the same table. TAESD has a little more contrast and sharper fur.

Command: `uv run convert_weights.py taesd`. Device: Mac.

```
taesd: 134 tensors, 9780736 bytes, float32, largest decoder.layers.11.weight 147456 bytes, read back equal
```

The file also holds the encoder. The 67 decoder tensors are 4,890,124 bytes.

The converter change must not change the SD-Turbo files. Command: `md5 -q weights/vae.bin weights/vae.json` before and after `uv run convert_weights.py vae`. The two hashes were the same, and the converter printed:

```
vae: 248 tensors, 334616576 bytes, float32, largest decoder.mid_block.resnets.0.conv1.weight 9437184 bytes, read back equal
```

Command: `cargo test --release --test taesd -- --nocapture`, in `engine/`. Device: Mac native. The second run:

```
taesd_conv_in: largest error 3.5762786865234375e-7, mean error 7.693369365180391e-9, worst at [0, 29, 27, 36] (engine 1.0526974, golden 1.0526971), 1 submits, longest 2.4 ms, peak 2097152 bytes
taesd_blocks_0: largest error 7.152557373046875e-7, mean error 2.206338905013359e-8, worst at [0, 0, 3, 10] (engine 0.77598745, golden 0.77598673), 9 submits, longest 3.3 ms, peak 3293488 bytes
taesd_upsample_0: largest error 8.344650268554688e-7, mean error 4.9740736324253626e-8, worst at [0, 26, 35, 19] (engine 0.29395735, golden 0.29395652), 1 submits, longest 5.5 ms, peak 9584944 bytes
taesd_blocks_1: largest error 1.2516975402832031e-6, mean error 4.2683773871434204e-8, worst at [0, 35, 57, 52] (engine 0.35247242, golden 0.35247368), 9 submits, longest 7.4 ms, peak 12730672 bytes
taesd_upsample_1: largest error 1.5497207641601563e-6, mean error 9.625100853339526e-8, worst at [0, 48, 111, 145] (engine 1.2396799, golden 1.2396784), 1 submits, longest 14.7 ms, peak 37896496 bytes
taesd_blocks_2: largest error 9.626150131225586e-6, mean error 6.80045430784633e-8, worst at [0, 33, 212, 253] (engine 0.19404295, golden 0.19403332), 9 submits, longest 10.6 ms, peak 50479408 bytes
taesd_upsample_2: largest error 3.866851329803467e-6, mean error 1.394131174756538e-7, worst at [0, 58, 124, 216] (engine 0.034366712, golden 0.03437058), 1 submits, longest 55.4 ms, peak 151142704 bytes
taesd_blocks_3: largest error 1.6063451766967773e-5, mean error 3.397037897636887e-8, worst at [0, 41, 87, 199] (engine 0.09152818, golden 0.09154424), 3 submits, longest 45.6 ms, peak 201474352 bytes
taesd_image: largest error 7.867813110351563e-6, mean error 2.493081107483401e-7, worst at [0, 2, 87, 199] (engine -0.00572145, golden -0.005713582), 1 submits, longest 5.2 ms, peak 201474352 bytes
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.61s
```

The first run printed the same errors. Its longest submits were 3.0, 6.4, 3.7, 5.9, 23.1, 15.0, 68.9, 42.6 and 4.1 ms.

The tolerance is 1e-3 for every part. It was set before the first run. The largest error is 1.6e-5, in `taesd_blocks_3`.

Calculations:

- Multiply-adds per convolution = in channels x out channels x 9 x pixels.
- 64 x 64: 4 x 64 x 9 x 4096 = 9,437,184, plus 9 block convolutions x 150,994,944 = 1,358,954,496.
- 128 x 128: 10 convolutions x 603,979,776 = 6,039,797,760.
- 256 x 256: 10 convolutions x 2,415,919,104 = 24,159,191,040.
- 512 x 512: 4 convolutions x 9,663,676,416 = 38,654,705,664, plus the last convolution, 64 x 3 x 9 x 262,144 = 452,984,832.
- Total: 70,675,070,976 multiply-adds = 141,350,141,952 operations. One 64-channel convolution at 512 x 512 is 19,327,352,832 operations.
- The longest submit of `taesd_blocks_3` is one 512 x 512 convolution with its ReLU: 19,327,352,832 / 0.0456 s = about 424 G ops/s. The conv3x3 case in "Linear and convolution kernels" ran at about 440 G ops/s on Mac native.
- Peak of `taesd_blocks_3` = 3 activations at 512 x 512 x 64 + one convolution's weights and sizes = 3 x 67,108,864 + 147,760 = 201,474,352 bytes. The 3 activations are the block input, the second convolution output, and the third convolution output.
- Peak of `taesd_upsample_2` = input at 256 x 256 + upsampled copy + convolution output + weights and sizes = 16,777,216 + 67,108,864 + 67,108,864 + 147,760 = 151,142,704 bytes.

Mutation checks. Each bug made the test fail, and `cmp` showed each shader restored:

```
clamp.wgsl, output = input (no tanh):
taesd_conv_in: largest error 3.5013678669929504e0, mean error 2.2761314266360917e-3, worst at [0, 60, 1, 0] (engine 4.344418, golden 0.8430502), 1 submits, longest 2.6 ms, peak 2097152 bytes
taesd_conv_in: first mismatch, largest error 3.5013678669929504e0 is above 1e-3

add_relu.wgsl, skip multiplied by 0:
taesd_blocks_0: largest error 1.731501217931509e0, mean error 4.139686098809496e-2, worst at [0, 13, 1, 0] (engine 0.023814086, golden 1.7553153), 9 submits, longest 2.8 ms, peak 3293488 bytes
taesd_blocks_0: first mismatch, largest error 1.731501217931509e0 is above 1e-3

upsample.wgsl, left and right pixels swapped:
taesd_upsample_0: largest error 2.2390787303447723e0, mean error 6.623470668570569e-2, worst at [0, 25, 55, 48] (engine -0.39075682, golden 1.8483219), 1 submits, longest 8.2 ms, peak 9584944 bytes
taesd_upsample_0: first mismatch, largest error 2.2390787303447723e0 is above 1e-3
```

The other tests after the refactor, Mac native: `cargo test --release --lib` printed `test result: ok. 14 passed`. `cargo test --release --test unet -- --nocapture` printed the same conv_in errors as before:

```
conv_in: largest error 9.5367431640625e-7, mean error 2.2269295524637855e-8, worst at [0, 240, 11, 1] (engine -2.2016876, golden -2.2016885), 1 submits, longest 2.1 ms, peak 10551296 bytes
```

The layers, silu and attention tests passed.

Failed runs:

- The first WASM build, `cargo build --release --target wasm32-unknown-unknown`, failed with `error: this arithmetic operation will overflow` in `engine/src/taesd/kernels.rs`. The cause: `u32::MAX as usize * 4` overflows, because usize is 32 bits in WASM. The limit now uses `saturating_mul`. The second build finished with no warnings.
- One run of `cargo test --release --test silu --test attention` printed `test result: FAILED. 0 passed; 1 failed` for the attention test, after 3.80 s. The output was filtered, so the failure message is lost. Four later runs of the attention test passed. The cause is not known. The TAESD test ran on the GPU just before it.

Not measured on the iPhone. The longest Mac native submit is 55.4 ms. The iPhone conv3x3 rate was below the Mac rate in "Layer kernels on the iPhone", so each 512 x 512 submit needs an iPhone run.

Conclusion: the engine decodes the SD-Turbo latent with TAESD on the GPU and matches diffusers `AutoencoderTiny` to 1.6e-5, with 35 submits and a peak of 201 MB of engine buffers.

Next step: a TAESD web page that fetches `weights/taesd.bin` and `golden/float32/final_latent`, runs the decoder in WASM, and draws the image on a canvas. That page needs a browser weight reader, which does not exist yet. A page from prompt to image needs the tokenizer, the text encoder and the full UNet first.

## 2026-11-18: TAESD web page

Files added:

- `engine/src/weights/in_memory.rs`: `WeightBytes`, a weight reader over a whole .bin file in memory. The browser fetches `weights/taesd.bin` and `weights/taesd.json` and passes them in.
- `engine/src/image.rs`: `image_to_rgba`, from [3][height][width] in [-1, 1] to RGBA bytes for a canvas.
- `engine/src/taesd/report.rs`: `decode_and_report` runs the decoder, writes one line per part, compares the image with the diffusers image, and makes the RGBA bytes.
- `engine/examples/taesd.rs`: the same report on Mac native, with the same in-memory reader as the page.
- `engine/web/taesd.html`: the page.

Files changed: `engine/src/weights/mod.rs` (the index parser is now built for WASM too), `engine/src/block.rs` (`check_block` returns the values it read), `engine/src/taesd/mod.rs` (`run_taesd` returns the image), `engine/src/web.rs` (`decodeTaesd` and `TaesdResult`), `engine/src/submit.rs`, `engine/src/gpu.rs` (comments), `engine/src/lib.rs`.

Design choices:

- `WeightBytes` keeps the whole file in WASM memory. TAESD is 9,780,736 bytes, so this is small. The UNet (3,463,643,136 bytes) cannot work this way, because one WASM memory is at most 4 GB and the tab limit is about 1920 MB. The UNet needs a different reader later.
- The page fetches `final_latent` and `taesd_image` from `golden/float32/`. It does not fetch the 8 inner golden tensors, because two of them are 67 MB each. The page compares only the final image. The native test compares every part.
- The page draws the engine image on a canvas, with the diffusers PNG under it for comparison.
- The RGBA conversion is the diffusers postprocess: `(x / 2 + 0.5)`, clamp to [0, 1], times 255, round.
- The page is at `engine/web/taesd.html` and reads `../../weights/` and `../../golden/`. The server must serve the project folder, not `engine/`. The server that was already running on port 8000 serves `engine/`, so it cannot serve this page.
- `submit_within_limit` now calls `gpu.finish_uploads()` before each timed submit. The reason is in "Failed runs" below.

Command, Mac native: `cargo run --release --example taesd`, in `engine/`. This run is with the wait before each submit:

```
taesd_conv_in: 1 submits, longest 2.2 ms, peak 2097152 bytes
taesd_blocks_0: 9 submits, longest 2.0 ms, peak 3293488 bytes
taesd_upsample_0: 1 submits, longest 3.5 ms, peak 9584944 bytes
taesd_blocks_1: 9 submits, longest 4.7 ms, peak 12730672 bytes
taesd_upsample_1: 1 submits, longest 14.9 ms, peak 37896496 bytes
taesd_blocks_2: 9 submits, longest 13.2 ms, peak 50479408 bytes
taesd_upsample_2: 1 submits, longest 44.8 ms, peak 151142704 bytes
taesd_blocks_3: 3 submits, longest 44.2 ms, peak 201474352 bytes
taesd_image: 1 submits, longest 2.6 ms, peak 201474352 bytes
taesd_image: largest error 7.867813e-6 against diffusers
total 481.6 ms from latent upload to image on the CPU, with 9 read-backs
```

The first run, before the wait, printed `total 806.0 ms` and the same error.

Page, Mac Chromium: `python3 -m http.server 8765 --bind 127.0.0.1` in the project folder, then `http://127.0.0.1:8765/engine/web/taesd.html` in Playwright Chromium. The run with the wait, on a browser that had loaded the page before:

```
taesd_conv_in: 1 submits, longest 2.7 ms, peak 2097152 bytes
taesd_blocks_0: 9 submits, longest 4.0 ms, peak 3293488 bytes
taesd_upsample_0: 1 submits, longest 9.2 ms, peak 9584944 bytes
taesd_blocks_1: 9 submits, longest 10.3 ms, peak 12730672 bytes
taesd_upsample_1: 1 submits, longest 26.9 ms, peak 37896496 bytes
taesd_blocks_2: 9 submits, longest 31.6 ms, peak 50479408 bytes
taesd_upsample_2: 1 submits, longest 79.5 ms, peak 151142704 bytes
taesd_blocks_3: 3 submits, longest 47.7 ms, peak 201474352 bytes
taesd_image: 1 submits, longest 3.3 ms, peak 201474352 bytes
taesd_image: largest error 7.867813e-6 against diffusers
total 769.6 ms from latent upload to image on the CPU, with 9 read-backs
```

The canvas shows the cat. A script on the page compared the canvas bytes with the diffusers PNG:

```
bytes 1048576, differing 16, largest byte difference 1
```

Calculation: 16 / 1,048,576 = 0.0015% of the bytes differ, each by 1. The image error of 7.9e-6 on the [-1, 1] scale is 7.9e-6 x 127.5 = 0.001 of one byte step. A value that close to a half step can round the other way. That explains a difference of 1 in a few bytes.

Failed runs:

- The first page load in Chromium failed: `FAILED: taesd conv_in: one submit took 590.9 ms, the limit is 250.0 ms`. The same submit takes 2.2 ms on Mac native. The hypothesis: the browser compiles the 3 new shaders, and copies the weight and latent buffers, inside the first timed submit. These are not GPU kernel time, so the wait now runs before each timed submit, as in "The first iPhone benchmark failed".
- Check of the fix: the page with the wait passed, with `taesd_conv_in` at 3.6 ms. Then a build without the wait also passed on the same browser, with 4.0 ms. A page close and a new load without the wait passed again, with 6.2 ms. So the 590.9 ms happened only on the first load of the browser. It was probably a cold shader compile. Playwright keeps the browser between page loads, so a cold load could not be made again. The fix is not proven. The first iPhone load is a cold load, so it is the real test.

Unit tests: `cargo test --release --lib` printed `test result: ok. 18 passed`. The new tests check the in-memory reader (bytes at the offset, a file that is too short) and the RGBA conversion (interleaving, clamping, rounding, a wrong length). The taesd, unet, layers, silu and attention tests passed. `cargo build --release --target wasm32-unknown-unknown` had no warnings.

Not measured on the iPhone.

Next step: the user opens the page on the iPhone through ngrok, on a cold load and a second load, and sends both outputs.

## 2026-11-18: TAESD web page on the iPhone

Page: `engine/web/taesd.html`, served from the project folder with `python3 -m http.server 8765` and opened through `ngrok http 8765`. Device: iPhone. The user sent one output. The user did not say whether it was the first (cold) load or a reload, or how the two images look.

```
taesd_conv_in: 1 submits, longest 3.0 ms, peak 2097152 bytes
taesd_blocks_0: 9 submits, longest 5.0 ms, peak 3293488 bytes
taesd_upsample_0: 1 submits, longest 13.0 ms, peak 9584944 bytes
taesd_blocks_1: 9 submits, longest 8.0 ms, peak 12730672 bytes
taesd_upsample_1: 1 submits, longest 23.0 ms, peak 37896496 bytes
taesd_blocks_2: 9 submits, longest 19.0 ms, peak 50479408 bytes
taesd_upsample_2: 1 submits, longest 66.0 ms, peak 151142704 bytes
taesd_blocks_3: 3 submits, longest 71.0 ms, peak 201474352 bytes
taesd_image: 1 submits, longest 13.0 ms, peak 201474352 bytes
taesd_image: largest error 6.735325e-6 against diffusers
total 1218.0 ms from latent upload to image on the CPU, with 9 read-backs
```

Calculations:

- The longest submit is 71.0 ms, in `taesd_blocks_3`. That is 71.0 / 250 = 28% of the project limit.
- That submit is one 64-channel convolution at 512 x 512 with its ReLU or add + ReLU: 19,327,352,832 operations / 0.071 s = about 272 G ops/s.
- The conv3x3 case in "Layer kernels on the iPhone" ran at about 346 G ops/s. 272 / 346 = about 79% of that rate. That case had 320 channels at 64 x 64. Here, the 3 x 3 window covers 64 channels at 512 x 512, so each invocation loops over fewer input channels, and the output is 4 times larger.
- The largest image error is 6.7e-6 on the iPhone and 7.9e-6 on Mac native and Mac Chromium. The GPUs round in a different order. Both are far below the 1e-3 tolerance.
- The total is 1218.0 / 769.6 = 1.58 times the Mac Chromium total. The total includes shader compiles, weight uploads and 9 read-backs, so it is not GPU time alone.
- The engine buffers peak at 201,474,352 bytes. That is 201.5 / 1920 = about 10.5% of the tab limit. The JS heap and WASM memory are not in this number.
- The peaks are the same as on the Mac, byte for byte. They count only the engine buffers, and those are the same on every device.

Not known yet: whether this was a cold load. If it was, the wait before each timed submit (see "TAESD web page") kept the first submit at 3.0 ms on a cold load.

Conclusion: the TAESD decoder runs on the iPhone, matches diffusers to 6.7e-6, and its longest submit is 71.0 ms. Each 512 x 512 convolution has its own submit, so there is room under the 250 ms limit.

## 2026-11-18: TAESD web page on the iPhone, load order and reload

This entry completes "TAESD web page on the iPhone". The user said that the output in that entry was the very first load of the page. So it was a cold load. The user sent the output of a reload, on the same iPhone and the same page:

```
taesd_conv_in: 1 submits, longest 4.0 ms, peak 2097152 bytes
taesd_blocks_0: 9 submits, longest 6.0 ms, peak 3293488 bytes
taesd_upsample_0: 1 submits, longest 14.0 ms, peak 9584944 bytes
taesd_blocks_1: 9 submits, longest 7.0 ms, peak 12730672 bytes
taesd_upsample_1: 1 submits, longest 20.0 ms, peak 37896496 bytes
taesd_blocks_2: 9 submits, longest 17.0 ms, peak 50479408 bytes
taesd_upsample_2: 1 submits, longest 83.0 ms, peak 151142704 bytes
taesd_blocks_3: 3 submits, longest 65.0 ms, peak 201474352 bytes
taesd_image: 1 submits, longest 5.0 ms, peak 201474352 bytes
taesd_image: largest error 6.735325e-6 against diffusers
total 835.0 ms from latent upload to image on the CPU, with 9 read-backs
```

Calculations:

- The longest submit of the reload is 83.0 ms, in `taesd_upsample_2` (upsample and one convolution at 512 x 512). That is 83.0 / 250 = 33% of the limit. Over both loads, the longest submit is 83.0 ms.
- The longest submit of `taesd_blocks_3` is 65.0 ms: 19,327,352,832 operations / 0.065 s = about 297 G ops/s. On the cold load it was 71.0 ms, about 272 G ops/s.
- The total fell from 1218.0 ms (cold) to 835.0 ms (reload), 383.0 ms less. The time was not split into parts, so the cause is not measured. A shader cache in Safari is a possible cause.
- The reload total is 835.0 / 769.6 = 1.08 times the Mac Chromium total.
- The image error is 6.735325e-6 on both loads.

The cold load: with the wait before each timed submit, the first submit on a cold iPhone load took 3.0 ms, and the run passed. There is no cold iPhone load without the wait, so this does not prove that the wait was needed on the iPhone. It proves that the page passes on a cold load with the wait.

Not reported: how the engine image and the diffusers image look side by side on the iPhone.

Conclusion: TAESD passes on the iPhone on a cold load and on a reload. The longest submit is 83.0 ms, and the error is 6.7e-6.

Next step: the UNet time embedding (sinusoidal timestep features, then linear, SiLU, linear) and its compare with `time_embedding`.

## 2026-11-18: TAESD images on the iPhone, side by side

This entry completes "TAESD web page on the iPhone, load order and reload". That entry says that the look of the two images was not reported. The user reported that on the iPhone, the engine image (canvas) and the diffusers image (`golden/float32/taesd_image.png`) looked the same.

This is a check by eye. The byte compare of the two images ran only in Mac Chromium, in "TAESD web page": 16 of 1,048,576 bytes differed, each by 1.

Device: iPhone. No new command ran.

## 2026-11-18: UNet weight rounding, one layer at a time

Files: `quant_sensitivity.py` (new) and `quant_sensitivity.txt` (new, the full output of the run).

Design choices:

- The script rounds only one layer at a time. It rounds the layer to 4 or 8 bits, runs the pipeline, and puts the float16 weight back. So each row shows the cost of that one layer alone.
- A layer is a `Conv2d` or a `Linear` module of the UNet, 282 layers in total. Norm weights and biases stay float16, because they are small.
- The rounding is symmetric, with one scale per output channel. The scale is the largest absolute value of the row, divided by 7 (4 bits) or 127 (8 bits). The script multiplies the rounded integers back by the scale, so the math stays float16 and only the weight values change.
- The run uses the Mac GPU through PyTorch ("mps"), not the CPU. One float16 CPU run took more than 64 minutes (see "Float16 golden set after the UNet hooks"). One GPU run takes about 0.19 s, and 564 runs are needed.
- The reference for every error is the float16 GPU run with no rounded layer, not `golden/float16/final_latent`. The GPU and the CPU differ by up to 6.25e-2 (see below). That difference is larger than most 8-bit errors, so it would hide them.
- The script reads the text encoder output from `golden/float16/text_encoder.safetensors`, because it is the same for every run.
- The table sorts by the root mean square (rms) error over all 16,384 latent values. The largest error alone jumps in float16 steps (0.015625 near 16), so many layers tie on it.

Command: `uv run quant_sensitivity.py > quant_sensitivity.txt`, in the project folder. Device: Mac, Python through `uv run`, PyTorch on the Mac GPU ("mps"). This runtime is not in the table, because it does not use wgpu. Run time: `2:40.23 total`.

The first 3 lines, then the top 11 and the last 5 rows of each table. All 282 rows of each table are in `quant_sensitivity.txt`.

```
282 layers, 865466880 weights
baseline repeat: max 0.000e+00
baseline vs golden/float16 final_latent: max 6.250e-02, rms 6.967e-03
4-bit, one layer at a time, sorted by rms error of the final latent:
rank  layer                                                                       weights        rms        max
   1  conv_out                                                                      11520  5.595e-01  2.336e+00
   2  conv_in                                                                       11520  2.803e-01  2.035e+00
   3  up_blocks.2.resnets.0.conv_shortcut                                         1228800  2.623e-01  2.875e+00
   4  up_blocks.1.resnets.0.conv1                                                29491200  2.329e-01  3.109e+00
   5  down_blocks.0.resnets.0.time_emb_proj                                        409600  2.259e-01  2.297e+00
   6  down_blocks.0.resnets.0.conv2                                                921600  2.076e-01  2.031e+00
   7  down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0              1638400  2.043e-01  2.797e+00
   8  down_blocks.1.resnets.0.conv2                                               3686400  1.908e-01  2.844e+00
   9  down_blocks.0.attentions.1.proj_out                                          102400  1.881e-01  3.156e+00
  10  time_embedding.linear_2                                                     1638400  1.822e-01  3.086e+00
  11  up_blocks.3.resnets.2.conv_shortcut                                          204800  1.710e-01  8.750e-01
 278  up_blocks.0.resnets.0.time_emb_proj                                         1638400  4.526e-03  3.125e-02
 279  mid_block.attentions.0.transformer_blocks.0.attn1.to_q                      1638400  4.510e-03  3.125e-02
 280  mid_block.attentions.0.transformer_blocks.0.attn1.to_v                      1638400  4.485e-03  6.641e-02
 281  mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0                  1638400  4.401e-03  5.859e-02
 282  mid_block.attentions.0.transformer_blocks.0.attn2.to_q                      1638400  4.155e-03  3.125e-02
8-bit, one layer at a time, sorted by rms error of the final latent:
rank  layer                                                                       weights        rms        max
   1  conv_out                                                                      11520  3.149e-02  1.289e-01
   2  down_blocks.1.attentions.1.proj_out                                          409600  2.606e-02  1.027e+00
   3  up_blocks.2.resnets.0.conv_shortcut                                         1228800  1.944e-02  2.188e-01
   4  conv_in                                                                       11520  1.747e-02  1.250e-01
   5  down_blocks.1.resnets.1.conv2                                               3686400  1.366e-02  5.234e-01
   6  down_blocks.0.resnets.0.conv2                                                921600  1.307e-02  1.699e-01
   7  down_blocks.0.resnets.0.time_emb_proj                                        409600  1.270e-02  2.031e-01
   8  down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k                  1310720  1.242e-02  2.402e-01
   9  down_blocks.2.attentions.1.proj_in                                          1638400  1.160e-02  3.125e-01
  10  down_blocks.1.attentions.1.proj_in                                           409600  1.155e-02  2.188e-01
  11  up_blocks.3.resnets.2.conv_shortcut                                          204800  1.069e-02  4.688e-02
 278  up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v                     327680  2.953e-03  3.125e-02
 279  up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q                     102400  2.855e-03  3.125e-02
 280  up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q                     102400  2.813e-03  3.125e-02
 281  up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0                 102400  2.795e-03  3.125e-02
 282  up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k                     327680  2.671e-03  3.125e-02
```

Checks before the run, with a probe script in the scratchpad (not in the project). Device: the same Mac GPU.

- Three float16 GPU runs: `2.96358585357666`, `0.18961286544799805`, `0.1926860809326172` seconds. The first run includes the shader setup. The repeats printed `repeat diff 0.0`.
- All UNet parameters: 865,910,724. The largest absolute value of the golden float16 final latent: 16.953125.
- `fake_quantize` on a random 5 x 300 float16 tensor. 4 bits gave 13 to 15 different values per row, and 15 is the most that -7 to 7 allows. The largest change per row, divided by the row maximum, was 0.0706 to 0.0713, and 1 / (2 x 7) = 0.0714. 8 bits gave 0.0039 to 0.0040, and 1 / (2 x 127) = 0.0039.

Calculations:

- The rounded layers hold 865,466,880 of 865,910,724 UNet parameters: 99.95%.
- The GPU baseline is 6.25e-2 from the CPU golden latent at most. Near 16, one float16 step is 2^4 x 2^-10 = 0.015625. So 6.25e-2 is 4 steps.
- 4 bits: 47 of 282 layers have rms above 0.1. They hold 210,536,960 weights: 210,536,960 / 865,466,880 = 24.3%. The median rms is 4.462e-02. The worst layer is `conv_out` (rms 5.595e-01, max 2.336), and the second is `conv_in`. Each of the two holds only 11,520 weights.
- 8 bits: no layer has rms above 0.05. The median rms is 5.173e-03. That is less than the 6.967e-03 rms between the GPU and the CPU float16 runs. The worst layer is again `conv_out` (rms 3.149e-02). 185 of 282 layers have a largest error of 6.25e-2 or less.
- 8 bits had a lower rms than 4 bits for 281 of the 282 layers. The one exception is `mid_block.attentions.0.transformer_blocks.0.attn2.to_q`: 4.155e-03 at 4 bits, 4.282e-03 at 8 bits. Both are at the float16 noise level of this run.
- The medians and counts come from a short Python count over the rows of `quant_sensitivity.txt`.

Rejected:

- The CPU float16 run, because one run took more than 64 minutes.
- The CPU golden latent as the reference, because its 6.25e-2 offset is larger than most 8-bit errors.

Limits of this result:

- Each row rounds one layer only. The errors of many rounded layers do not simply add, so the error of a fully rounded UNet is not measured.
- The run used one prompt and one noise.
- The scale stays float32 here. The engine format for the scale is not decided.

Not measured on the iPhone.

Conclusion: at 8 bits, every layer alone stays within 3.2e-2 rms of float16. At 4 bits, a quarter of the weights move the latent by more than 0.1 rms, and the first and last convolutions are the most sensitive.

Next step: round all layers at once (all 8 bits, then 4 bits with the most sensitive layers kept at 8 bits), and compare the final latent and the decoded image.

## 2026-11-18: Mixed 4-bit and 8-bit plan, and a 20-image test

Files:

- `quantize.py` (new): `fake_quantize` and `quantizable_layers`, moved out of `quant_sensitivity.py`, so that three scripts share them.
- `quant_sensitivity.py` (changed): imports both functions from `quantize.py`. The math did not change.
- `quant_plan.py` (new): picks 4 or 8 bits per layer under a size limit, and writes `quant_plan.json` (new).
- `quant_eval.py` (new): makes 20 images for 3 versions, prints PSNR and LPIPS, and writes `quant_eval/grid.png` (new). Its output is in `quant_eval.txt` (new).
- `quant_eval/noise.safetensors` (new): one noise per prompt, seeds 0 to 19.

Design choices:

- The user set the size limit to 700 MB (700,000,000 bytes) for the UNet weights.
- The size counts the rounded weights (half a byte at 4 bits, one byte at 8 bits), one float16 scale per output channel, and every other UNet tensor (biases, norms) in float16. The scale and the other tensors take the same bytes in every plan.
- The cost of a layer is its rms error from `quant_sensitivity.txt`, squared. Errors that do not depend on each other add in squares, so the plan minimizes the sum of the squared rms. This assumes that the layer errors add. "UNet weight rounding, one layer at a time" did not test that.
- `scipy.optimize.milp` picks the plan. It solves the yes-or-no choice for all 282 layers exactly, so the result is the best plan for this cost, not a greedy guess.
- The 20 prompts are fixed in `quant_eval.py` and cover animals, people, rooms, landscapes, objects, text and art styles. The noise of image i comes from seed i and is saved on the first run.
- The UNet and the text encoder run in float16 on the Mac GPU. The VAE decodes in float32, so the decoder adds no float16 error of its own. The text encoder is not rounded, so all versions share the same text embeddings.
- PSNR uses the 8-bit images, with 255 as the peak. LPIPS uses the default AlexNet model of the `lpips` package. Lower LPIPS means closer, and 0 means equal.

Command 1: `uv run quant_plan.py 700`, in the project folder. Device: Mac, Python through `uv run`, CPU.

```
limit 700000000 bytes, plan 699997456 bytes, all 4-bit 434257936 bytes, all 8-bit 866991376 bytes
8-bit layers 228 of 282, 8-bit weights 531479040 of 865466880
predicted rms: all 4-bit 1.369e+00, plan 1.618e-01, all 8-bit 1.123e-01
wrote quant_plan.json
```

Command 2: `uv run quant_plan.py 400`, to check a limit below the all-4-bit size:

```
no plan fits 400000000 bytes: The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)
```

Command 3: `uv run quant_eval.py > quant_eval.txt`, in the project folder. Device: Mac, Python through `uv run`, PyTorch on the Mac GPU ("mps"). LPIPS ran on the CPU. Run time: `2:50.71 total`. The first run downloaded the AlexNet weights for LPIPS.

```
image     all 4-bit PSNR   all 4-bit LPIPS    mixed plan PSNR  mixed plan LPIPS  prompt
    0              11.08            0.8007              29.22            0.0470  a photo of a cat sitting on a wooden table
    1              11.23            0.7146              21.13            0.1594  a portrait of an old fisherman with a grey beard, natural light
    2              13.27            0.6336              29.49            0.0326  a red sports car parked on a wet city street at night
    3              10.53            0.7276              25.99            0.0497  a bowl of ramen with an egg and green onions, top view
    4              10.87            0.6869              26.25            0.0584  a snowy mountain lake at sunrise, reflections on the water
    5              12.88            0.6586              33.32            0.0216  a golden retriever puppy running on green grass
    6              10.67            0.7532              21.00            0.1362  a cozy living room with a fireplace and bookshelves
    7              12.12            0.7164              28.96            0.0540  a watercolor painting of a lighthouse on a cliff
    8              13.22            0.8380              29.94            0.0319  a close-up photo of a hummingbird drinking from a flower
    9              10.89            0.7406              22.86            0.0885  a busy market street in Marrakech with colorful spices
   10              11.30            0.7089              26.67            0.0516  a futuristic city skyline with flying cars, digital art
   11              10.33            0.7727              24.05            0.0740  a sign that says OPEN on a coffee shop door
   12              10.89            0.7595              23.99            0.0618  a pile of fresh oranges and lemons in a wooden crate
   13              10.89            0.7222              25.33            0.0688  an astronaut riding a horse on the moon
   14              10.65            0.6662              23.00            0.0897  a young woman reading a book in a sunny cafe
   15              11.50            0.7632              25.81            0.0632  a pencil sketch of an owl on a branch
   16              11.39            0.7487              25.19            0.0705  a tropical beach with palm trees and turquoise water
   17              10.83            0.6807              23.09            0.0825  a plate of chocolate chip cookies next to a glass of milk
   18              13.87            0.8753              35.10            0.0368  an old castle in the fog, oil painting
   19              11.82            0.6433              27.82            0.0291  a macro photo of dew drops on a spider web
 mean              11.51            0.7305              26.41            0.0654
worst              10.33            0.8753              21.00            0.1594
wrote quant_eval/grid.png
```

The grid is 1536 x 10272 pixels and 30,371,839 bytes: 3 columns of 512, and a 32-pixel title strip over 20 rows of 512.

What the grid shows, by eye (the assistant looked at rows 0 to 3 at half size and row 6 at full size):

- All 4-bit: every image is colored noise. The objects of the prompt are gone, and only a rough layout is left in some rows.
- Mixed plan: each image shows the same subject, layout and colors as float16. Row 6 (the living room, the lowest PSNR) differs in small parts, for example the shape of the fireplace on the right.

Calculations:

- Fixed bytes: 318,404 output channels x 2 bytes = 636,808 bytes of scales. 443,844 other values x 2 bytes = 887,688 bytes. Together 1,524,496 bytes.
- All 4-bit: 865,466,880 / 2 + 1,524,496 = 434,257,936 bytes. All 8-bit: 865,466,880 + 1,524,496 = 866,991,376 bytes. Both match the output.
- The plan leaves 700,000,000 - 699,997,456 = 2,544 bytes unused.
- The plan keeps 531,479,040 / 865,466,880 = 61.4% of the weights at 8 bits. The 54 layers at 4 bits hold 333,987,840 weights. The largest of them are the `conv1` and `conv2` of `up_blocks.0` and `down_blocks.3`, the 8 x 8 and 16 x 16 parts of the UNet.
- The plan is 699,997,456 / 1,731,821,448 = 40.4% of the float16 UNet, and 700 / 1920 = 36.5% of the tab limit.
- Mixed plan against all 4-bit: the mean PSNR is 26.41 - 11.51 = 14.90 dB higher, and the mean LPIPS is 0.7305 / 0.0654 = 11.2 times lower.

Rejected:

- All 4-bit with one scale per output channel: it destroys every image. The predicted rms of 1.369 already said so, but the prediction assumes that errors add, and the real error is not measured on the latent.

Check of the move to `quantize.py`: `uv run quant_sensitivity.py` ran again after the change, and `cmp` compared its output with `quant_sensitivity.txt`. Device: the same Mac GPU. Output:

```
IDENTICAL
```

Not measured on the iPhone.

Conclusion: under 700 MB, the mixed plan keeps the images close to float16 (mean PSNR 26.41 dB, mean LPIPS 0.0654). All 4-bit with per-channel scales is not usable.

Next step: the user decides if the mixed plan is good enough for the engine, or if 4 bits needs a smaller group per scale (for example one scale per 64 or 128 weights) before it is used.

## 2026-11-18: Mixed plan grid, checked by the user

This entry completes "Mixed 4-bit and 8-bit plan, and a 20-image test". The user looked at `quant_eval/grid.png` and reported that the mixed plan has really good results.

This is a check by eye on the Mac. No new command ran.

Not measured on the iPhone.

Next step: write the UNet weights in the mixed format (4-bit or 8-bit integers per `quant_plan.json`, one float16 scale per output channel) and check the file size against the 699,997,456 bytes of the plan.

## 2026-11-18: Packed 4-bit and 8-bit weights, converter and kernels

Files:

- `pack_weights.py` (new): the packed format. It rounds, packs and unpacks the weights. The converter and the test data script share it.
- `convert_weights.py` (changed): new option `--plan`. It packs the weights of the layers in the plan, and checks them after writing.
- `engine/src/layers/mod.rs` (changed): one shared function `bind_plan` connects any list of input buffers. `PreparedLayer` keeps a list of input buffers instead of exactly 3.
- `engine/src/layers/packed/mod.rs` (new): `WeightBits`, `PackedTensors`, and `prepare_packed_layer`.
- `engine/src/layers/packed/linear.wgsl`, `conv1x1.wgsl`, `conv3x3.wgsl` (new): the packed kernels.
- `engine/src/cases.rs` (changed): `report_layer` compares and times a prepared layer. The float cases and the packed cases share it.
- `engine/src/packed_cases.rs` (new): 8 packed cases, the 4 float cases at 4 and at 8 bits.
- `engine/src/lib.rs` (changed): exports the packed types and cases.
- `engine/testdata/make_layers.py` (changed): the case loop runs only as a script, so another script can import `CASES`.
- `engine/testdata/make_packed_layers.py` (new) and `engine/testdata/<case>_q4/`, `<case>_q8/` (new): the packed test data.
- `engine/tests/packed_layers.rs` (new): the packed test.
- `weights/unet_packed.bin` and `weights/unet_packed.json` (new).

The packed format:

- One row is one output channel: all weights of that channel in PyTorch order, for example in_channels x 3 x 3.
- Each weight is a signed integer from -8 to 7 (4 bits) or -128 to 127 (8 bits), in two's complement.
- The integers of a row are packed into u32 words, lowest bits first. Value i of a word sits at bits i x bits to i x bits + bits - 1.
- Each row ends with zero bits up to a whole word. So every row starts on a word, and `row_words` = ceil(row values x bits / 32).
- Each row has one float16 scale. The weight is integer x scale.
- In the index, a packed weight has the dtype `int4` or `int8` and keeps its PyTorch shape. Its scales follow as `<name>_scale`, float16, shape `[out_channels]`.

Design choices:

- The scale is float16, because the 700 MB plan counted 2 bytes per scale. Float32 scales would add 636,808 bytes, and the plan has only 2,544 bytes of room.
- The rounding divides by the float16 scale as stored, so each integer is the nearest one for the scale that the engine reads. `quantize.py` (used for the image test) keeps the scale in float32. The two differ by the float16 rounding of the scale: at most 2^-11 of the weight.
- Each kernel unpacks the integers inside its inner loop with `extractBits` on an `i32`. `extractBits` on an `i32` copies the top bit of the field upward, so 0b1111 becomes -1.
- Linear and conv1x1 read 4 neighbor weights at a time. 4 weights take 16 bits (4-bit) or 32 bits (8-bit), so they always sit in one word. conv3x3 reads one weight at a time. 32 is a multiple of 4 and of 8, so a weight never crosses a word. For this reason only 4 and 8 bits are allowed. `WeightBits` has only these two values.
- The kernels multiply the integer sums by the scale once, after the inner loop, and not once per product. The scale is the same for every product of one output channel, so the result is the same math with fewer multiplies.
- The kernels read the scales as u32 words with `unpack2x16float`. That WGSL function needs no f16 feature, so it also works where the browser has no `shader-f16`.
- The packed plan reuses the float plan of the same shape: the same size checks, workgroups and output length. Only the shader, two extra Shape fields (`bits`, `row_words`) and the weight and scale buffers differ.
- The packed weights go to `weights/unet_packed.bin`, so `weights/unet.bin` (float32, used by the UNet test) does not change.
- The test data uses the same random layers and inputs as the float cases (same seed). PyTorch runs the layer in float64 with weight = integer x float16 scale, from the same `quantize_rows` as the converter.
- The converter checks each packed weight on disk: `unpack_rows` (a separate function from the packer) reads the integers back, and integer x scale must be at most 0.5 + 1e-3 steps from the source weight.

Command 1: `cp weights/taesd.bin weights/taesd.json <scratchpad>/`, then `uv run convert_weights.py taesd`, then `cmp` of both files. Device: Mac, Python through `uv run`, CPU.

```
taesd: 134 tensors, 9780736 bytes, float32, largest decoder.layers.11.weight 147456 bytes, read back equal
TAESD-UNCHANGED
```

Command 2: `uv run convert_weights.py vae --plan quant_plan.json`.

```
convert_weights.py: error: --plan works only with the single network unet
```

Command 3: `uv run convert_weights.py unet --plan quant_plan.json --dtype float16`.

```
unet_packed: 968 tensors, 700021120 bytes, float16, packed 54 int4, 228 int8, largest up_blocks.1.resnets.0.conv1.weight 29491200 bytes, read back equal
```

Command 4: `uv run testdata/make_packed_layers.py`, in `engine/`. Device: Mac, CPU.

```
linear_q4 (down_blocks.0.attentions.0.proj_in): input (4096, 320) -> output (4096, 320), levels -7 to 7
linear_q8 (down_blocks.0.attentions.0.proj_in): input (4096, 320) -> output (4096, 320), levels -127 to 127
conv1x1_q4 (down_blocks.1.resnets.0.conv_shortcut): input (1, 320, 32, 32) -> output (1, 640, 32, 32), levels -7 to 7
conv1x1_q8 (down_blocks.1.resnets.0.conv_shortcut): input (1, 320, 32, 32) -> output (1, 640, 32, 32), levels -127 to 127
conv3x3_q4 (down_blocks.0.resnets.0.conv1): input (1, 320, 64, 64) -> output (1, 320, 64, 64), levels -7 to 7
conv3x3_q8 (down_blocks.0.resnets.0.conv1): input (1, 320, 64, 64) -> output (1, 320, 64, 64), levels -127 to 127
conv3x3_stride2_q4 (down_blocks.0.downsamplers.0.conv): input (1, 320, 64, 64) -> output (1, 320, 32, 32), levels -7 to 7
conv3x3_stride2_q8 (down_blocks.0.downsamplers.0.conv): input (1, 320, 64, 64) -> output (1, 320, 32, 32), levels -127 to 127
```

Command 5: `cargo test --release --test packed_layers -- --nocapture`, in `engine/`. Device: Mac native.

```
linear_q4: 9.84 GB/s median, 13.08 GB/s best (30 submits, 64 runs per submit, 68.5 ms median submit, 71.4 ms longest), largest error 1.0728836e-6
linear_q8: 12.32 GB/s median, 14.03 GB/s best (30 submits, 64 runs per submit, 55.0 ms median submit, 70.2 ms longest), largest error 1.4305115e-6
conv1x1_q4: 5.72 GB/s median, 8.33 GB/s best (30 submits, 128 runs per submit, 90.3 ms median submit, 96.1 ms longest), largest error 2.1457672e-6
conv1x1_q8: 6.56 GB/s median, 8.75 GB/s best (30 submits, 128 runs per submit, 80.8 ms median submit, 85.0 ms longest), largest error 2.2649765e-6
conv3x3_q4: 0.68 GB/s median, 0.73 GB/s best (30 submits, 4 runs per submit, 64.5 ms median submit, 67.6 ms longest), largest error 6.198883e-6
conv3x3_q8: 0.67 GB/s median, 0.72 GB/s best (30 submits, 4 runs per submit, 68.1 ms median submit, 71.8 ms longest), largest error 6.198883e-6
conv3x3_stride2_q4: 0.87 GB/s median, 0.92 GB/s best (30 submits, 8 runs per submit, 64.8 ms median submit, 73.5 ms longest), largest error 5.841255e-6
conv3x3_stride2_q8: 0.93 GB/s median, 0.98 GB/s best (30 submits, 8 runs per submit, 64.2 ms median submit, 69.2 ms longest), largest error 6.198883e-6
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 18.02s
```

Command 6: `cargo test --release`, in `engine/`. Device: Mac native. The `test result` lines:

```
test result: ok. 18 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.65s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.32s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 18.07s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.68s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

They are, in order: the unit tests, attention, layers, packed_layers, silu, taesd, unet, and the doc tests. `cargo build --release --target wasm32-unknown-unknown` printed no warnings.

Command 7: `cargo test --release --test layers -- --nocapture`, for the float speeds on the same Mac, right after command 6:

```
linear: 12.99 GB/s median, 17.82 GB/s best (30 submits, 64 runs per submit, 53.7 ms median submit, 55.5 ms longest), largest error 1.1920929e-6
conv1x1: 21.23 GB/s median, 24.64 GB/s best (30 submits, 256 runs per submit, 57.3 ms median submit, 86.5 ms longest), largest error 1.9073486e-6
conv3x3: 0.64 GB/s median, 0.77 GB/s best (30 submits, 4 runs per submit, 88.0 ms median submit, 100.9 ms longest), largest error 6.4373016e-6
conv3x3_stride2: 1.20 GB/s median, 1.31 GB/s best (30 submits, 8 runs per submit, 68.2 ms median submit, 82.0 ms longest), largest error 6.198883e-6
```

Deliberate bugs (mutation checks), each undone after the run:

- In `packed/linear.wgsl`, `column_scale` read the other half of the scale word (`[1u - column % 2u]`). In `packed/conv3x3.wgsl`, `weight_value` read the word as `u32`, so `extractBits` did not copy the sign bit. Command: `cargo test --release --test packed_layers -- --nocapture`. Output:

```
thread 'packed_layers_match_pytorch' (10225624) panicked at tests/packed_layers.rs:51:5:
largest error above 1e-4: ["linear_q4: 2.717638e-2", "linear_q8: 2.7259588e-2", "conv3x3_q4: 7.298081e0", "conv3x3_q8: 6.5805464e0", "conv3x3_stride2_q4: 7.0399323e0", "conv3x3_stride2_q8: 6.5805464e0"]
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 15.60s
```

  The conv1x1 cases passed, because neither bug was in its kernel. After the undo, the same test printed `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 15.43s`.

- `check_packed` of the converter, called alone on a random 640 x 320 weight at 4 bits (a short Python script in the scratchpad), with correct data, with the two nibbles of each byte swapped, and with the scales of rows 0 and 1 swapped:

```
correct packing: passed
caught: nibbles swapped: a packed value is 13.939438790137615 steps from its source value
caught: scales of two rows swapped: a packed value is 0.5271584090593785 steps from its source value
```

Calculations:

- Index entries: 686 source tensors + 282 scale tensors = 968. This matches the output.
- File size: 700,021,120 - 699,997,456 (the plan) = 23,664 bytes. This is the 256-byte alignment padding between the 968 tensors, which the plan did not count. The file is 700,021,120 - 700,000,000 = 21,120 bytes over the 700 MB limit.
- Packed weight file sizes: linear_q4 320 x 320 / 2 = 51,200 bytes. conv3x3_q4 320 x 2880 / 2 = 460,800 bytes. Both match `ls`.
- Time per run = median submit / runs per submit:

| Case | float | 4-bit | 8-bit |
|---|---|---|---|
| linear | 53.7 / 64 = 0.839 ms | 68.5 / 64 = 1.070 ms | 55.0 / 64 = 0.859 ms |
| conv1x1 | 57.3 / 256 = 0.224 ms | 90.3 / 128 = 0.705 ms | 80.8 / 128 = 0.631 ms |
| conv3x3 | 88.0 / 4 = 22.0 ms | 64.5 / 4 = 16.1 ms | 68.1 / 4 = 17.0 ms |
| conv3x3_stride2 | 68.2 / 8 = 8.53 ms | 64.8 / 8 = 8.10 ms | 64.2 / 8 = 8.03 ms |

- The packed conv1x1 is 0.705 / 0.224 = 3.1 times (4-bit) and 0.631 / 0.224 = 2.8 times (8-bit) slower than float. The packed conv3x3 is 22.0 / 16.1 = 1.37 times faster at 4 bits. Each speed is from one run on the Mac. The cause of each difference is not measured.

Not done in this step: the engine weight reader still refuses every dtype other than float32 (`WeightDtype`), so it cannot read `unet_packed.bin` yet. The UNet does not use the packed kernels yet. `unet_packed.bin` stores biases and norms as float16, and the kernels read f32 biases.

Not measured on the iPhone.

Conclusion: the converter writes the packed format, and the three packed kernels match PyTorch on the same rounded weights within 6.2e-6 at both bit widths. The packed file is 21,120 bytes over 700 MB because of alignment.

Next step: the user decides how to handle the 21,120 bytes over the limit. Then the engine weight reader learns `int4`, `int8` and `float16`, and the iPhone runs the packed layer cases.

## 2026-11-18: Streaming weight loader, OPFS cache and GPU memory per category

Files added:

- `engine/src/weights/layout.rs`: `spans` finds the place and the buffer size of each tensor in the file. It refuses an offset that is not a multiple of 4, two tensors that share bytes, and a buffer over the limit.
- `engine/src/weights/cursor.rs`: `StreamCursor` cuts a file that arrives in chunks of any size into writes of whole 4-byte words, one tensor at a time.
- `engine/src/weights/upload.rs`: `WeightUpload` (begin, write, finish) streams one weight file into GPU buffers with `queue.write_buffer`.
- `engine/src/weights/gpu_weights.rs`: `GpuWeights` and `GpuTensor`, the tensors of one file on the GPU.
- `engine/src/memory/category.rs`, `counter.rs`, `report.rs`: the counter keeps the live bytes per category (weights, activations, uniforms, read-back), and `MemoryReport` prints them.
- `engine/src/web/engine.rs`: the WASM `Engine` object: `open`, `beginWeights`, `writeWeights`, `settle`, `finishWeights`, `abortWeights`, `freeWeights`, `memoryReport`.
- `engine/tests/stream_weights.rs`: streams the real files in chunks of 65,537 bytes and reads every tensor back.
- `engine/web/weights.js`: the stream loop in the browser and the OPFS cache.
- `engine/web/weights.html`: the page, with the live counter.

Files changed: `engine/src/memory.rs` moved to `engine/src/memory/mod.rs`. `engine/src/gpu.rs` (each buffer has a category, `empty_weight_buffer`, `read_bytes`, `memory_report`, `max_buffer_bytes`, `finish_uploads` is public). `engine/src/weights/index.rs` (`stored_byte_length` for float32, float16, int8 and int4, `into_file_order`). `engine/src/error.rs` (`UnknownDtype`, `WeightLayout`, `TensorTooLarge`, `StreamEnded`). `engine/src/web.rs`, `engine/src/lib.rs`, `engine/src/weights/mod.rs`. The callers of `storage_buffer_from` and `storage_buffer_from_bytes` now name a category: `activation.rs`, `silu/mod.rs`, `bench/math.rs`, `attention/mod.rs`, `layers/mod.rs`, `layers/packed/mod.rs`, `conv.rs`.

Design choices:

- One GPU buffer per tensor. The largest tensor sets the largest buffer, so no buffer comes near the limit of about 1073 MB. A tensor is never split over two buffers. The loader still checks each buffer against the smaller of 1,073,000,000 bytes and the device limit. The device opens with the default WebGPU limits, so the device limit is 268,435,456 bytes, and that is the limit that applies now.
- When the first byte of a tensor arrives, the loader makes the buffer of that tensor. So the weights counter grows with the stream and shows the real GPU memory at each moment.
- `write_buffer` needs offsets and sizes that are multiples of 4. A chunk can end inside a word, so the cursor keeps the last 1 to 3 bytes for the next chunk. Each buffer is the stored length rounded up to 4 bytes. The 0 to 3 extra bytes come from the file padding, which `convert_weights.py` writes as zeros.
- JavaScript reads the stream with `getReader()`, because Safari cannot loop over a stream with `for await`. The loop waits for each OPFS write before the next read. So the page holds one chunk at a time. `response.body.tee()` was rejected: if one branch is slower, `tee` keeps the chunks in memory.
- Each chunk goes to `writeWeights` in pieces of at most 1 MiB (`subarray`, a view). wasm-bindgen copies each piece into WASM memory for the call, and wgpu calls `GPUQueue.writeBuffer` with a view of that memory. So WASM memory holds at most one 1 MiB piece of weights. A direct `writeBuffer` from the JavaScript chunk can skip this copy, but the buffers belong to wgpu in Rust, and JavaScript cannot reach them.
- After each 32 MiB of writes, the page waits for the GPU (`settle`, an empty submit and `onSubmittedWorkDone`). The reason: `writeBuffer` copies into a staging area. Without a wait, that area can grow to the whole file. The size of that area in Safari is not measured.
- Cache: OPFS folder `weights/`. A download writes `<name>.bin` through `createWritable`. The page writes `<name>.json` only after the engine accepted the whole file. If both files exist, the cached copy counts. If one is missing, the page downloads the file. If a cached file fails, the page removes its `.json`, so the next run downloads the file again. Safari supports `createWritable` from Safari 26 (MDN). An older Safari loads from the network without a cache.
- Category for each existing caller: kernel outputs and inputs are activations. The test weights in `layers/mod.rs` and `layers/packed/mod.rs`, and the weights of `conv.rs`, are weights. Uniform buffers are uniforms. The staging buffer of a read-back is read-back. The peak stays one number for all categories, as before.
- The page loads `taesd,unet_packed` by default (`?files=` changes this). `text_encoder.bin` is float32, 1,361,551,360 bytes. With the UNet, that is more than the tab limit.

Command 1: `cargo test --release --lib`, in `engine/`. Device: Mac native.

```
test result: ok. 26 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

The new unit tests:

- The cursor, with every chunk size from 1 to 41 bytes. The small test file has gaps and odd lengths. Its last tensor ends in the middle of a word.
- A short stream.
- The layout checks.
- The stored length of each dtype.
- The 5 real index files: `taesd`, `unet_packed`, `text_encoder`, `unet`, `vae`. In each one, a tensor ends less than 256 bytes before the next one starts. This proves the size rules on the real files.
- The categories count apart.

A unit test failed on the first run:

```
assertion `left == right` failed: chunk size 1
  left: [[1, 2, 3, 4, 5, 6, 7, 8], [17, 18, 19, 20], [33, 34, 35, 36, 37, 38, 39, 40, 41, 0, 0, 0]]
 right: [[1, 2, 3, 4, 5, 6, 0, 0], [17, 18, 19, 20], [33, 34, 35, 36, 37, 38, 39, 40, 41, 0, 0, 0]]
```

The test expected zeros in the 2 padding bytes of buffer 0, but the cursor copies the file bytes there. The test file had no zeros in its gaps. The real files have zero padding, so the expectation in the test changed, not the code. The real-file test below checks that every padding byte on the GPU is zero.

Command 2: `cargo test --release --test stream_weights -- --nocapture --test-threads=1`, in `engine/`. Device: Mac native.

```
test a_cut_stream_fails ... cut stream: the weight stream ended after 4890368 bytes, the index needs 9780736 bytes
ok
test streamed_weights_match_the_files ... taesd: 134 tensors in 134 GPU buffers, 9780252 bytes, largest buffer 147456 bytes, streamed in 9.6 ms
weights: 9780252 bytes (9.8 MB)
activations: 0 bytes (0.0 MB)
uniforms: 0 bytes (0.0 MB)
read-back: 0 bytes (0.0 MB)
total: 9780252 bytes (9.8 MB)
peak: 9780252 bytes (9.8 MB)
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes, streamed in 662.1 ms
weights: 699997456 bytes (700.0 MB)
activations: 0 bytes (0.0 MB)
uniforms: 0 bytes (0.0 MB)
read-back: 0 bytes (0.0 MB)
total: 699997456 bytes (700.0 MB)
peak: 699997456 bytes (700.0 MB)
ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.22s
```

The test reads every tensor back from the GPU and compares it with its bytes in the file. It also checks that the weights category equals the sum of the buffers, and that the category is 0 after the weights are dropped.

Deliberate bugs (mutation checks), each undone after the run, with the same command:

- `cursor.rs` drops the 1 to 3 bytes at the end of each chunk, instead of keeping them:

```
taesd: decoder.layers.11.weight differs from the file
test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
```

- `upload.rs` writes each piece 4 bytes too far into its buffer:

```
taesd: decoder.layers.11.weight differs from the file
test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
```

After the undo: `test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.13s`.

Command 3: `cargo test --release`, in `engine/`. Device: Mac native. The `test result` lines, in order: unit tests, attention, layers, packed_layers, silu, stream_weights, taesd, unet, doc tests:

```
test result: ok. 26 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 6.25s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.13s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 16.70s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.43s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.71s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

`cargo build --release --target wasm32-unknown-unknown` printed no warnings. `wasm-pack build --release --target web --out-dir web/pkg --no-typescript` finished.

Command 4: page `http://127.0.0.1:8765/engine/web/weights.html`, with `python3 -m http.server 8765` in the project folder. Device: Mac Chromium. First load, empty cache. The counter and the log at the end:

```
GPU buffers of the engine:
weights: 709777708 bytes (709.8 MB)
activations: 0 bytes (0.0 MB)
uniforms: 0 bytes (0.0 MB)
read-back: 0 bytes (0.0 MB)
total: 709777708 bytes (709.8 MB)
peak: 709777708 bytes (709.8 MB)
WASM memory: 3473408 bytes
JS heap: 184802884 bytes
----
persistent storage: false
cache: OPFS folder weights/
taesd: 134 tensors in 134 GPU buffers, 9780252 bytes, largest buffer 147456 bytes
taesd: from network, 9780736 bytes in 0.12 s, 78.9 MB/s, largest chunk 1048576 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes
unet_packed: from network, 700021120 bytes in 2.49 s, 281.2 MB/s, largest chunk 2097152 bytes
storage: 709935995 bytes used, quota 11447354235 bytes
```

The JS heap line comes from `performance.memory`. 10 s later it said `185113641 used, 189121633 total`. The Chrome DevTools Protocol (`Runtime.getHeapUsage`, then `HeapProfiler.collectGarbage`, then `Runtime.getHeapUsage`) gave this, at the same moment:

```
{"before":{"usedSize":2054600,"totalSize":2949120,"embedderHeapUsedSize":772336,"backingStorageSize":65738},"after":{"usedSize":1996456,"totalSize":2686976,"embedderHeapUsedSize":556088,"backingStorageSize":65730}}
```

So the JS heap holds about 2 MB, and the `performance.memory` number is not a live measure. Chromium updates it late and rounds it. The page line stays, because it is the only JS number that a page can read, but it does not prove the JS memory.

Command 5: the same page, button "Free the weights and load again". Device: Mac Chromium. The log lines after the first load:

```
freed:
weights: 0 bytes (0.0 MB)
activations: 0 bytes (0.0 MB)
uniforms: 0 bytes (0.0 MB)
read-back: 0 bytes (0.0 MB)
total: 0 bytes (0.0 MB)
peak: 709777708 bytes (709.8 MB)
cache: OPFS folder weights/
taesd: 134 tensors in 134 GPU buffers, 9780252 bytes, largest buffer 147456 bytes
taesd: from cache, 9780736 bytes in 0.03 s, 337.3 MB/s, largest chunk 1245184 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes
unet_packed: from cache, 700021120 bytes in 0.74 s, 950.2 MB/s, largest chunk 2097152 bytes
storage: 709935995 bytes used, quota 11447354235 bytes
```

Command 6: a cut cache file. A Playwright script cut the cached `unet_packed.bin` to 350,000,000 bytes, reloaded the page, then reloaded it again. Device: Mac Chromium.

```
cut cache file to 350000000 bytes
== reload 1
persistent storage: false
cache: OPFS folder weights/
taesd: 134 tensors in 134 GPU buffers, 9780252 bytes, largest buffer 147456 bytes
taesd: from cache, 9780736 bytes in 0.01 s, 889.2 MB/s, largest chunk 2097152 bytes
FAILED: the weight stream ended after 350000000 bytes, the index needs 700021120 bytes

== reload 2
persistent storage: false
cache: OPFS folder weights/
taesd: 134 tensors in 134 GPU buffers, 9780252 bytes, largest buffer 147456 bytes
taesd: from cache, 9780736 bytes in 0.01 s, 865.6 MB/s, largest chunk 2097152 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes
unet_packed: from network, 700021120 bytes in 2.67 s, 262.6 MB/s, largest chunk 2097152 bytes
storage: 709935995 bytes used, quota 11447354235 bytes
```

The cleanup after a failed cache read (`removeEntry` of the `.json`) was added before this check ran. Without it, a cut file fails on every run until the user presses "Clear the cache". The check never ran without the cleanup.

Command 7: button "Clear the cache", then a script listed the OPFS root. Device: Mac Chromium.

```
OPFS root entries after clear: []
```

Calculations:

- GPU weights: 9,780,252 + 699,997,456 = 709,777,708 bytes. This matches the counter on the page.
- Bytes in the files that are not on the GPU: taesd 9,780,736 - 9,780,252 = 484. unet_packed 700,021,120 - 699,997,456 = 23,664. Both are the 256-byte alignment padding between tensors. The 23,664 is the same as in "Packed 4-bit and 8-bit weights, converter and kernels".
- The 699,997,456 bytes of `unet_packed` on the GPU are the plan number of that entry, byte for byte.
- Largest buffer: 29,491,200 / 268,435,456 (the device limit that applies) = 11.0%. 29,491,200 / 1,073,000,000 = 2.7%.
- WASM memory at the end: 3,473,408 bytes. That is 3,473,408 / 709,777,708 = 0.49% of the weights on the GPU. 3,473,408 - 2,097,152 = 1,376,256 bytes more than one 2 MiB chunk.
- Mac native stream speed: 700,021,120 bytes / 0.6621 s = 1,057 MB/s (file read and GPU write, no network).
- Mac Chromium: network 700,021,120 / 2.49 s = 281 MB/s, from localhost. Cache 700,021,120 / 0.74 s = 946 MB/s. The page prints 281.2 and 950.2, because it divides by the seconds before rounding. The cache is 2.49 / 0.74 = 3.4 times faster than localhost.
- The weights use 709.8 / 1920 = 37.0% of the iPhone tab limit. The JS heap, WASM memory and the browser's own GPU memory are not in this number.

Not measured on the iPhone. These Safari numbers are not known: the chunk size of `fetch`, the chunk size of `File.stream()`, and the size of the `writeBuffer` staging area. The OPFS quota of Safari for 710 MB of cache is also not known.

Conclusion: the loader streams each weight file into one GPU buffer per tensor with `queue.write_buffer`. On the Mac, every tensor matches its file bytes, and the counter shows the GPU bytes per category while the file streams. The second load reads the OPFS cache, and a broken cache file downloads again on the next run. WASM memory stays at 3,473,408 bytes in Chromium. The kernels do not read `GpuWeights` yet.

Next step: the user opens `engine/web/weights.html` on the iPhone through ngrok, two times. The first load downloads, and the second load reads the cache. The user sends both logs and the counter.

## 2026-11-18: HTTPS server on the local network, in place of ngrok

The user reported that loading through ngrok is very slow. The user sent no numbers. ngrok sends every byte from the Mac to an ngrok server on the internet, and then back to the iPhone. So the speed is limited by the internet connection and by ngrok, not by the local network.

Files: no project file changed. The certificate is in `~/.config/inference-lan/` (`lan-cert.pem`, `lan-key.pem`), outside the project folder, so that the server cannot serve the key.

Design choices:

- WebGPU and OPFS work only in a secure context: HTTPS, or `localhost`. A plain `http://10.120.1.125` address is not secure, so the page cannot use the GPU there. The server must use HTTPS.
- `mkcert` makes the certificate, with the local certificate authority (CA) that already exists on this Mac (`mkcert -CAROOT`). The certificate names `10.120.1.125`, `Teimurs-MacBook-Pro.local`, `localhost` and `127.0.0.1`. It expires on 18 February 2029.
- Python 3.14 `http.server` has `--tls-cert` and `--tls-key`, so no new server program is needed.
- The iPhone must trust the mkcert root certificate (`rootCA.pem`), one time. The key of that root (`rootCA-key.pem`) stays on the Mac.
- Rejected: Tailscale (`tailscale serve`). It gives a certificate that the iPhone trusts without a CA install, but it needs the Tailscale app and an account on the iPhone.

Commands, in the project folder:

```
mkcert -cert-file ~/.config/inference-lan/lan-cert.pem -key-file ~/.config/inference-lan/lan-key.pem 10.120.1.125 Teimurs-MacBook-Pro.local localhost 127.0.0.1
python3 -m http.server 8443 --bind 0.0.0.0 --tls-cert ~/.config/inference-lan/lan-cert.pem --tls-key ~/.config/inference-lan/lan-key.pem
curl -s --cacert "$(mkcert -CAROOT)/rootCA.pem" -o /dev/null -w "%{size_download} bytes in %{time_total} s\n" https://10.120.1.125:8443/weights/unet_packed.bin
```

Device: Mac native (curl on the same Mac, to its own network address). Output of the curl command:

```
700021120 bytes in 5.421785 s
```

Calculation: 700,021,120 / 5.421785 s = 129 MB/s. This is the speed of the Python TLS server on the Mac without Wi-Fi. It is an upper limit for the iPhone. The Wi-Fi speed is not measured.

Not measured on the iPhone.

Next step: the user installs `rootCA.pem` on the iPhone, opens `https://10.120.1.125:8443/engine/web/weights.html`, and sends the log of the first load and of a reload.

## 2026-11-18: Weight loader on the iPhone, first load over the local network

Page: `https://10.120.1.125:8443/engine/web/weights.html`, served by the HTTPS server of "HTTPS server on the local network, in place of ngrok". Device: iPhone. The user sent a screenshot of the page at 14:24. The address bar shows "Not Secure", so Safari did not trust the certificate. The user did not say whether the mkcert root is installed. The page still opened the GPU and OPFS. These are the lines of the screenshot. The screen wraps long lines, and each line below is one line of the page:

```
GPU buffers of the engine:
weights: 709777708 bytes (709.8 MB)
activations: 0 bytes (0.0 MB)
uniforms: 0 bytes (0.0 MB)
read-back: 0 bytes (0.0 MB)
total: 709777708 bytes (709.8 MB)
peak: 709777708 bytes (709.8 MB)
WASM memory: 3670016 bytes
JS heap: not reported by this browser
```

```
persistent storage: false
cache: OPFS folder weights/
taesd: 134 tensors in 134 GPU buffers, 9780252 bytes, largest buffer 147456 bytes
taesd: from network, 9780736 bytes in 1.04 s, 9.4 MB/s, largest chunk 1867776 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes
unet_packed: from network, 700021120 bytes in 95.19 s, 7.4 MB/s, largest chunk 2179072 bytes
storage: 144244686895 bytes used, quota 153330332468 bytes
```

Calculations:

- The GPU weights are 709,777,708 bytes, the same number as Mac native and Mac Chromium, byte for byte.
- The tab did not crash. The weights use 709.8 / 1920 = 37.0% of the tab limit.
- WASM memory: 3,670,016 bytes. That is 3,670,016 / 709,777,708 = 0.52% of the weights on the GPU.
- The largest `fetch` chunk in Safari: 2,179,072 bytes for the UNet and 1,867,776 bytes for TAESD. The page cuts each chunk into pieces of at most 1,048,576 bytes before WASM.
- Speed of the UNet file: 700,021,120 / 95.19 s = 7.35 MB/s, which is 7.35 x 8 = 58.8 Mbit/s.
- curl on the Mac, from the same server, took 5.42 s for the same file (129 MB/s). The iPhone load took 95.19 / 5.42 = 17.6 times longer.
- The `storage` line is the estimate of Safari. Its usage of 144,244,686,895 bytes is about 203 times the 709,935,995 bytes that Chromium reported for the same cache. So this number is not the size of the cache alone. Its meaning in Safari is not known.

What the 95.19 s contains: the Wi-Fi transfer, the TLS decryption, the OPFS write of each chunk, and the GPU writes. The page waits for each OPFS write before it reads the next chunk. This run does not split the time into these parts, so the cause of the 7.35 MB/s is not known.

Not measured on the iPhone yet: a reload from the cache. It has no network part, so it separates the network from the work on the iPhone.

Conclusion: the streaming loader works on the iPhone. It loads 709,777,708 bytes into 1,102 GPU buffers, WASM memory stays at 3,670,016 bytes, and the tab survives.

Next step: the user reloads the page and sends the two `from cache` lines.

## 2026-11-18: Weight loader on the iPhone, reload from the cache

This entry continues "Weight loader on the iPhone, first load over the local network". The user reloaded the page and sent this text, not the page lines:

```
from cache 407mb/s
```

Device: iPhone. The user did not say which file this is, and did not send the seconds or the `taesd` line. The calculations below assume that it is the `unet_packed` line, because that file is 98.6% of the bytes (700,021,120 / 709,801,856).

Calculations:

- Time for the UNet file from the cache: 700,021,120 / 407 MB/s = 1.72 s.
- The cache load is 407 / 7.35 = 55 times faster than the first load over the network (95.19 s).
- Mac Chromium read the same cache at 950.2 MB/s. The iPhone reaches 407 / 950.2 = 43% of that.

What this proves: on the iPhone, the OPFS read and the GPU writes together take about 1.72 s for the UNet. So they are not the cause of the 95.19 s. The cache load does not write to OPFS and does not decrypt TLS. So these causes remain: the Wi-Fi transfer, the TLS decryption, and the OPFS write of each chunk. The Python server is not the limit, because it sent the same file to curl on the Mac at 129 MB/s.

Not measured: the time of each remaining part.

Conclusion: the OPFS cache works on the iPhone. A load from the cache takes about 1.72 s for the UNet. So the slow download happens one time per phone. It happens again only after a cache clear, or after Safari evicts the cache (`persistent storage: false`).

Next step: if the first download must be faster, the page measures three times. They are the wait for the network, the OPFS writes and the GPU writes.

## 2026-11-18: UNet activation plan, from the forward pass

The user asked for three things. First, run in stages: load the text encoder, run it, free its buffers, then the UNet, then the decoder. Second, plan the UNet tensors between kernels: work out the size and lifetime of each one from the forward pass, and put them into a small set of reusable buffers. Print the planned peak in MB before the run and the measured peak after. Third, make the bind groups once and reuse them. The tab limit is about 1920 MB.

The engine cannot run the stages yet. It has no text encoder, and the UNet runs only `conv_in`. The project rule allows one step per prompt. This step builds the plan. A plan can exist before the kernels, and it shows if the UNet stage fits. The measured peak and the bind groups need a run on the planned buffers, so they come in the next step.

Files added:

- `engine/src/plan/mod.rs`: `TensorLife` (bytes, first op, last op) and `MemoryPlan` (the slot of each tensor and the size of each slot).
- `engine/src/plan/graph.rs`: `Graph`, a forward pass as a list of ops. Each op reads tensors and writes new tensors. It computes the lifetime of each tensor.
- `engine/src/plan/slots.rs`: `assign_slots`, which puts the tensors into slots.
- `engine/src/plan/live.rs`: `live_peak`, the most bytes that are live at one op.
- `engine/src/unet/plan/mod.rs`: `plan_unet_activations` and the tests against the golden files.
- `engine/src/unet/plan/forward.rs`, `resnet.rs`, `transformer.rs`: the UNet forward pass as ops.
- `engine/examples/memory_plan.rs`: prints the plan and one line per stage.

Files changed: `engine/src/lib.rs`, `engine/src/unet/mod.rs`, `engine/src/weights/mod.rs`, `engine/src/weights/index.rs` (new `weight_bytes_on_gpu`).

Design choices:

- A slot is one GPU buffer. A slot holds one tensor at a time. Two tensors share a slot only if no op uses both. The reason for separate buffers, not offsets in one large buffer: one binding is limited to 128 MiB by default, and a kernel can bind a whole slot.
- The lifetime of a tensor runs from the op that writes it to the last op that reads it, both included. So the input and the output of one op are never in one slot. A tensor that nothing reads (the attention scores scratch) lives only at its writer.
- The ops follow diffusers `UNet2DConditionModel.forward` and the numbers of `unet/config.json`: levels of 320, 640, 1280 and 1280 channels, 5, 10, 20 and 20 heads, 2 ResNet blocks per down level and 3 per up level, `use_linear_projection`, GEGLU with 4 times the width.
- The ops are the kernels that the engine will run, not the PyTorch modules. GroupNorm and SiLU are one op. A residual add, the time vector add and SiLU on `time_embedding.linear_1` work in place, so they make no new tensor. `torch.cat` of the up path makes a new tensor. If a later kernel differs, the graph must change with it.
- Attention has a scores scratch of heads x key tokens x 256 rows. 256 is the chunk of "Attention on the iPhone", which took 145.0 ms per submit.
- The latent input lives to the end, because the scheduler step after the UNet reads it again. The text tensor lives until the last cross attention. The UNet output lives to the end.
- The planner is "greedy by size": the largest tensor first, into the smallest free slot, or into a new slot. A slot grows to its largest tensor. With the size order this never happens, but the rule keeps the plan correct for any order.
- Every activation is f32, as in the kernels today.
- The plan does not count the uniform buffers, the read-back buffers, the pipelines, or the memory of the browser itself.

Command: `cargo run --release --example memory_plan`, in `engine/`. Device: Mac native (no GPU work. The program reads the three `.json` indexes and plans on the CPU).

```
unet forward pass: 568 ops, 491 tensors
unet activations, most live at one op: 84272128 bytes (84.3 MB) at down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.geglu
unet activations, planned: 22 buffers, 111141888 bytes (111.1 MB), largest buffer 41943040 bytes
stage 1 text_encoder: weights 1361551360 bytes (1361.6 MB), activations not planned
stage 2 unet: weights 699997456 + activations 111141888 = 811139344 bytes (811.1 MB, 42.2% of the 1920 MB tab limit)
stage 3 taesd: weights 9780252 bytes (9.8 MB), activations not planned
```

Command: `cargo test --release --lib`, in `engine/`. Device: Mac native.

```
test unet::plan::tests::plan_is_above_the_lower_bound_and_fits_one_binding ... ok
test unet::plan::tests::tensors_that_are_live_together_never_share_a_buffer ... ok
test unet::plan::tests::block_shapes_match_the_golden_files ... ok
test result: ok. 34 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

The golden test reads the 12 golden files (`time_embedding`, `conv_in`, `down_block_0` to `down_block_3`, `mid_block`, `up_block_0` to `up_block_3`, `unet_output`). It checks that the planned shape of each block output is the PyTorch shape. A wrong skip order in the up path would change the channels of a block.

Command: `cargo test --release --test unet -- --nocapture`, in `engine/`. Device: Mac native. The UNet run did not change:

```
conv_in: largest error 9.5367431640625e-7, mean error 2.2269295524637855e-8, worst at [0, 240, 11, 1] (engine -2.2016876, golden -2.2016885), 1 submits, longest 1.7 ms, peak 10551296 bytes
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
```

`cargo build --release --target wasm32-unknown-unknown` gave no warnings.

Calculations:

- The lower bound at the GEGLU op of `down_blocks.0.attentions.1`: latent 4 x 64 x 64 x 4 = 65,536. Text 77 x 1024 x 4 = 315,392. SiLU(time embedding) 1280 x 4 = 5,120. Four tensors of 320 x 64 x 64 x 4 = 5,242,880 each: the skips `conv_in` and `attentions.0`, the transformer input, and the transformer stream. `ff.net.0.proj` 4096 x 2560 x 4 = 41,943,040. The GEGLU output 4096 x 1280 x 4 = 20,971,520. Sum: 65,536 + 315,392 + 5,120 + 4 x 5,242,880 + 41,943,040 + 20,971,520 = 84,272,128 bytes. That is the printed number.
- The largest buffer, 41,943,040 bytes, is `ff.net.0.proj` at 64 x 64. 41,943,040 / 134,217,728 (the default binding limit) = 31.3%.
- The self attention scratch at 64 x 64: 5 heads x 4096 keys x 256 rows x 4 = 20,971,520 bytes. The full table would be 335,544,320 bytes.
- The plan uses 111,141,888 / 84,272,128 = 1.319 times the lower bound, so 26,869,760 bytes (26.9 MB) more.
- The weights of stage 2, 699,997,456 bytes, are the GPU bytes of "Streaming weight loader, OPFS cache and GPU memory per category", byte for byte.
- Stage 2 total: 699,997,456 + 111,141,888 = 811,139,344 bytes. 811,139,344 / 1,920,000,000 = 42.2%.
- Without stages, all weights at once: 1,361,551,360 + 699,997,456 + 9,780,252 = 2,071,329,068 bytes. That is 151,329,068 bytes over 1,920,000,000 before any activation. So the stages are necessary with the f32 text encoder.
- Stage 1 with the f32 text encoder: 1,361,551,360 / 1,920,000,000 = 70.9% for the weights alone.

Mutation checks. Each bug made a test fail, and the code was restored:

```
overlap rule with < in place of <= (a tensor that ends at op k and one that starts at op k may share):
test plan::slots::tests::a_small_tensor_takes_the_smallest_free_slot ... FAILED
test plan::slots::tests::the_last_reader_and_a_new_writer_do_not_share ... FAILED
test plan::slots::tests::tensors_that_never_meet_share_a_slot ... FAILED
test unet::plan::tests::tensors_that_are_live_together_never_share_a_buffer ... FAILED

first free slot in place of the smallest free slot:
test plan::slots::tests::a_small_tensor_takes_the_smallest_free_slot ... FAILED
test result: FAILED. 8 passed; 1 failed; 0 ignored; 0 measured; 25 filtered out; finished in 0.02s
unet activations, planned: 22 buffers, 111141888 bytes (111.1 MB), largest buffer 41943040 bytes

slot does not grow, tensors ordered by first op:
test unet::plan::tests::plan_is_above_the_lower_bound_and_fits_one_binding ... FAILED
test unet::plan::tests::tensors_that_are_live_together_never_share_a_buffer ... FAILED
time_embedding.linear_1 does not fit
```

The second bug gives the same UNet plan, 111,141,888 bytes. Only the unit test finds it.

Failed and rejected designs:

- The first version of `assign_slots` did not grow a slot. It was correct only because of the size order. A try with the order "by first op" printed `unet activations, planned: 22 buffers, 50190852 bytes (50.2 MB), largest buffer 5242880 bytes`. That is below the lower bound of 84,272,128 bytes, which is not possible. The first tensor of each slot was small, and larger tensors joined it later. The slot now grows to its largest tensor.
- With growing slots, the order "by first op" printed `unet activations, planned: 22 buffers, 519110656 bytes (519.1 MB), largest buffer 41943040 bytes`. That is 519,110,656 / 111,141,888 = 4.67 times the size order. Rejected.

Not measured on the iPhone. Nothing ran on a GPU in this step, except the `conv_in` test.

Not done in this step:

- The measured peak. The UNet does not run on the planned buffers yet.
- The bind groups. They need the buffers of the plan. With fixed slots, each op has fixed buffers, so its bind group can be made once.
- The stages. The engine has no text encoder.
- A plan for the activations of the text encoder and of TAESD.

Notes for the next step: a slot can be larger than its tensor. So a kernel must take its sizes from a uniform, not from `arrayLength`. `silu.wgsl` uses `arrayLength` today. The bench kernels also use it, but they do not run on slots.

Conclusion: the UNet forward pass has 568 ops and 491 tensors. Its activations fit into 22 buffers of 111,141,888 bytes in total. With the 4-bit and 8-bit weights, stage 2 needs 811,139,344 bytes, 42.2% of the tab limit, before the browser's own memory.

Next step: a buffer pool. It makes the 22 buffers once, at the start of stage 2. `conv_in` writes into its planned buffer, with a bind group that is made once. The test prints the planned peak before the run and the measured peak after it.

## 2026-11-18: Activation buffer pool, conv_in on its planned buffers

This step continues "UNet activation plan, from the forward pass". The UNet stage now makes the 22 planned buffers one time, binds `conv_in` to its planned buffers one time, and runs it two times. The test prints the planned peak before the runs and the measured peak after each run.

Files added:

- `engine/src/plan/pool.rs`: `ActivationPool`. It makes one GPU buffer per slot and finds the buffer and the size of a tensor by its name.

Files changed:

- `engine/src/unet/mod.rs`: `run_unet` is replaced by `UnetStage`. `UnetStage::prepare` plans, makes the slot buffers, reads the weights and binds the kernels. `UnetStage::run` uploads the latent, submits, and reads back. It makes no activation buffer and no bind group.
- `engine/src/unet/plan/mod.rs`: `planned_unet` gives the graph and the plan to the stage. New test `every_tensor_name_is_unique`.
- `engine/src/plan/mod.rs`, `engine/src/plan/graph.rs`: the pool reads `slot_of`, `tensor_count` and `dims_at`.
- `engine/src/conv.rs`: `load_conv3x3` reads and uploads the weights. `bind_conv3x3` (TAESD) and the new `bind_conv3x3_into` (slots) both use it. `bind_conv3x3_into` stops with `PlanMismatch` if the kernel output size is not the planned size.
- `engine/src/layers/mod.rs`: `bind_layer_into` and `SlotLayer`, a layer that writes into a buffer that the caller owns.
- `engine/src/checks.rs`: `check_buffer_holds`. A slot can be larger than its tensor, so the input and output check is "at least". The weight check stays exact.
- `engine/src/block.rs`: `check_tensor` reads only the values of the tensor. `check_block` calls it.
- `engine/src/gpu.rs`: `write_f32`, `read_f32_prefix`, `bind_groups_made` and `buffers_made`. The read-back copy has the size of the read, not of the whole buffer.
- `engine/src/dispatch.rs`: each bind group adds 1 to the count.
- `engine/src/memory/counter.rs`, `engine/src/memory/report.rs`: a count of the buffers made, and `MemoryReport::peak`.
- `engine/src/error.rs`: `UnplannedTensor` and `PlanMismatch`.
- `engine/tests/unet.rs`: prepares once, runs two times, prints the planned and the measured numbers, and checks them.

Design choices:

- The pool makes all 22 slots at the start of the stage. So the stage holds its planned peak from the start. The reason: a run must not make or free buffers, and the bind groups need fixed buffers.
- A tensor is found by name, for example `"sample"` and `"conv_in"`. These are the names of the graph. A name that the plan does not have stops `prepare` with `UnplannedTensor`.
- The weights of `conv_in` are read in `prepare` and stay until the stage is dropped. They still come from the f32 `unet.bin`, as in "UNet step 1, weight loader and conv_in".
- The test proves "no buffer and no bind group in a run" with two counts. A run may make only one read-back copy per block that it shows. Dropping the stage frees every buffer, because each `GpuBuffer` frees itself.

Command: `cargo test --release --test unet -- --nocapture`, in `engine/`. Device: Mac native.

```
planned activations: 111141888 bytes (111.1 MB)
after prepare: 1 bind groups, 25 buffers
conv_in: largest error 9.5367431640625e-7, mean error 2.2269295524637855e-8, worst at [0, 240, 11, 1] (engine -2.2016876, golden -2.2016885), 1 submits, longest 1.7 ms, peak 116432160 bytes
run 1: measured peak 116432160 bytes (116.4 MB); live now: activations 111141888, weights 47360, uniforms 32, read-back 0; bind groups 1, buffers made in the run 1
conv_in: largest error 9.5367431640625e-7, mean error 2.2269295524637855e-8, worst at [0, 240, 11, 1] (engine -2.2016876, golden -2.2016885), 1 submits, longest 0.7 ms, peak 116432160 bytes
run 2: measured peak 116432160 bytes (116.4 MB); live now: activations 111141888, weights 47360, uniforms 32, read-back 0; bind groups 1, buffers made in the run 1
test unet_matches_golden_block_by_block ... ok
```

The first run of the new test in this session printed `longest 5.0 ms` for run 1 and `longest 0.8 ms` for run 2, with the same errors and peaks.

Command: `cargo test --release`, in `engine/`. Device: Mac native. Every test file passed. The library tests printed `test result: ok. 35 passed`. The TAESD test printed the same peaks as in "TAESD decoder and its golden set":

```
taesd_upsample_2: largest error 3.866851329803467e-6, mean error 1.394131174756538e-7, worst at [0, 58, 124, 216] (engine 0.034366712, golden 0.03437058), 1 submits, longest 59.5 ms, peak 151142704 bytes
taesd_blocks_3: largest error 1.6063451766967773e-5, mean error 3.397037897636887e-8, worst at [0, 41, 87, 199] (engine 0.09152818, golden 0.09154424), 3 submits, longest 52.0 ms, peak 201474352 bytes
taesd_image: largest error 7.867813110351563e-6, mean error 2.493081107483401e-7, worst at [0, 2, 87, 199] (engine -0.00572145, golden -0.005713582), 1 submits, longest 2.8 ms, peak 201474352 bytes
```

`cargo build --release --target wasm32-unknown-unknown` gave no warnings.

Calculations:

- Measured peak = slots + conv_in weights + uniform + one read-back copy = 111,141,888 + 47,360 + 32 + 5,242,880 = 116,432,160 bytes. That is the printed peak.
- The live activations after each run, 111,141,888 bytes, are the planned bytes, byte for byte.
- conv_in weights: 320 x 4 x 9 x 4 = 46,080 bytes, and the bias 320 x 4 = 1,280 bytes. 46,080 + 1,280 = 47,360.
- The uniform: 5 sizes, padded to 8 u32 values, x 4 bytes = 32 bytes.
- The read-back copy: 320 x 64 x 64 x 4 = 5,242,880 bytes, only the `conv_in` tensor. Its slot can be larger.
- Buffers after prepare: 22 slots + weight + bias + uniform = 25.
- The old `conv_in` test had a peak of 10,551,296 bytes. The stage now holds all 22 slots from the start, so the peak is 116,432,160 bytes. This is the planned stage peak, not a leak.
- The UNet stage with the packed weights: 699,997,456 + 111,141,888 = 811,139,344 bytes, as in the plan entry. The measured part today is only the activations.

Mutation checks. Each bug made the test fail, and the code was restored:

```
a run makes and frees a 1 MB activation buffer:
assertion `left == right` failed: run 1: buffers outside the plan
  left: 2
 right: 1

a run also runs a SiLU kernel (new buffers and a bind group):
assertion `left == right` failed: run 1 made a bind group
  left: 2
 right: 1

conv_in writes into the slot of the latent:
sample: the plan has size [4, 64, 64], the kernel writes [320, 64, 64]

two tensors with one name (norm2 named norm1):
test unet::plan::tests::every_tensor_name_is_unique ... FAILED
```

Failed and rejected designs:

- The first check for buffers outside the plan was `peak == live bytes + largest read-back`. The 1 MB mutation passed it: `run 1: measured peak 116432160 bytes (116.4 MB); live now: activations 111141888, weights 47360, uniforms 32, read-back 0; bind groups 1`. The extra buffer lived when no read-back copy existed, so it did not raise the peak. The count of buffers made replaced this check.

Not measured on the iPhone. `UnetStage` has no WASM entry yet, so no page runs it.

Conclusion: the UNet stage makes its 22 slot buffers and its one bind group in `prepare`. Two runs of `conv_in` match PyTorch to 9.5e-7. Each run makes one buffer, its read-back copy, and no bind group. The live activations equal the plan, 111,141,888 bytes.

Next step: the time embedding (sinusoidal features, linear_1, SiLU, linear_2) on its planned slots, checked against `time_embedding`. It needs a linear kernel and a SiLU kernel that bind into slots. SiLU must take its length from a uniform, not from `arrayLength`.

## 2026-11-18: Time embedding on its planned slots

This step continues "Activation buffer pool, conv_in on its planned buffers". The UNet stage now runs the time embedding (sinusoidal features, linear_1, SiLU, linear_2) on its planned slots, with bind groups that `prepare` makes one time.

Files added:

- `engine/src/layers/vector.wgsl`, `engine/src/layers/vector.rs`: nn.Linear on one row. A flag applies SiLU to each input value as the kernel reads it.
- `engine/src/vector_linear.rs`: `bind_linear_into` reads `<prefix>.weight` and `<prefix>.bias` and binds the kernel between two slots.
- `engine/src/unet/time.rs`: `time_features`, the 320 sinusoidal features, on the CPU.

Files changed:

- `engine/src/unet/mod.rs`: `UnetStage` binds `time_embedding.linear_1` and `time_embedding.linear_2` in `prepare`. `run` takes the timestep and runs the time embedding before `conv_in`.
- `engine/src/unet/plan/forward.rs`, `engine/src/unet/plan/resnet.rs`: the time part of the graph (see below).
- `engine/src/layers/mod.rs`: `bind_plan_into`, the slot checks and the binding that `bind_layer_into` and the one-row layer share.
- `engine/src/plan/pool.rs`, `engine/src/plan/mod.rs`, `engine/src/conv.rs`: `SlotTensor` moved from `conv.rs` into the pool, because two kinds of kernel use it now.
- `engine/src/block.rs`, `engine/tests/common/mod.rs`: `BlockOutput.shape` is now a `Vec<usize>` with the PyTorch shape. Old format: always `[1, channels, height, width]`. New format: the golden shape, for example `[1, 1280]` for `time_embedding`.
- `engine/tests/unet.rs`: reads the golden `timestep` and checks `time_embedding`.

Design choices:

- The linear kernel of "Linear and convolution kernels" reads 4 rows per invocation with no row check, and its row count must be a multiple of 4. The time vector is one row. So a new kernel computes one output per invocation: one dot product of the row with one weight row.
- SiLU is part of the one-row kernel, as a flag in its uniform. diffusers applies SiLU to the time embedding before linear_2 and before each `time_emb_proj`. With the flag, no op writes a SiLU tensor, and the stage needs no SiLU kernel on slots. The flag is the same for every invocation, so the branch is cheap.
- The graph changed with the kernel. Old: `timestep`, `time_proj`, `linear_1`, SiLU in place, `linear_2`, and a SiLU copy `time_embedding.silu` that the ResNet blocks read. New: `time_proj` (written by the CPU), `linear_1`, `linear_2`. The ResNet blocks read `linear_2` directly.
- The CPU computes the 320 features. They depend only on the timestep. The steps follow `get_timestep_embedding` of diffusers in f32: -ln(10000) as f32, times i, divided by 160, exp, times t, then cos for the first half and sin for the second half (`flip_sin_to_cos=True`).
- The weights come from the f32 `unet.bin`, as for `conv_in`.

Command: `cargo test --release --test unet -- --nocapture`, in `engine/`. Device: Mac native.

```
planned activations: 111141888 bytes (111.1 MB)
after prepare: 3 bind groups, 31 buffers
time_embedding: largest error 9.5367431640625e-7, mean error 3.0501723813358694e-8, worst at [0, 144] (engine -4.7638474, golden -4.7638464), 1 submits, longest 3.7 ms, peak 119396672 bytes
conv_in: largest error 9.5367431640625e-7, mean error 2.2269295524637855e-8, worst at [0, 240, 11, 1] (engine -2.2016876, golden -2.2016885), 1 submits, longest 1.3 ms, peak 124634432 bytes
run 1: measured peak 124634432 bytes (124.6 MB); live now: activations 111141888, weights 8249600, uniforms 64, read-back 0; bind groups 3, buffers made in the run 2
time_embedding: largest error 9.5367431640625e-7, mean error 3.0501723813358694e-8, worst at [0, 144] (engine -4.7638474, golden -4.7638464), 1 submits, longest 0.8 ms, peak 124634432 bytes
conv_in: largest error 9.5367431640625e-7, mean error 2.2269295524637855e-8, worst at [0, 240, 11, 1] (engine -2.2016876, golden -2.2016885), 1 submits, longest 1.2 ms, peak 124634432 bytes
run 2: measured peak 124634432 bytes (124.6 MB); live now: activations 111141888, weights 8249600, uniforms 64, read-back 0; bind groups 3, buffers made in the run 2
test unet_matches_golden_block_by_block ... ok
```

The first run of this test used a tolerance of 1e9 for `time_embedding`, to measure the error first. It printed the same `time_embedding` error, 9.5367431640625e-7, with `longest 1.0 ms`. The tolerance is now 1e-4.

Command: `cargo run --release --example memory_plan`, in `engine/`. Device: Mac native.

```
unet forward pass: 565 ops, 489 tensors
unet activations, most live at one op: 84272128 bytes (84.3 MB) at down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.geglu
unet activations, planned: 22 buffers, 111141888 bytes (111.1 MB), largest buffer 41943040 bytes
stage 2 unet: weights 699997456 + activations 111141888 = 811139344 bytes (811.1 MB, 42.2% of the 1920 MB tab limit)
```

Command: `cargo test --release`, in `engine/`. Device: Mac native. Every test file passed. The library tests printed `test result: ok. 36 passed`. `cargo build --release --target wasm32-unknown-unknown` gave no warnings.

Calculations:

- The graph lost 3 ops (`timestep`, the SiLU in place, `time_embedding.silu`) and 2 tensors (`timestep`, `time_embedding.silu`): 568 - 3 = 565 and 491 - 2 = 489. The planned bytes did not change, 111,141,888. The removed tensors were 4 and 5,120 bytes.
- Weights: conv_in 47,360 + linear_1 (1280 x 320 x 4 = 1,638,400, bias 5,120) + linear_2 (1280 x 1280 x 4 = 6,553,600, bias 5,120) = 8,249,600 bytes. That is the printed number.
- Uniforms: conv_in 32 + two one-row layers of 3 sizes padded to 4 u32 = 16 bytes each. 32 + 16 + 16 = 64.
- Peak at the time embedding check: 111,141,888 + 8,249,600 + 64 + 1280 x 4 (read-back) = 119,396,672. At the conv_in check: 111,141,888 + 8,249,600 + 64 + 5,242,880 = 124,634,432. Both are the printed peaks.
- Buffers after prepare: 22 slots + 6 weight buffers + 3 uniforms = 31. Buffers in each run: 2 read-back copies, one per block.
- The worst value: 4.7638474 - 4.7638464 = 1.0e-6. Near 4.76, one f32 step is 2^-21 = 4.77e-7, so the error is 2 steps.
- linear_2 work: 1280 x 1280 = 1,638,400 multiply-adds. It reads 6,553,600 bytes of weights.

Mutation checks. Each bug made a test fail, and the code was restored:

```
frequencies divided by 159 in place of 160:
time_embedding: first mismatch, largest error 6.358811855316162e-1 is above 1e-4

sines first, then cosines:
time_embedding: first mismatch, largest error 4.711569786071777e0 is above 1e-4
test unet::time::tests::cosines_come_first ... FAILED

no SiLU before linear_2:
time_embedding: first mismatch, largest error 8.362144470214844e0 is above 1e-4

the one-row kernel skips the last vec4 of each row:
time_embedding: first mismatch, largest error 4.578876495361328e-2 is above 1e-4
```

Not measured on the iPhone. `UnetStage` has no WASM entry yet.

Conclusion: the UNet stage computes the time embedding on its planned slots and matches PyTorch to 9.5e-7. `prepare` makes 3 bind groups, and two runs make no new ones. Each run makes only its 2 read-back copies.

Next step: the first ResNet block, `down_blocks.0.resnets.0`. It needs GroupNorm + SiLU on slots, the time vector add, and conv1x1 on slots. There is no golden tensor inside a down block, so the test needs a new golden tensor from `dump_golden.py`, or the check waits for the end of `down_block_0`.

## 2026-11-18: First ResNet block, down_blocks.0.resnets.0, on its planned slots

This step continues "Time embedding on its planned slots". The UNet stage now runs the first ResNet block after `conv_in`. The previous entry said that no golden tensor exists inside a down block. The user did not choose between the two options. This step takes the cheaper option: a new script saves the tensor, and the existing golden files do not change.

Files added:

- `dump_unet_parts.py`: loads only the UNet, runs it on the golden `unet_input`, `timestep` and `text_encoder`, and saves the outputs of the modules in `PARTS`. It stops if its UNet output is not the saved `unet_output`. Today `PARTS` is `["down_blocks.0.resnets.0"]`.
- `engine/src/unet/kernels/group_norm.wgsl`, `group_norm.rs`: GroupNorm with 32 groups and a SiLU flag, between two slots.
- `engine/src/unet/kernels/add.wgsl`, `add.rs`: an in-place add. The addend has the size of the tensor, or one value per channel.
- `engine/src/unet/kernels/mod.rs`.
- `engine/src/unet/resnet.rs`: `bind_resnet` binds the 7 kernels of one ResNet block to the slots of the plan.

Files changed: `engine/src/unet/mod.rs` (the stage binds and runs the block), `engine/src/error.rs` (`NotImplemented`), `engine/tests/unet.rs` (tolerance of the block).

Design choices:

- The new golden script runs the UNet alone, not the whole pipeline. dump_golden.py ran for more than 64 minutes ("Float16 golden set after the UNet hooks"). The check of `unet_output` proves that the new tensor comes from the same run as the old ones.
- One workgroup of 256 invocations handles one group. It reads the group three times: mean, then variance, then the output. Two passes for the variance are more exact than the sum of squares in one pass. The variance is divided by the count, as in PyTorch.
- 32 groups give only 32 workgroups. GroupNorm is not the slow part at this size, so this is enough for now.
- The time vector add and the residual add use one kernel. The kind of add comes from the sizes in the plan.
- The block uses two submits. Each submit holds one 3x3 convolution: [norm1, time_emb_proj, conv1, time add] and [norm2, conv2, residual add].
- A block that changes the channel count needs `conv_shortcut`. `bind_resnet` stops with `NotImplemented` for such a block. down_blocks.0.resnets.0 keeps 320 channels.

Command: `uv run dump_unet_parts.py`, in the project folder. Device: Mac, Python through `uv run`, CPU. `time` printed `14.224 total`.

```
unet_output: largest difference from the saved file 0.000000e+00
golden/float32:
  down_blocks.0.resnets.0  shape=(1, 320, 64, 64) dtype=float32
```

Command: `cargo test --release --test unet -- --nocapture`, in `engine/`. Device: Mac native.

```
planned activations: 111141888 bytes (111.1 MB)
after prepare: 10 bind groups, 48 buffers
time_embedding: largest error 9.5367431640625e-7, mean error 3.0501723813358694e-8, worst at [0, 144] (engine -4.7638474, golden -4.7638464), 1 submits, longest 1.3 ms, peak 128416976 bytes
conv_in: largest error 9.5367431640625e-7, mean error 2.2269295524637855e-8, worst at [0, 240, 11, 1] (engine -2.2016876, golden -2.2016885), 1 submits, longest 1.2 ms, peak 133654736 bytes
down_blocks.0.resnets.0: largest error 1.049041748046875e-5, mean error 2.6353929483136085e-7, worst at [0, 208, 15, 4] (engine -3.4205034, golden -3.4205139), 2 submits, longest 28.5 ms, peak 133654736 bytes
run 1: measured peak 133654736 bytes (133.7 MB); live now: activations 111141888, weights 17269760, uniforms 208, read-back 0; bind groups 10, buffers made in the run 3
time_embedding: largest error 9.5367431640625e-7, mean error 3.0501723813358694e-8, worst at [0, 144] (engine -4.7638474, golden -4.7638464), 1 submits, longest 0.4 ms, peak 133654736 bytes
conv_in: largest error 9.5367431640625e-7, mean error 2.2269295524637855e-8, worst at [0, 240, 11, 1] (engine -2.2016876, golden -2.2016885), 1 submits, longest 0.4 ms, peak 133654736 bytes
down_blocks.0.resnets.0: largest error 1.049041748046875e-5, mean error 2.6353929483136085e-7, worst at [0, 208, 15, 4] (engine -3.4205034, golden -3.4205139), 2 submits, longest 14.1 ms, peak 133654736 bytes
run 2: measured peak 133654736 bytes (133.7 MB); live now: activations 111141888, weights 17269760, uniforms 208, read-back 0; bind groups 10, buffers made in the run 3
```

The first run of this test used a tolerance of 1e9 for the block, to measure the error first. It printed the same block error, 1.049041748046875e-5, with `longest 31.4 ms` in run 1 and `longest 14.0 ms` in run 2. The tolerance is now 1e-4.

Command: `cargo test --release`, in `engine/`. Device: Mac native. Every test file passed, and the library tests printed `test result: ok. 36 passed`. `cargo build --release --target wasm32-unknown-unknown` gave no warnings.

Calculations:

- New weights: norm1 320 x 4 x 2 = 2,560. time_emb_proj 320 x 1280 x 4 = 1,638,400, bias 1,280. conv1 320 x 320 x 9 x 4 = 3,686,400, bias 1,280. norm2 2,560. conv2 3,687,680. Sum 9,020,160. 8,249,600 + 9,020,160 = 17,269,760 bytes, the printed number.
- Uniforms: 3 convolutions x 32 + 3 one-row layers x 16 + 2 GroupNorms x 16 + 2 adds x 16 = 96 + 48 + 32 + 32 = 208 bytes.
- Bind groups: 3 + 7 = 10. Buffers after prepare: 31 + 10 weight buffers + 7 uniforms = 48. Each run makes 3 read-back copies, one per block.
- Peak: 111,141,888 + 17,269,760 + 208 + 5,242,880 = 133,654,736 bytes, the printed peak. The live activations stay at the planned 111,141,888.
- The worst value: 3.4205139 - 3.4205034 = 1.05e-5. That is 1.05e-5 / 3.42 = 3.1e-6 of the value.
- One convolution of the block: 320 x 64 x 64 x 320 x 9 = 3,774,873,600 multiply-adds. Two per block.
- The longest submit of run 2, 14.1 ms, holds one of them. 3,774,873,600 / 0.0141 s = 268 G multiply-adds per second on the Mac. The iPhone time is not known.

Mutation checks. Each bug made the test fail, and the code was restored:

```
variance divided by count - 1:
down_blocks.0.resnets.0: first mismatch, largest error 1.18255615234375e-4 is above 1e-4

eps 1e-6 in place of 1e-5:
down_blocks.0.resnets.0: first mismatch, largest error 3.4618377685546875e-4 is above 1e-4

the scale and shift of group 0 for every group:
down_blocks.0.resnets.0: first mismatch, largest error 6.857569456100464e0 is above 1e-4

no time vector add:
down_blocks.0.resnets.0: first mismatch, largest error 5.809128046035767e0 is above 1e-4

no SiLU after norm2:
down_blocks.0.resnets.0: first mismatch, largest error 1.592929744720459e1 is above 1e-4
```

The count - 1 bug is only 1.18 times the tolerance. A smaller bug of this kind would pass the test.

Not measured on the iPhone. `UnetStage` has no WASM entry yet. Each convolution submit at 64 x 64 must stay under 0.25 s on the iPhone. The test checks this only on the Mac.

Conclusion: the first ResNet block runs on its planned slots and matches PyTorch to 1.05e-5. `prepare` makes 10 bind groups, and two runs make no new ones.

Next step: the first transformer, `down_blocks.0.attentions.0`. It needs GroupNorm without SiLU (eps 1e-6), the linear kernel on slots for 4096 tokens, LayerNorm, the attention kernel on slots, GEGLU, and the layout change from [channels][pixels] to [tokens][channels] and back.

## 2026-11-18: First transformer, down_blocks.0.attentions.0, on its planned slots

This step continues "First ResNet block, down_blocks.0.resnets.0, on its planned slots". The UNet stage now runs the first transformer after the first ResNet block. The stage became a list of steps that `prepare` makes once.

Files added:

- `engine/src/unet/step.rs`: `Step` (one submit: its kernels and an optional check) and `Layout` (plan size to PyTorch shape).
- `engine/src/unet/transformer.rs`: `bind_transformer`, 7 steps for one transformer.
- `engine/src/unet/kernels/layer_norm.wgsl`, `layer_norm.rs`: LayerNorm over the channels of each token row.
- `engine/src/unet/kernels/geglu.wgsl`, `geglu.rs`: value x GELU(gate), with erf by Abramowitz and Stegun 7.1.26.
- `engine/src/unet/kernels/transpose_add.wgsl`, `transpose_add.rs`: token rows back to [channels][pixels], plus the transformer input.
- `engine/src/unet/kernels/weight.rs`: the reader of a norm scale or shift, shared by GroupNorm and LayerNorm.
- `engine/src/attention/slots.rs`: the attention kernel on slots, one dispatch and one uniform per chunk.
- `engine/src/linear.rs`: replaces `engine/src/vector_linear.rs`. `bind_vector_linear` (one row) and `bind_rows_linear` (token rows) share `load_linear`.

Files changed:

- `engine/src/dispatch.rs`: `BoundKernel`, the dispatches of one kernel with the buffers that only it owns. It replaces `SlotLayer`, `SlotConv`, `SlotLinear`, `SlotGroupNorm` and `SlotAdd`.
- `engine/src/layers/mod.rs`, `engine/src/layers/vector.rs`, `engine/src/conv.rs`: return `BoundKernel`.
- `engine/src/layers/linear.wgsl`, `engine/src/layers/linear.rs`: any row count of 1 or more.
- `engine/src/unet/kernels/group_norm.wgsl`, `group_norm.rs`: a flag that writes the output as token rows.
- `engine/src/unet/plan/transformer.rs`: the plan change at the start and the end of a transformer (see below).
- `engine/src/unet/resnet.rs`, `engine/src/unet/mod.rs`: steps. `UnetStage::run` takes `UnetInputs` (latent, timestep, text).
- `engine/src/block.rs`, `engine/tests/common/mod.rs`: block names are `&str` and `String`, not `&'static str`, because the steps make their names at run time.
- `dump_unet_parts.py`: 5 more parts. A transformer returns a tuple, so the hook takes item 0.
- `engine/tests/unet.rs`: the text input, and the tolerances of the 5 new checks.

Design choices:

- One `Step` is one submit. A check after a step reads its tensor before a later step can write over its slot. The 5 checks show where a wrong result starts.
- Self attention has a step of its own. It took 145.0 ms per run on the iPhone at 64 x 64 ("Attention on the iPhone"). to_out.0 has its own step after it, so the step stays at that time.
- Cross attention reads 77 text keys, so it fits one step with its add, norm2, q, k, v and to_out.0.
- GroupNorm writes token rows when the flag is set. So proj_in is a plain linear layer, and no layout-change kernel is needed at the start.
- At the end, proj_out writes token rows, and one kernel turns them into [channels][pixels] and adds the input. In the plan, this is a new tensor `<prefix>.output`, in place of an in-place add on `proj_out`. The plan grew by 16 tensors, one per transformer: 489 to 505. The ops stayed at 565, because one in-place add became one op. The planned bytes did not change.
- to_q, to_k and to_v have no bias (bias=False in diffusers). `Bias::Zero` gives them a buffer of zeros.
- The linear kernel took only row counts that are multiples of 4. The text has 77 rows. Now a row past the last one reads the last row, and its output is not written. The row of each of the 4 inputs is picked once, before the inner loop.
- WGSL has no erf. Formula 7.1.26 has a largest error of 1.5e-7.

Command: `uv run dump_unet_parts.py`, in the project folder. Device: Mac, Python through `uv run`, CPU.

```
unet_output: largest difference from the saved file 0.000000e+00
golden/float32:
  down_blocks.0.resnets.0  shape=(1, 320, 64, 64) dtype=float32
  down_blocks.0.attentions.0.proj_in shape=(1, 4096, 320) dtype=float32
  down_blocks.0.attentions.0.transformer_blocks.0.attn1 shape=(1, 4096, 320) dtype=float32
  down_blocks.0.attentions.0.transformer_blocks.0.attn2 shape=(1, 4096, 320) dtype=float32
  down_blocks.0.attentions.0.transformer_blocks.0.ff shape=(1, 4096, 320) dtype=float32
  down_blocks.0.attentions.0 shape=(1, 320, 64, 64) dtype=float32
```

Command: `cargo test --release --test unet -- --nocapture`, in `engine/`. Device: Mac native. This run uses the final linear kernel. The lines before the transformer are the same as in "First ResNet block, down_blocks.0.resnets.0, on its planned slots", except the times and the peaks:

```
planned activations: 111141888 bytes (111.1 MB)
after prepare: 63 bind groups, 133 buffers
down_blocks.0.attentions.0.proj_in: largest error 2.0265579223632813e-6, mean error 1.7411045664245117e-7, worst at [0, 3968, 214] (engine -1.788565, golden -1.788563), 1 submits, longest 1.5 ms, peak 144504144 bytes
down_blocks.0.attentions.0.transformer_blocks.0.attn1: largest error 1.5497207641601563e-6, mean error 1.368120758123842e-7, worst at [0, 508, 67] (engine 0.68202174, golden 0.6820233), 3 submits, longest 70.3 ms, peak 144504144 bytes
down_blocks.0.attentions.0.transformer_blocks.0.attn2: largest error 1.5497207641601563e-6, mean error 7.2727715204942e-8, worst at [0, 1634, 72] (engine 0.7009412, golden 0.70094275), 1 submits, longest 8.4 ms, peak 144504144 bytes
down_blocks.0.attentions.0.transformer_blocks.0.ff: largest error 3.814697265625e-6, mean error 2.4115551267632573e-7, worst at [0, 1044, 72] (engine -1.4832414, golden -1.4832376), 1 submits, longest 7.8 ms, peak 144504144 bytes
down_blocks.0.attentions.0: largest error 9.775161743164063e-6, mean error 3.495430521249787e-7, worst at [0, 208, 15, 4] (engine -2.3248463, golden -2.324856), 1 submits, longest 1.4 ms, peak 144504144 bytes
run 1: measured peak 144504144 bytes (144.5 MB); live now: activations 111141888, weights 28117760, uniforms 1616, read-back 0; bind groups 63, buffers made in the run 8
down_blocks.0.attentions.0.proj_in: largest error 2.0265579223632813e-6, mean error 1.7411045664245117e-7, worst at [0, 3968, 214] (engine -1.788565, golden -1.788563), 1 submits, longest 1.6 ms, peak 144504144 bytes
down_blocks.0.attentions.0.transformer_blocks.0.attn1: largest error 1.5497207641601563e-6, mean error 1.368120758123842e-7, worst at [0, 508, 67] (engine 0.68202174, golden 0.6820233), 3 submits, longest 67.4 ms, peak 144504144 bytes
down_blocks.0.attentions.0.transformer_blocks.0.attn2: largest error 1.5497207641601563e-6, mean error 7.2727715204942e-8, worst at [0, 1634, 72] (engine 0.7009412, golden 0.70094275), 1 submits, longest 4.1 ms, peak 144504144 bytes
down_blocks.0.attentions.0.transformer_blocks.0.ff: largest error 3.814697265625e-6, mean error 2.4115551267632573e-7, worst at [0, 1044, 72] (engine -1.4832414, golden -1.4832376), 1 submits, longest 7.7 ms, peak 144504144 bytes
down_blocks.0.attentions.0: largest error 9.775161743164063e-6, mean error 3.495430521249787e-7, worst at [0, 208, 15, 4] (engine -2.3248463, golden -2.324856), 1 submits, longest 1.3 ms, peak 144504144 bytes
run 2: measured peak 144504144 bytes (144.5 MB); live now: activations 111141888, weights 28117760, uniforms 1616, read-back 0; bind groups 63, buffers made in the run 8
```

The first run of this test used a tolerance of 1e9 for the 5 new checks, to measure the errors first. It printed the same 5 errors. All tolerances are now 1e-4.

Command: `cargo run --release --example memory_plan`, in `engine/`. Device: Mac native.

```
unet forward pass: 565 ops, 505 tensors
unet activations, most live at one op: 84272128 bytes (84.3 MB) at down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.geglu
unet activations, planned: 22 buffers, 111141888 bytes (111.1 MB), largest buffer 41943040 bytes
stage 2 unet: weights 699997456 + activations 111141888 = 811139344 bytes (811.1 MB, 42.2% of the 1920 MB tab limit)
```

Command: `cargo test --release`, in `engine/`. Device: Mac native. With the final linear kernel, every test file passed except `attention` (see "Failed runs"). The library tests printed `test result: ok. 37 passed`. `cargo build --release --target wasm32-unknown-unknown` gave no warnings.

Calculations:

- New weights: 28,117,760 - 17,269,760 = 10,848,000 bytes. By tensor: GroupNorm 2,560. proj_in 409,600 + 1,280. norm1, norm2, norm3 3 x 2,560 = 7,680. attn1 to_q, to_k, to_v 3 x 409,600 = 1,228,800, with 3 zero biases of 1,280. attn1 to_out.0 410,880. attn2 to_q 409,600 + 1,280 zeros. attn2 to_k and to_v 2 x 320 x 1024 x 4 = 2,621,440, with 2 zero biases of 1,280. attn2 to_out.0 410,880. ff.net.0.proj 2560 x 320 x 4 = 3,276,800 + 10,240. ff.net.2 320 x 1280 x 4 = 1,638,400 + 1,280. proj_out 410,880. The sum is 10,848,000.
- Bind groups: 10 + 53 = 63. The transformer has 53: 2 + 4 + 16 attention chunks + 1 + (1 + 4 + 16 + 1) + 5 + 3. 4096 query rows / 256 rows per chunk = 16 chunks per attention.
- Uniforms: 1,616 - 208 = 1,408 bytes. The two ResNet GroupNorms grew from 16 to 32 bytes (6 sizes padded to 8), which is 32 bytes. The transformer uniforms are 1,376 bytes: 32 attention chunks x 32 = 1,024, 1 GroupNorm 32, 3 LayerNorms x 16, 12 linear layers x 16 = 192, 3 adds x 16, GEGLU 16, transpose add 16.
- Buffers after prepare: 48 + 32 weight buffers + 53 uniforms = 133. Each run makes 8 buffers, one read-back copy per check.
- Peak: 111,141,888 + 28,117,760 + 1,616 + 5,242,880 = 144,504,144 bytes, the printed peak. The largest read-back copies are 4096 x 320 x 4 = 5,242,880 bytes.
- The transformer output error: 2.324856 - 2.3248463 = 9.7e-6, which is 9.7e-6 / 2.32 = 4.2e-6 of the value.
- The self attention step took 67.4 ms in run 2 on the Mac. The attention test measured 67.6 ms per run of the same size. On the iPhone, the attention test measured 145.0 ms. The iPhone time of this step is not measured.

Linear kernel speed. Command: `cargo test --release --test layers -- --nocapture`, the old kernel and the new kernel one after the other. Device: Mac native.

```
old: linear: 19.00 GB/s median, 19.11 GB/s best (30 submits, 128 runs per submit, 73.4 ms median submit, 82.9 ms longest), largest error 1.1920929e-6
new: linear: 17.22 GB/s median, 17.27 GB/s best (30 submits, 128 runs per submit, 81.0 ms median submit, 93.4 ms longest), largest error 1.1920929e-6
old: linear: 4.85 GB/s median, 5.10 GB/s best (30 submits, 8 runs per submit, 18.0 ms median submit, 24.9 ms longest), largest error 1.1920929e-6
new: linear: 4.63 GB/s median, 4.66 GB/s best (30 submits, 32 runs per submit, 75.2 ms median submit, 78.9 ms longest), largest error 1.1920929e-6
```

The first version picked the rows with `min` inside the inner loop. It was 17.22 / 19.00 = 0.906 of the old speed, so 9.4% slower. Rejected. The second pair is not valid: both kernels fell to about 4.8 GB/s, because other work used the Mac at that time. The final version picks the rows before the loop:

```
old: linear: 5.13 GB/s median, 5.16 GB/s best (30 submits, 32 runs per submit, 67.9 ms median submit, 70.8 ms longest), largest error 1.1920929e-6
hoisted: linear: 18.85 GB/s median, 18.99 GB/s best (30 submits, 32 runs per submit, 18.5 ms median submit, 71.0 ms longest), largest error 1.1920929e-6
old: linear: 19.06 GB/s median, 19.14 GB/s best (30 submits, 128 runs per submit, 73.2 ms median submit, 131.7 ms longest), largest error 1.1920929e-6
hoisted: linear: 18.03 GB/s median, 19.08 GB/s best (30 submits, 128 runs per submit, 77.4 ms median submit, 137.1 ms longest), largest error 1.1920929e-6
old: linear: 18.78 GB/s median, 19.15 GB/s best (30 submits, 128 runs per submit, 74.3 ms median submit, 109.0 ms longest), largest error 1.1920929e-6
hoisted: linear: 18.98 GB/s median, 19.14 GB/s best (30 submits, 128 runs per submit, 73.5 ms median submit, 87.7 ms longest), largest error 1.1920929e-6
```

The first pair is not valid for the same reason (the old kernel at 5.13 GB/s). In the other two pairs, the best speeds are 19.08 and 19.14 GB/s for the final kernel, and 19.14 and 19.15 GB/s for the old one. That is the same within the noise of these runs.

Mutation checks. Each bug made the test fail, and the code was restored:

```
GELU with the tanh formula in place of erf:
down_blocks.0.attentions.0.transformer_blocks.0.ff: first mismatch, largest error 7.345676422119141e-4 is above 1e-4

transformer GroupNorm eps 1e-5 in place of 1e-6:
down_blocks.0.attentions.0.transformer_blocks.0.attn2: first mismatch, largest error 1.4030933380126953e-4 is above 1e-4

cross attention reads keys from the image rows, not the text (prepare stops):
down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight has shape [320, 1024], the layer needs [out, 320]

self attention result not added before norm2:
down_blocks.0.attentions.0.transformer_blocks.0.attn2: first mismatch, largest error 1.2540435791015625e0 is above 1e-4

GEGLU halves swapped:
down_blocks.0.attentions.0.transformer_blocks.0.ff: first mismatch, largest error 4.758825302124023e0 is above 1e-4
```

The eps bug shows up only at attn2, not at proj_in or attn1, and it is only 1.40 times the tolerance. A smaller bug of this kind would pass.

Failed runs: the attention test failed 2 times out of 4 runs after the linear kernel change, each time on the 0.25 s limit, in a different case:

```
attention_self: attention_self: timed submit 9 submit with 1 runs took 435.5 ms, the limit is 250.0 ms
attention_cross: attention_cross: timed submit 16 submit with 32 runs took 381.5 ms, the limit is 250.0 ms
```

In the same runs, the median submits were 67.6 ms and 57.7 ms. This step did not change `attention.wgsl` or the attention test. `ps -Ao pcpu,comm -r` showed other work on the Mac: `fseventsd` 73.9%, `rm` 68.6%, `mds_stores` 62.4%, OrbStack 35.2%, and an iOS Simulator app (`Kibi.app`) 10.0%. These are not part of this project. The two other runs of the attention test passed. The same load can stop the UNet test on the 0.25 s limit.

Not measured on the iPhone. `UnetStage` has no WASM entry yet.

Conclusion: the first transformer runs on its planned slots and matches PyTorch to 9.8e-6. All 8 checks of the stage pass on two runs. `prepare` makes 63 bind groups and 133 buffers, and a run makes only its 8 read-back copies.

Next step: the rest of down level 0 (resnets.1, attentions.1, the downsample) with the same binders, checked against `down_block_0`. Before that, the user should run `cargo test --release` again when the Mac is idle, to see the attention test pass.

## 2026-11-18: Down level 0 on its planned slots, and checks that the caller picks

This step continues "First transformer, down_blocks.0.attentions.0, on its planned slots". The UNet stage now runs all of down level 0: 2 pairs of ResNet block and transformer, then the downsample. The session restarted after `dump_unet_parts.py` saved the two new golden tensors. The engine still built, and the two files were on disk, so that run was not repeated.

Files added:

- `engine/src/unet/down.rs`: `bind_down_level` binds one down level from the level layout of the plan.

Files changed:

- `dump_unet_parts.py`: `PARTS` has `down_blocks.0.resnets.1` and `down_blocks.0.attentions.1` too.
- `engine/src/unet/plan/forward.rs`, `engine/src/unet/plan/mod.rs`: `LEVEL_CHANNELS`, `DOWN_LAYERS` and `DOWN_NAMES` are `pub(crate)`, so the stage follows the same layout as the plan.
- `engine/src/unet/step.rs`: a step has a list of checks, not one. On level 3, the last ResNet step will also end the level.
- `engine/src/unet/mod.rs`: `prepare` binds down level 0 with `bind_down_level`. `run` takes `wanted`, the golden names to check.
- `engine/tests/unet.rs`: 3 runs, and 2 new assertions.

Design choices:

- The caller names the checks it wants. Each transformer has 5 checks, and the 16 transformers would need 80 golden files. Now the steps keep their checks, but only the wanted ones read back. A real generation passes an empty list, so it reads nothing back.
- The test asserts that a run shows exactly the parts of its tolerance list, in data-flow order. So a part in the list that the stage never shows fails the test, and no golden file is skipped without a message.
- Run 2 wants no checks, as a real generation. It must make 0 buffers and 0 bind groups. Runs 1 and 3 check all 11 parts.
- The downsample is the 3x3 convolution of "Linear and convolution kernels" with stride 2 and padding 1 (downsample_padding=1 in unet/config.json).

Command: `uv run dump_unet_parts.py`, in the project folder. Device: Mac, Python through `uv run`, CPU.

```
unet_output: largest difference from the saved file 0.000000e+00
golden/float32:
  down_blocks.0.resnets.0  shape=(1, 320, 64, 64) dtype=float32
  down_blocks.0.attentions.0.proj_in shape=(1, 4096, 320) dtype=float32
  down_blocks.0.attentions.0.transformer_blocks.0.attn1 shape=(1, 4096, 320) dtype=float32
  down_blocks.0.attentions.0.transformer_blocks.0.attn2 shape=(1, 4096, 320) dtype=float32
  down_blocks.0.attentions.0.transformer_blocks.0.ff shape=(1, 4096, 320) dtype=float32
  down_blocks.0.attentions.0 shape=(1, 320, 64, 64) dtype=float32
  down_blocks.0.resnets.1  shape=(1, 320, 64, 64) dtype=float32
  down_blocks.0.attentions.1 shape=(1, 320, 64, 64) dtype=float32
```

Command: `cargo test --release --test unet -- --nocapture`, in `engine/`. Device: Mac native. The lines of the 8 older parts are the same as in "First transformer, down_blocks.0.attentions.0, on its planned slots", except the times and the peaks. The first run used a tolerance of 1e9 for the 3 new parts, to measure the errors first:

```
planned activations: 111141888 bytes (111.1 MB)
after prepare: 124 bind groups, 238 buffers
down_blocks.0.resnets.1: largest error 1.1920928955078125e-5, mean error 6.171330160231747e-7, worst at [0, 208, 45, 15] (engine -1.4316056, golden -1.4316175), 2 submits, longest 14.5 ms, peak 168061568 bytes
down_blocks.0.attentions.1: largest error 8.821487426757813e-6, mean error 5.815527921282371e-7, worst at [0, 208, 13, 32] (engine -0.5012994, golden -0.50129056), 7 submits, longest 68.4 ms, peak 168061568 bytes
down_block_0: largest error 1.0967254638671875e-5, mean error 1.2153876178311317e-6, worst at [0, 248, 15, 31] (engine -5.068474, golden -5.068463), 1 submits, longest 8.1 ms, peak 168061568 bytes
run 1 (11 checks): measured peak 168061568 bytes (168.1 MB); live now: activations 111141888, weights 51673600, uniforms 3200, read-back 0; bind groups 124, buffers made in the run 11
run 2 (0 checks): measured peak 168061568 bytes (168.1 MB); live now: activations 111141888, weights 51673600, uniforms 3200, read-back 0; bind groups 124, buffers made in the run 0
down_blocks.0.resnets.1: largest error 1.1920928955078125e-5, mean error 6.171330160231747e-7, worst at [0, 208, 45, 15] (engine -1.4316056, golden -1.4316175), 2 submits, longest 13.9 ms, peak 168061568 bytes
down_blocks.0.attentions.1: largest error 8.821487426757813e-6, mean error 5.815527921282371e-7, worst at [0, 208, 13, 32] (engine -0.5012994, golden -0.50129056), 7 submits, longest 68.7 ms, peak 168061568 bytes
down_block_0: largest error 1.0967254638671875e-5, mean error 1.2153876178311317e-6, worst at [0, 248, 15, 31] (engine -5.068474, golden -5.068463), 1 submits, longest 8.0 ms, peak 168061568 bytes
run 3 (11 checks): measured peak 168061568 bytes (168.1 MB); live now: activations 111141888, weights 51673600, uniforms 3200, read-back 0; bind groups 124, buffers made in the run 11
test unet_matches_golden_block_by_block ... ok
```

With all tolerances at 1e-4, the same command printed the same errors. The lines of the last part and the runs:

```
down_block_0: largest error 1.0967254638671875e-5, mean error 1.2153876178311317e-6, worst at [0, 248, 15, 31] (engine -5.068474, golden -5.068463), 1 submits, longest 8.3 ms, peak 168061568 bytes
run 1 (11 checks): measured peak 168061568 bytes (168.1 MB); live now: activations 111141888, weights 51673600, uniforms 3200, read-back 0; bind groups 124, buffers made in the run 11
run 2 (0 checks): measured peak 168061568 bytes (168.1 MB); live now: activations 111141888, weights 51673600, uniforms 3200, read-back 0; bind groups 124, buffers made in the run 0
run 3 (11 checks): measured peak 168061568 bytes (168.1 MB); live now: activations 111141888, weights 51673600, uniforms 3200, read-back 0; bind groups 124, buffers made in the run 11
```

Command: `cargo test --release`, in `engine/`. Device: Mac native. Every test file passed, the attention test too. The library tests printed `test result: ok. 37 passed`. `cargo build --release --target wasm32-unknown-unknown` gave no warnings.

Calculations:

- New weights: 51,673,600 - 28,117,760 = 23,555,840 bytes = resnets.1 9,020,160 + attentions.1 10,848,000 + the downsample 3,687,680 (320 x 320 x 9 x 4 + 1,280). The first two are the numbers of resnets.0 and attentions.0, because the sizes are the same.
- Bind groups: 63 + 7 + 53 + 1 = 124. Buffers after prepare: 133 + 17 + 85 + 3 = 238.
- Uniforms: 1,616 + 176 (ResNet: 2 convolutions x 32, 1 linear 16, 2 GroupNorms x 32, 2 adds x 16) + 1,376 (transformer) + 32 (downsample) = 3,200 bytes.
- Peak: 111,141,888 + 51,673,600 + 3,200 + 5,242,880 = 168,061,568 bytes, the printed peak. The live activations stay at the planned 111,141,888 in all 3 runs.
- Submits of down level 0: 2 layers x (2 ResNet + 7 transformer) + 1 downsample = 19. With the time embedding and conv_in, a run has 21 submits.
- down_block_0 worst value: 5.068474 - 5.068463 = 1.1e-5, which is 1.1e-5 / 5.07 = 2.2e-6 of the value.
- The longest submit of a run is the self attention, 68.7 ms on the Mac. The iPhone time is not measured.

Mutation checks. Each bug made the test fail, and the code was restored:

```
downsample with stride 1 (prepare stops):
down_blocks.0.downsamplers.0.conv: the plan has size [320, 32, 32], the kernel writes [320, 64, 64]

resnets.1 reads conv_in, not the first transformer:
down_blocks.0.resnets.1: first mismatch, largest error 7.321195960044861e0 is above 1e-4

the level output is not checked (the list assertion fails):
assertion `left == right` failed: run 1: the parts that the stage showed
  left: ["time_embedding", "conv_in", "down_blocks.0.resnets.0", "down_blocks.0.attentions.0.proj_in", "down_blocks.0.attentions.0.transformer_blocks.0.attn1", "down_blocks.0.attentions.0.transformer_blocks.0.attn2", "down_blocks.0.attentions.0.transformer_blocks.0.ff", "down_blocks.0.attentions.0", "down_blocks.0.resnets.1", "down_blocks.0.attentions.1"]
 right: ["time_embedding", "conv_in", "down_blocks.0.resnets.0", "down_blocks.0.attentions.0.proj_in", "down_blocks.0.attentions.0.transformer_blocks.0.attn1", "down_blocks.0.attentions.0.transformer_blocks.0.attn2", "down_blocks.0.attentions.0.transformer_blocks.0.ff", "down_blocks.0.attentions.0", "down_blocks.0.resnets.1", "down_blocks.0.attentions.1", "down_block_0"]

the level reads the transformer stream, not its output (prepare stops):
down_blocks.0.resnets.1.norm1: the plan has size [320, 64, 64], the kernel writes [4096, 320, 1]
```

Rejected design: `bind_down_level` first returned the level output name too. Nothing reads it before level 1 exists, so the build warned `field `output` is never read`. The function returns only the steps now.

Not measured on the iPhone. `UnetStage` has no WASM entry yet.

Conclusion: the UNet stage runs down level 0 on its planned slots, and `down_block_0` matches PyTorch to 1.1e-5. A run that wants no checks makes no buffer and no bind group.

Next step: down level 1. Its first ResNet block changes 320 channels to 640, so `bind_resnet` needs the 1x1 convolution `conv_shortcut` on slots. `bind_down_level` will then also return the level output.

## 2026-11-18: The whole UNet on its planned slots

The user asked to complete all the leftover steps of the original request in one prompt. This replaces the "one step per prompt" rule for this prompt only. The leftover steps are: the rest of the UNet, the text encoder stage, and the staged run of text encoder, UNet and decoder with the planned and the measured peaks. This entry is the first of them. It continues "Down level 0 on its planned slots, and checks that the caller picks".

Files added:

- `engine/src/unet/bind.rs`: `bind_unet` binds the whole forward pass, level by level, in the order of diffusers.
- `engine/src/unet/middle.rs`: the middle block.
- `engine/src/unet/up.rs`: one up level: concat, ResNet block, transformer (levels 1 to 3), and the upsample (levels 0 to 2).
- `engine/src/unet/kernels/concat.wgsl`, `concat.rs`: torch.cat along the channels. In the [channels][pixels] layout, this is the first tensor followed by the second.
- `engine/src/unet/kernels/upsample.rs`: nearest 2x upsample on slots, with the TAESD shader `taesd/upsample.wgsl`.

Files changed:

- `engine/src/conv.rs`: `bind_conv1x1_into`, the 1x1 convolution on slots, for `conv_shortcut`.
- `engine/src/unet/resnet.rs`: a block whose plan has `conv_shortcut` adds the 1x1 convolution of its input. `NotImplemented` is gone from `engine/src/error.rs`, because nothing returns it now.
- `engine/src/unet/down.rs`: returns the level output and its skip tensors.
- `engine/src/unet/mod.rs`: `prepare` calls `bind_unet`. `run` returns the UNet output, [1, 4, 64, 64].
- `engine/src/taesd/kernels.rs`, `engine/src/taesd/mod.rs`: the upsample shader is `pub(crate)`.
- `engine/src/unet/plan/forward.rs`, `engine/src/unet/plan/mod.rs`: `UP_NAMES` is `pub(crate)`.
- `engine/tests/unet.rs`: 20 checks, and the returned output of every run is compared with the golden `unet_output`.

Design choices:

- `run` always reads back the UNet output (65,536 bytes). The scheduler step and the decoder stage need it after the UNet stage is freed. So a run with no checks makes exactly 1 buffer.
- Each concat has a step of its own, before its ResNet block. The ResNet block keeps its two steps with one 3x3 convolution each.
- The up path reads the skip tensors in reverse order, 3 per level, the same rule as `up_path` in `unet/plan/forward.rs`.
- The upsample and its convolution share one step.
- The tolerances of the deeper parts come from a float64 run of PyTorch (see below): 2 times the float32 rounding of PyTorch at that part, rounded up. A fixed 1e-4 is below the rounding of PyTorch itself from down_block_1 on.

Command: `cargo test --release --test unet -- --nocapture`, in `engine/`. Device: Mac native. The 11 older parts print the same errors as in "Down level 0 on its planned slots, and checks that the caller picks". The new lines of run 1, and the three run lines:

```
planned activations: 111141888 bytes (111.1 MB)
after prepare: 742 bind groups, 1546 buffers
down_block_1: largest error 9.965896606445313e-5, mean error 3.2429745172701805e-6, worst at [0, 324, 0, 15] (engine -6.791672, golden -6.7915726), 19 submits, longest 13.5 ms, peak 3580344256 bytes
down_block_2: largest error 9.965896606445313e-5, mean error 1.0528115880692467e-5, worst at [0, 408, 7, 4] (engine -3.8616314, golden -3.8615317), 19 submits, longest 22.6 ms, peak 3580344256 bytes
down_block_3: largest error 1.0013580322265625e-4, mean error 1.0892237503412616e-5, worst at [0, 408, 7, 4] (engine -2.6965625, golden -2.6964624), 4 submits, longest 12.4 ms, peak 3580344256 bytes
mid_block: largest error 9.918212890625e-5, mean error 1.1105096484698152e-5, worst at [0, 408, 7, 4] (engine -3.4822924, golden -3.4821932), 11 submits, longest 10.9 ms, peak 3580344256 bytes
up_block_0: largest error 7.82012939453125e-5, mean error 2.429610630561019e-6, worst at [0, 131, 5, 4] (engine 13.368305, golden 13.368227), 10 submits, longest 23.3 ms, peak 3580344256 bytes
up_block_1: largest error 3.204345703125e-4, mean error 2.6899862131130047e-5, worst at [0, 624, 20, 30] (engine 41.881878, golden 41.881557), 31 submits, longest 49.7 ms, peak 3580344256 bytes
up_block_2: largest error 3.325939178466797e-4, mean error 1.3365193069968597e-5, worst at [0, 236, 48, 25] (engine 2.9820182, golden 2.9816856), 31 submits, longest 51.7 ms, peak 3585587136 bytes
up_block_3: largest error 1.3780593872070313e-4, mean error 3.233692720971959e-6, worst at [0, 72, 30, 24] (engine 3.7908752, golden 3.7907374), 30 submits, longest 68.5 ms, peak 3585587136 bytes
unet_output: largest error 4.0531158447265625e-6, mean error 4.195402265594339e-7, worst at [0, 0, 30, 24] (engine -0.8219042, golden -0.8219001), 1 submits, longest 2.5 ms, peak 3585587136 bytes
run 1 (20 checks): returned output error 4.0531158447265625e-6, 16384 values; measured peak 3585587136 bytes (3585.6 MB); live now: activations 111141888, weights 3463942416, uniforms 17072, read-back 0; bind groups 742, buffers made in the run 21
run 2 (0 checks): returned output error 4.0531158447265625e-6, 16384 values; measured peak 3585587136 bytes (3585.6 MB); live now: activations 111141888, weights 3463942416, uniforms 17072, read-back 0; bind groups 742, buffers made in the run 1
run 3 (20 checks): returned output error 4.0531158447265625e-6, 16384 values; measured peak 3585587136 bytes (3585.6 MB); live now: activations 111141888, weights 3463942416, uniforms 17072, read-back 0; bind groups 742, buffers made in the run 21
```

Run 3 printed the same errors as run 1. The first run of the new parts used a tolerance of 1e9, and printed the same errors.

Command: a script in the scratchpad (not in the project). It loads the PyTorch UNet in float64, runs it on the float32 golden inputs, and prints the largest difference of each golden float32 file from the float64 result. Device: Mac, Python through `uv run`, CPU.

```
down_block_0: float32 PyTorch vs float64 PyTorch largest difference 2.410e-05, largest value 13.197
down_block_1: float32 PyTorch vs float64 PyTorch largest difference 1.457e-04, largest value 43.877
down_block_2: float32 PyTorch vs float64 PyTorch largest difference 1.234e-04, largest value 48.459
down_block_3: float32 PyTorch vs float64 PyTorch largest difference 1.431e-04, largest value 40.499
mid_block: float32 PyTorch vs float64 PyTorch largest difference 1.430e-04, largest value 39.485
up_block_0: float32 PyTorch vs float64 PyTorch largest difference 1.441e-04, largest value 37.177
up_block_1: float32 PyTorch vs float64 PyTorch largest difference 4.852e-04, largest value 105.033
up_block_2: float32 PyTorch vs float64 PyTorch largest difference 3.666e-04, largest value 84.942
up_block_3: float32 PyTorch vs float64 PyTorch largest difference 3.024e-04, largest value 31.900
unet_output: float32 PyTorch vs float64 PyTorch largest difference 1.128e-05, largest value 4.196
```

At every part, the engine is closer to PyTorch float32 than PyTorch float32 is to float64. For example, at up_block_1 the engine is 3.20e-4 from PyTorch float32, and PyTorch float32 is 4.85e-4 from float64. At unet_output: 4.05e-6 and 1.13e-5. So the larger errors of the deeper parts are float32 rounding, not a kernel fault.

The tolerances, 2 x the float64 difference, rounded up: down_block_1, down_block_2, down_block_3, mid_block and up_block_0 3e-4 (2 x 1.46e-4 = 2.9e-4 at most). up_block_1 and up_block_2 1e-3 (2 x 4.85e-4 = 9.7e-4). up_block_3 7e-4 (2 x 3.02e-4 = 6.0e-4). unet_output 3e-5 (2 x 1.13e-5 = 2.3e-5).

Command: `cargo test --release`, in `engine/`. Device: Mac native. Every test file passed. The library tests printed `test result: ok. 37 passed`. `cargo build --release --target wasm32-unknown-unknown` gave no warnings. `cargo run --release --example memory_plan` printed the same plan as before: `unet activations, planned: 22 buffers, 111141888 bytes (111.1 MB), largest buffer 41943040 bytes`.

Calculations:

- Weights on the GPU: the tensors of `unet.json`, 3,463,642,896 bytes (a Python sum over its 686 entries), plus the zero biases of to_q, to_k and to_v: 16 transformers x 6 layers, 299,520 bytes. 3,463,642,896 + 299,520 = 3,463,942,416, the printed number.
- Peak: 111,141,888 + 3,463,942,416 + 17,072 = 3,575,101,376 bytes live. The peak is 10,485,760 bytes more: the read-back copy of up_block_2, 640 x 64 x 64 x 4. 3,575,101,376 + 10,485,760 = 3,585,587,136, the printed peak.
- The live activations stay at the planned 111,141,888 bytes in all 3 runs.
- Bind groups: 742 after prepare, and no new one in any run. Run 2 made 1 buffer, the output copy.
- With these f32 weights, the UNet stage needs 3,585.6 MB, more than the 1,920 MB tab limit. The phone stage needs the packed weights of "Packed 4-bit and 8-bit weights, converter and kernels": 699,997,456 + 111,141,888 = 811,139,344 bytes in the plan. The slot kernels do not read packed weights yet.
- The longest submit on the Mac: 72.8 ms (up_block_3, the self attention at 64 x 64). The iPhone times are not measured.

Mutation checks. Each bug made the test fail, and the code was restored:

```
concat with the skip tensor first:
up_block_0: first mismatch, largest error 3.626484251022339e1 is above 3e-4

skip tensors in creation order, not reversed (prepare stops):
up_blocks.0.resnets.0.concat: the plan has size [2560, 8, 8], the kernel writes [1600, 8, 8]

no SiLU in conv_norm_out:
unet_output: first mismatch, largest error 4.93243670463562e0 is above 3e-5
```

Not measured on the iPhone. `UnetStage` has no WASM entry yet.

Conclusion: the whole UNet runs on 22 planned buffers and 742 bind groups that `prepare` makes once. Its output matches PyTorch to 4.05e-6. A run without checks makes 1 buffer and no bind group.

Next step: the text encoder stage.

## 2026-11-18: Text encoder stage on its planned slots

This is the second of the leftover steps in "The whole UNet on its planned slots". The text encoder (CLIPTextModel, 23 layers) now runs as a stage like the UNet: an activation plan, slots made once, bind groups made once, and steps.

Files added:

- `dump_text_parts.py`: saves the 77 token ids of the golden prompt (`text_input_ids`, as float32), the sigma of the one step (`sigma`), and two tensors inside the text encoder (`text_embeddings`, `text_layer_0`). It stops if its text encoder output or its timestep differs from the saved files.
- `engine/src/text/plan.rs`: the forward pass as ops, and its plan.
- `engine/src/text/embed.rs`: token embedding + position embedding, on the CPU.
- `engine/src/text/bind.rs`: one step per layer, and one for final_layer_norm.
- `engine/src/text/mod.rs`: `TextStage`.
- `engine/src/kernels/gelu.wgsl`, `gelu.rs`: GELU in place, for the MLP.
- `engine/src/kernels/erf.wgsl`: the erf of GEGLU, in its own file. Rust puts it in front of the GEGLU and GELU shaders with `concat!`, because WGSL has no include.
- `engine/tests/text.rs`: the test of the stage.

Files changed:

- `engine/src/unet/kernels/` moved to `engine/src/kernels/`, and `engine/src/unet/step.rs` moved to `engine/src/step.rs`. Both stages use them now.
- `engine/src/attention/attention.wgsl`, `mod.rs`, `slots.rs`: a `causal` flag. With it, query row i reads only keys 0 to i. The attention test cases and the UNet pass 0.
- `engine/src/weights/mod.rs`, `index.rs`, `file.rs`, `in_memory.rs`: `WeightSource::read_row` reads one row of a float32 tensor, with `WeightEntry::row_range`.
- `engine/src/lib.rs`: exports `TextStage`.

Design choices:

- The engine takes token ids, not text. A BPE tokenizer is a separate part. The test reads the ids that the Python tokenizer gave.
- The token table is 49,408 x 1024 x 4 = 202,375,168 bytes. It does not fit in one 128 MiB binding, and a run needs only 77 of its rows. So the CPU reads 77 rows with `read_row` and adds the position embeddings, and the table never goes to the GPU.
- One step per layer. A layer has 0.97 G multiply-adds (see below). The UNet has submits with 3.77 G.
- 77 query rows fit in one attention chunk, so each layer has 1 attention dispatch.
- The LayerNorm kernel of the transformers has eps 1e-5, which is also layer_norm_eps of the text encoder.

Command: `uv run dump_text_parts.py`, in the project folder. Device: Mac, Python through `uv run`, CPU. `time` printed `10.111 total`.

```
text_encoder: largest difference from the saved file 0.000000e+00
timestep: largest difference from the saved file 0.000000e+00
golden/float32:
  text_input_ids       shape=(1, 77) dtype=float32
  sigma                shape=(1,) dtype=float32
  text_embeddings      shape=(1, 77, 1024) dtype=float32
  text_layer_0         shape=(1, 77, 1024) dtype=float32
sigma 14.614646911621094
```

Command: `cargo test --release --test text -- --nocapture`, in `engine/`. Device: Mac native.

```
planned activations: 2838528 bytes (2.8 MB)
after prepare: 277 bind groups, 653 buffers
text_embeddings: largest error 0e0, mean error 0e0, worst at [0, 0, 0] (engine 0.0008400101, golden 0.0008400101), 1 submits, longest 0.2 ms, peak 1162019520 bytes
text_layer_0: largest error 9.5367431640625e-5, mean error 2.1296295446884504e-7, worst at [0, 0, 457] (engine 53.513607, golden 53.513702), 1 submits, longest 10.2 ms, peak 1162019520 bytes
text_encoder: largest error 2.288818359375e-5, mean error 2.481718759764614e-6, worst at [0, 30, 267] (engine 10.422267, golden 10.422244), 23 submits, longest 4.5 ms, peak 1162019520 bytes
run 1 (3 checks): returned output error 2.288818359375e-5, 78848 values; measured peak 1162019520 bytes (1162.0 MB); live now: activations 2838528, weights 1158860800, uniforms 4800, read-back 0; bind groups 277, buffers made in the run 4
run 2 (0 checks): returned output error 2.288818359375e-5, 78848 values; measured peak 1162019520 bytes (1162.0 MB); live now: activations 2838528, weights 1158860800, uniforms 4800, read-back 0; bind groups 277, buffers made in the run 1
text_embeddings: largest error 0e0, mean error 0e0, worst at [0, 0, 0] (engine 0.0008400101, golden 0.0008400101), 1 submits, longest 0.1 ms, peak 1162019520 bytes
text_layer_0: largest error 9.5367431640625e-5, mean error 2.1296295446884504e-7, worst at [0, 0, 457] (engine 53.513607, golden 53.513702), 1 submits, longest 2.5 ms, peak 1162019520 bytes
text_encoder: largest error 2.288818359375e-5, mean error 2.481718759764614e-6, worst at [0, 30, 267] (engine 10.422267, golden 10.422244), 23 submits, longest 2.6 ms, peak 1162019520 bytes
run 3 (3 checks): returned output error 2.288818359375e-5, 78848 values; measured peak 1162019520 bytes (1162.0 MB); live now: activations 2838528, weights 1158860800, uniforms 4800, read-back 0; bind groups 277, buffers made in the run 4
test text_encoder_matches_golden ... ok
```

The first run used a tolerance of 1e9 for all 3 parts, and printed the same errors.

Command: a script in the scratchpad (not in the project). It loads the PyTorch text encoder in float64 and runs it on the golden ids. Device: Mac, Python through `uv run`, CPU.

```
text_layer_0: float32 PyTorch vs float64 PyTorch largest difference 6.827e-05, largest value 53.514
text_encoder: float32 PyTorch vs float64 PyTorch largest difference 2.133e-05, largest value 13.023
```

The engine is 9.54e-5 from PyTorch float32 at layer 0, where PyTorch float32 is 6.83e-5 from float64. At the output: 2.29e-5 and 2.13e-5. Both engine errors are of the size of the float32 rounding of PyTorch, and below 2 times it. Tolerances, 2 x the float64 difference, rounded up: text_layer_0 2e-4 (2 x 6.83e-5 = 1.37e-4), text_encoder 5e-5 (2 x 2.13e-5 = 4.3e-5). text_embeddings 0, because the CPU makes the same one f32 add per value as PyTorch.

Command: `cargo test --release`, in `engine/`. Device: Mac native. Every test file passed, the attention test too. The library tests printed `test result: ok. 38 passed`. `cargo build --release --target wasm32-unknown-unknown` gave no warnings.

Calculations:

- GPU weights: 1,361,551,360 (the whole file) - 202,375,168 (token table) - 315,392 (position table, 77 x 1024 x 4, on the CPU) = 1,158,860,800 bytes, the printed number.
- Peak: 1,158,860,800 + 2,838,528 + 4,800 + 315,392 (the read-back copy of the output, 77 x 1024 x 4) = 1,162,019,520 bytes, the printed peak.
- Bind groups: 23 layers x 12 kernels + final_layer_norm = 277.
- Uniforms: per layer 2 LayerNorms x 16 + 6 linear layers x 16 + attention 32 + 2 adds x 16 + GELU 16 = 208. 23 x 208 + 16 = 4,800 bytes.
- Buffers: 23 layers x 16 weight buffers + 2 (final_layer_norm) = 370 weight buffers, 277 uniforms, and 653 - 370 - 277 = 6 slots.
- One layer: q, k, v and out_proj 4 x 77 x 1024 x 1024 = 322,961,408, fc1 and fc2 2 x 77 x 1024 x 4096 = 645,922,816, so 968,884,224 multiply-adds without attention.
- The stage peak, 1,162.0 MB, is 60.5% of the 1,920 MB tab limit. The JS heap, WASM memory and the browser's own GPU memory are not in it.

Mutation checks. Each bug made the test fail, and the code was restored:

```
no causal mask:
text_layer_0: first mismatch, largest error 5.499276328086853e1 is above 2e-4

no position embedding:
text_embeddings: first mismatch, largest error 4.8490843176841736e-1 is above 0e0

no GELU after fc1:
text_layer_0: first mismatch, largest error 1.1815956221520901e2 is above 2e-4
```

Not measured on the iPhone. `TextStage` has no WASM entry yet, and the browser has no `WeightSource` for files of 1.36 GB.

Conclusion: the text encoder runs on 6 planned buffers and 277 bind groups that `prepare` makes once. Its output matches PyTorch to 2.3e-5.

Next step: the staged run of all three stages, with the scheduler step between the UNet and the decoder.

## 2026-11-18: The three stages, one after the other, with planned and measured peaks

This is the last of the leftover steps in "The whole UNet on its planned slots". It completes the original request: run the text encoder, free it, then the UNet, then the decoder. It also prints the planned peak of each stage before its run and the measured peak after it.

Files added:

- `engine/src/pipeline/mod.rs`: `generate` runs the three stages in order and drops each one before the next one loads.
- `engine/src/pipeline/event.rs`: `Stage` and `StageEvent` (Prepared, Ran, Freed), which `generate` reports.
- `engine/src/pipeline/schedule.rs`: the Euler step of SD-Turbo on the CPU: `scale_input` before the UNet and `euler_step` after it.
- `engine/examples/generate.rs`: runs `generate` on the golden inputs, prints each event and the difference of each output from PyTorch, and writes `engine/target/generate.ppm`.
- `engine/tests/pipeline.rs`: the same run, with assertions.

Files changed:

- `engine/src/text/mod.rs`, `engine/src/unet/mod.rs`: `output_bytes`, the size of the one copy that a run without checks makes.
- `engine/src/gpu.rs`: `reset_peak_bytes` is public.
- `engine/src/lib.rs`: exports the pipeline.
- `engine/tests/common/mod.rs`: `#![allow(dead_code)]`, because the pipeline test uses only part of the shared helpers. Before this, its build printed `warning: function `check_block` is never used` and 2 more warnings.
- `engine/examples/memory_plan.rs`: the text encoder line no longer says "activations not planned". Old: `stage 1 text_encoder: weights 1361551360 bytes (1361.6 MB), activations not planned`. New: `stage 1 text_encoder: weights in the file 1361551360 bytes (1361.6 MB)`.

Design choices:

- Each stage resets the peak counter when it starts, loads its weights, makes its slots and bind groups, runs without checks, and is dropped. Only the outputs go from stage to stage, on the CPU: 315,392 bytes of text, then 65,536 bytes of UNet output.
- The planned peak of a stage is known after `prepare`, before the run: all live bytes, plus the one output copy that a run makes.
- The Euler step runs on the CPU, because it is 16,384 values. It follows the operations of EulerDiscreteScheduler in f32.
- TAESD keeps its own code from "TAESD decoder and its golden set". It reads its weights block by block and copies each block output back for its check, so it has no plan, and its measured peak includes those copies.
- The image test has two rules: 9.8e-4 at most on the [-1, 1] values, which is 1/8 of one 8-bit step (2 / 255 / 8), and at most 1 for each 8-bit byte.

Command: `cargo test --release --lib schedule -- --nocapture`, in `engine/`. Device: Mac native.

```
euler step: largest difference from final_latent 0e0
test pipeline::schedule::tests::step_gives_the_pipeline_latent ... ok
test pipeline::schedule::tests::scaled_noise_is_the_pipeline_unet_input ... ok
```

The first run of these tests failed, because the test read the noise from `golden/float32/`. The file is `golden/starting_noise.safetensors`:

```
starting_noise: cannot read /Users/teimurgasanov/Projects/inference/engine/../golden/float32/starting_noise.safetensors: No such file or directory (os error 2)
```

Command: `cargo run --release --example generate`, in `engine/`. Device: Mac native.

```
text_encoder: prepared; planned activations 2838528 bytes (2.8 MB); planned peak 1162019520 bytes (1162.0 MB); live 1161704128 bytes
text_encoder: ran in 0.51 s; measured peak 1162019520 bytes (1162.0 MB)
text_encoder: freed; live 0 bytes
unet: prepared; planned activations 111141888 bytes (111.1 MB); planned peak 3575166912 bytes (3575.2 MB); live 3575101376 bytes
unet: ran in 2.74 s; measured peak 3575166912 bytes (3575.2 MB)
unet: freed; live 0 bytes
taesd: not planned; live 0 bytes
taesd: ran in 0.37 s; measured peak 201474352 bytes (201.5 MB)
taesd: freed; live 0 bytes
text_encoder: largest difference from PyTorch 2.2888184e-5
unet_output: largest difference from PyTorch 5.00679e-6
final_latent: largest difference from PyTorch 7.343292e-5
taesd_image: largest difference from PyTorch 2.6613474e-4
wrote /Users/teimurgasanov/Projects/inference/engine/target/generate.ppm
```

The image was converted with `sips -s format png` and compared by eye with `golden/float32/taesd_image.png`: the same cat on the same wooden table.

Command: `cargo test --release --test pipeline -- --nocapture`, in `engine/`. Device: Mac native.

```
Prepared { stage: TextEncoder, planned_activations: Some(2838528), planned_peak: Some(1162019520), live: 1161704128 }
Ran { stage: TextEncoder, measured_peak: 1162019520, seconds: 0.285482 }
Freed { stage: TextEncoder, live: 0 }
Prepared { stage: Unet, planned_activations: Some(111141888), planned_peak: Some(3575166912), live: 3575101376 }
Ran { stage: Unet, measured_peak: 3575166912, seconds: 2.7227560410000002 }
Freed { stage: Unet, live: 0 }
Prepared { stage: Decoder, planned_activations: None, planned_peak: None, live: 0 }
Ran { stage: Decoder, measured_peak: 201474352, seconds: 0.371807625 }
Freed { stage: Decoder, live: 0 }
text_encoder: largest difference from PyTorch 2.2888184e-5
unet_output: largest difference from PyTorch 5.00679e-6
final_latent: largest difference from PyTorch 7.343292e-5
taesd_image: largest difference from PyTorch 2.6613474e-4
image bytes: 547 of 1048576 differ, by at most 1
test stages_run_one_after_the_other_and_match_pytorch ... ok
```

Command: `cargo test --release`, in `engine/`. Device: Mac native. All 11 test files passed. The library tests printed `test result: ok. 40 passed`. `cargo build --release --target wasm32-unknown-unknown` gave no warnings.

Calculations:

- Text encoder: planned peak = 1,161,704,128 live + 315,392 output copy (77 x 1024 x 4) = 1,162,019,520 bytes = the measured peak.
- UNet: planned peak = 3,575,101,376 live + 65,536 output copy (4 x 64 x 64 x 4) = 3,575,166,912 bytes = the measured peak. The UNet test had a higher peak (3,585,587,136), because there the checks copied larger tensors back.
- After each stage: 0 live bytes. So the next stage starts with no buffer of the one before it.
- With the stages: the largest stage peak is 3,575.2 MB (the UNet). Without them, all at once: 1,162.0 + 3,575.2 + 201.5 = 4,938.7 MB.
- The latent error is the UNet error times sigma: 5.00679e-6 x 14.6146 = 7.317e-5. The printed 7.343e-5 is that size.
- The UNet output error here, 5.01e-6, is not the 4.05e-6 of tests/unet.rs. Here the UNet reads the engine's text encoder output, which differs from PyTorch by 2.29e-5. tests/unet.rs reads the PyTorch text.
- Image bytes: 547 / 1,048,576 = 0.052% of the bytes differ, each by 1. The alpha bytes are in the count and are all equal.
- Run times on the Mac: text encoder 0.29 to 0.51 s, UNet 2.72 to 2.74 s, TAESD 0.37 s.

Mutation checks. Each bug made the test fail, and the code was restored:

```
the UNet stage is never freed (std::mem::forget in place of drop):
assertion `left == right` failed: unet: live bytes after the stage was freed
  left: 3575101376

the Euler step starts from the scaled noise, not the noise:
final_latent: 1.4388275e-1 is above 5e-4

the planned peak leaves out the output copy:
assertion `left == right` failed: text_encoder: measured peak
  left: 1162019520
```

Not measured on the iPhone. Nothing of the three stages ran in a browser. These parts are still missing before the iPhone can run a generation:

- The slot kernels read f32 weights. With them, the UNet stage needs 3,575.2 MB, which is more than the 1,920 MB tab limit. The plan for the iPhone uses the packed 4-bit and 8-bit weights: 699,997,456 + 111,141,888 = 811,139,344 bytes in "UNet activation plan, from the forward pass". The packed kernels of "Packed 4-bit and 8-bit weights, converter and kernels" do not bind to slots yet. With packed weights, the golden compare needs other tolerances, because the weights differ from PyTorch.
- The text encoder stage needs 1,162.0 MB with f32 weights. That is 60.5% of the tab limit, before the JS heap, WASM memory and the browser's own GPU memory. Its file is 1,361,551,360 bytes to download.
- No WASM entry runs the stages, and the browser has no `WeightSource` for the large files. The streaming loader `GpuWeights` gives GPU buffers, not a `WeightSource`.
- No tokenizer in the engine. The ids come from Python.
- The TAESD stage still copies every block back for its checks.
- The submit times on the iPhone. The longest submit on the Mac is the self attention at 64 x 64, about 70 ms. The attention test measured 145.0 ms for the same work on the iPhone, against 67 ms on the Mac.

Conclusion: the engine generates the golden image in three stages on the Mac. Each stage frees all its buffers before the next one loads, and the planned peak of the text encoder and the UNet equals the measured peak, byte for byte. The output matches PyTorch: 547 of 1,048,576 image bytes differ, each by 1.

Next step: bind the packed 4-bit and 8-bit kernels to the slots, so that the UNet stage fits the iPhone budget. The user decides between this and a WASM entry for the stages.

## 2026-11-18: Packed 4-bit and 8-bit UNet weights on the planned slots

The user asked for two steps, in order: the packed weights on the slots, then a WASM entry for the stages. This entry is the first one. It continues "The three stages, one after the other, with planned and measured peaks".

Files added:

- `dump_packed_golden.py`: the golden set of the packed UNet, in `golden/packed/`. It loads the PyTorch UNet, puts in the weights of `weights/unet_packed.bin` (packed weight = integer x float16 scale, with `unpack_rows` of `pack_weights.py`; every other tensor = its float16 value), and runs it on the float32 golden inputs, in float32 and in float64. It also makes the Euler step (the f32 operations of `engine/src/pipeline/schedule.rs`) and the TAESD image, and saves `golden/packed/taesd_image.png`.
- `engine/src/weights/buffers.rs`: the trait `WeightBuffers`, which gives the slot kernels their weights as GPU buffers. `Uploading` reads a CPU reader and copies each tensor to a new buffer. `GpuWeights` lends its own buffers, gives packed matrices as words and scales, and widens float16 tensors to f32 once.
- `engine/src/weights/widen.wgsl`: float16 to f32.
- `engine/src/layers/packed/vector.wgsl`: the packed one-row linear layer, with the SiLU flag of `vector.wgsl`.
- `engine/src/pipeline/unet_weights.rs`: the trait `UnetWeights`, with `FloatUnet` (a CPU reader of float32) and `PackedUnetFile` (streams a packed file, native only).
- `engine/tests/unet_packed.rs`: the packed UNet against `golden/packed`.

Files changed:

- `engine/src/layers/packed/linear.wgsl`: the row check of the float linear kernel, for the 77 text rows of the cross attention.
- `engine/src/layers/packed/mod.rs`: `bind_packed_into` and `bind_packed_vector_into` bind the packed kernels to slots.
- `engine/src/layers/mod.rs`: `bind_matrix_into` and `bind_vector_matrix_into` pick the float or the packed kernel from the weight.
- `engine/src/layers/vector.rs`: `plan_vector` is visible to `packed/`.
- `engine/src/conv.rs`, `engine/src/linear.rs`, `engine/src/kernels/weight.rs`, `group_norm.rs`, `layer_norm.rs`: the slot binders take `WeightBuffers`. `slot_bias` gives a bias from the weights, or a buffer of zeros.
- `engine/src/unet/*.rs`, `engine/src/text/bind.rs`: the binders take `WeightBuffers`.
- `engine/src/unet/mod.rs`: `UnetStage::prepare_loaded` binds to a `GpuWeights` and keeps it, so dropping the stage frees the weights.
- `engine/src/text/mod.rs`: `prepare` wraps its reader in `Uploading`.
- `engine/src/weights/file.rs`: `load_to_gpu` streams a file into `GpuWeights` with the loader of the browser, in chunks of 2 MiB.
- `engine/src/pipeline/mod.rs`: `StageWeights::unet` is a `UnetWeights`. The UNet stage loads its weights after the text encoder is freed.
- `engine/tests/common/mod.rs`: `golden_in` and `check_block_in` read another golden set. The hint of a missing file names all four dump scripts.
- `engine/tests/pipeline.rs`: two tests, the float32 UNet and the packed UNet.
- `engine/examples/generate.rs`: `-- --packed`.

Design choices:

- Every weight matrix of the UNet is packed in the file: 54 int4 and 228 int8, with the time layers, conv_in and conv_out. The other 686 tensors are float16: 282 row scales and 404 small tensors (biases and norm scales), 1,524,496 bytes in all.
- The UNet stage binds to the GPU buffers of the streaming loader. A browser cannot hold the 700,021,120-byte file in WASM memory, and the loader puts each tensor into its own buffer as it streams. The Mac uses the same loader.
- The packed kernels read the packed words and the float16 row scales as the file stores them. The small float16 tensors are widened to f32 once, in `prepare`, by one submit of `widen.wgsl` each. So the float32 kernels and their tests did not change.
- The packed golden set separates two errors. The packed weights change the UNet output by 0.126 (printed below), which the user judged with the 20-image grid in "Mixed plan grid, checked by the user". This test checks only the kernels: PyTorch runs with exactly the weights that the engine reads.
- The stage owns the loaded weights. The bind groups point at the buffers of the weights, and in the browser a dropped buffer is destroyed. `_weights` is the last field, so Rust drops it after the steps.

Command: `uv run dump_packed_golden.py`, in the project folder. Device: Mac, Python through `uv run`, CPU. The second run, with the Euler step and TAESD, printed the same 13 numbers as the first run, and these lines:

```
time_embedding: float32 vs float64 largest difference 4.620e-07, largest value 6.117
conv_in: float32 vs float64 largest difference 8.515e-07, largest value 3.299
down_block_0: float32 vs float64 largest difference 1.928e-05, largest value 13.177
down_block_1: float32 vs float64 largest difference 1.250e-04, largest value 44.317
down_block_2: float32 vs float64 largest difference 1.191e-04, largest value 48.708
down_block_3: float32 vs float64 largest difference 1.272e-04, largest value 39.833
mid_block: float32 vs float64 largest difference 1.278e-04, largest value 39.933
up_block_0: float32 vs float64 largest difference 9.531e-05, largest value 40.515
up_block_1: float32 vs float64 largest difference 4.145e-04, largest value 102.728
up_block_2: float32 vs float64 largest difference 4.217e-04, largest value 87.822
up_block_3: float32 vs float64 largest difference 2.073e-04, largest value 31.855
unet_output: float32 vs float64 largest difference 8.720e-06, largest value 4.201
unet_output: packed vs float32 weights largest difference 1.259e-01
golden/packed:
  time_embedding   shape=(1, 1280) dtype=float32
  conv_in          shape=(1, 320, 64, 64) dtype=float32
  down_block_0     shape=(1, 320, 32, 32) dtype=float32
  down_block_1     shape=(1, 640, 16, 16) dtype=float32
  down_block_2     shape=(1, 1280, 8, 8) dtype=float32
  down_block_3     shape=(1, 1280, 8, 8) dtype=float32
  mid_block        shape=(1, 1280, 8, 8) dtype=float32
  up_block_0       shape=(1, 1280, 16, 16) dtype=float32
  up_block_1       shape=(1, 1280, 32, 32) dtype=float32
  up_block_2       shape=(1, 640, 64, 64) dtype=float32
  up_block_3       shape=(1, 320, 64, 64) dtype=float32
  unet_output      shape=(1, 4, 64, 64) dtype=float32
  final_latent     shape=(1, 4, 64, 64) dtype=float32
  taesd_image      shape=(1, 3, 512, 512) dtype=float32
```

The first run took `41.268 total` by `time`.

Command: `cargo test --release`, in `engine/`, after the change to `WeightBuffers` and before the packed test existed. Device: Mac native. All test files passed, and the float32 tests printed the numbers of the entries before, for example `unet_output: largest error 4.0531158447265625e-6` and `text_encoder: largest error 2.288818359375e-5`, with `after prepare: 742 bind groups, 1546 buffers` and `after prepare: 277 bind groups, 653 buffers`.

Command: `cargo test --release --test unet_packed -- --nocapture`, in `engine/`. Device: Mac native. A run that passed, with the tolerances still at 1e9:

```
loaded weights: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes
planned activations: 111141888 bytes (111.1 MB)
after prepare: 1146 bind groups, 2232 buffers, weights 702072352 bytes (2074896 added by prepare)
time_embedding: largest error 9.5367431640625e-7, mean error 3.107919965827932e-8, worst at [0, 700] (engine -4.732074, golden -4.732073), 1 submits, longest 2.4 ms, peak 813240944 bytes
conv_in: largest error 9.5367431640625e-7, mean error 3.506597009198496e-8, worst at [0, 232, 42, 57] (engine 2.3722181, golden 2.372219), 1 submits, longest 3.5 ms, peak 818478704 bytes
down_block_0: largest error 2.4318695068359375e-5, mean error 1.6560652267901682e-6, worst at [0, 274, 25, 17] (engine 7.7573805, golden 7.757405), 19 submits, longest 74.5 ms, peak 818478704 bytes
down_block_1: largest error 1.7261505126953125e-4, mean error 4.043955155996759e-6, worst at [0, 406, 0, 15] (engine 14.514702, golden 14.514529), 19 submits, longest 13.8 ms, peak 818478704 bytes
down_block_2: largest error 1.773834228515625e-4, mean error 1.3549372545051952e-5, worst at [0, 715, 3, 4] (engine 28.158918, golden 28.159096), 19 submits, longest 27.5 ms, peak 818478704 bytes
down_block_3: largest error 1.6117095947265625e-4, mean error 1.4314504983303778e-5, worst at [0, 124, 6, 5] (engine -13.813651, golden -13.813812), 4 submits, longest 31.4 ms, peak 818478704 bytes
mid_block: largest error 1.583099365234375e-4, mean error 1.4609752872729586e-5, worst at [0, 124, 6, 5] (engine -13.629911, golden -13.63007), 11 submits, longest 19.5 ms, peak 818478704 bytes
up_block_0: largest error 1.125335693359375e-4, mean error 3.5235607396821855e-6, worst at [0, 972, 5, 1] (engine 29.399057, golden 29.39917), 10 submits, longest 30.7 ms, peak 818478704 bytes
up_block_1: largest error 5.5694580078125e-4, mean error 3.9377276502064117e-5, worst at [0, 432, 17, 17] (engine -69.49545, golden -69.49601), 31 submits, longest 57.6 ms, peak 818478704 bytes
up_block_2: largest error 5.106925964355469e-4, mean error 2.1668187247846775e-5, worst at [0, 127, 44, 34] (engine 6.8343287, golden 6.833818), 31 submits, longest 119.3 ms, peak 823721584 bytes
up_block_3: largest error 2.751350402832031e-4, mean error 5.168745689942966e-6, worst at [0, 274, 55, 12] (engine 2.0752685, golden 2.0749934), 30 submits, longest 107.2 ms, peak 823721584 bytes
unet_output: largest error 9.775161743164063e-6, mean error 8.554381121239629e-7, worst at [0, 2, 15, 12] (engine 3.3576424, golden 3.3576326), 1 submits, longest 15.1 ms, peak 823721584 bytes
run 1 (12 checks): returned output error 9.775161743164063e-6; measured peak 823721584 bytes (823.7 MB); live now: activations 111141888, weights 702072352, uniforms 21584, read-back 0; bind groups 1146, buffers made in the run 13
run 2 (0 checks): returned output error 9.775161743164063e-6; measured peak 823721584 bytes (823.7 MB); live now: activations 111141888, weights 702072352, uniforms 21584, read-back 0; bind groups 1146, buffers made in the run 1
run 3 (12 checks): returned output error 9.775161743164063e-6; measured peak 823721584 bytes (823.7 MB); live now: activations 111141888, weights 702072352, uniforms 21584, read-back 0; bind groups 1146, buffers made in the run 13
after drop: live 0 bytes
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 15.17s
```

Run 3 printed the same errors as run 1.

Failed runs: two earlier runs of the same test failed on the 0.25 s limit. Run 1 of the first one passed every check up to up_block_2 with the errors above:

```
up_blocks.3.attentions.2.transformer_blocks.0.attn1 attention: one submit took 848.7 ms, the limit is 250.0 ms
down_blocks.2.resnets.0 part 1: one submit took 1333.2 ms, the limit is 250.0 ms
```

The same submits took 71.3 ms and 13.8 ms in other runs. `ps -Ao pcpu,comm -r` showed `duetexpertd` 46.0%, the iOS Simulator app `Kibi` 14.3%, OrbStack 9.5% and the Simulator's `backboardd` 7.3%. Right after, `cargo test --release --test unet` passed once and `cargo test --release --test unet_packed` passed twice. The times of the run above also vary (up to 148.7 ms in run 3). So these are slow submits of a busy Mac, as in "First transformer, down_blocks.0.attentions.0, on its planned slots".

Tolerances: time_embedding, conv_in and down_block_0 1e-4, as in tests/unet.rs. The others are 2 x the float64 difference of the packed model, rounded up: down_block_1, down_block_2, down_block_3, mid_block and up_block_0 3e-4 (2 x 1.278e-4 = 2.6e-4 at most). up_block_1 and up_block_2 1e-3 (2 x 4.217e-4 = 8.4e-4). up_block_3 5e-4 (2 x 2.073e-4 = 4.1e-4). unet_output 2e-5 (2 x 8.72e-6 = 1.7e-5). The packed errors are 1.34 times the float64 difference at up_block_1 (5.57e-4 / 4.145e-4), where the float32 UNet was at 0.66 times. The packed kernels add the integer products first and multiply by the scale once, and PyTorch multiplies each weight by its scale first. So the rounding order differs more than in the float32 kernels.

Command: `cargo run --release --example generate -- --packed`, in `engine/`. Device: Mac native.

```
text_encoder: prepared; planned activations 2838528 bytes (2.8 MB); planned peak 1162019520 bytes (1162.0 MB); live 1161704128 bytes
text_encoder: ran in 0.31 s; measured peak 1162019520 bytes (1162.0 MB)
text_encoder: freed; live 0 bytes
unet: prepared; planned activations 111141888 bytes (111.1 MB); planned peak 813301360 bytes (813.3 MB); live 813235824 bytes
unet: ran in 1.79 s; measured peak 813301360 bytes (813.3 MB)
unet: freed; live 0 bytes
taesd: not planned; live 0 bytes
taesd: ran in 0.40 s; measured peak 201474352 bytes (201.5 MB)
taesd: freed; live 0 bytes
text_encoder: largest difference from PyTorch (float32) 2.2888184e-5
unet_output: largest difference from PyTorch (packed) 9.655952e-6
final_latent: largest difference from PyTorch (packed) 1.411438e-4
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
wrote /Users/teimurgasanov/Projects/inference/engine/target/generate_packed.ppm
```

The image was converted with `sips -s format png` and looked at: the same cat on the same table as `generate.ppm`, with small changes in the fur and the wood from the packed weights.

Command: `cargo test --release --test pipeline -- --nocapture --test-threads 1`, in `engine/`. Device: Mac native.

```
test stages_with_float_unet_match_pytorch ... Prepared { stage: TextEncoder, planned_activations: Some(2838528), planned_peak: Some(1162019520), live: 1161704128 }
Prepared { stage: Unet, planned_activations: Some(111141888), planned_peak: Some(3575166912), live: 3575101376 }
Ran { stage: Unet, measured_peak: 3575166912, seconds: 2.958095209 }
text_encoder: largest difference from PyTorch (float32) 2.2888184e-5
unet_output: largest difference from PyTorch (float32) 5.00679e-6
final_latent: largest difference from PyTorch (float32) 7.343292e-5
taesd_image: largest difference from PyTorch (float32) 2.6613474e-4
image bytes: 547 of 1048576 differ, by at most 1
test stages_with_packed_unet_match_pytorch ... Prepared { stage: TextEncoder, planned_activations: Some(2838528), planned_peak: Some(1162019520), live: 1161704128 }
Prepared { stage: Unet, planned_activations: Some(111141888), planned_peak: Some(813301360), live: 813235824 }
Ran { stage: Unet, measured_peak: 813301360, seconds: 1.8083580000000001 }
text_encoder: largest difference from PyTorch (float32) 2.2888184e-5
unet_output: largest difference from PyTorch (packed) 9.655952e-6
final_latent: largest difference from PyTorch (packed) 1.411438e-4
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
image bytes: 1140 of 1048576 differ, by at most 1
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.31s
```

Command: `cargo test --release`, in `engine/`. Device: Mac native. All 12 test files passed. The library tests printed `test result: ok. 40 passed`. `cargo build --release --target wasm32-unknown-unknown` gave no warnings.

Calculations:

- Weights added by prepare: the float16 tensors that are not scales have 443,844 values (a Python sum over `unet_packed.json`), so their f32 copies are 443,844 x 4 = 1,775,376 bytes. With the zero biases, 299,520 bytes: 1,775,376 + 299,520 = 2,074,896, the printed number.
- Loaded weights: 699,997,456 bytes, the number of "Streaming weight loader, OPFS cache and GPU memory per category".
- UNet stage peak in a real run: 813,235,824 live + 65,536 output copy = 813,301,360 bytes = the measured peak. 813,301,360 / 1,920,000,000 = 42.4% of the tab limit. The plan of "UNet activation plan, from the forward pass" was 811,139,344 bytes. The difference, 2,162,016 bytes, is the 2,074,896 bytes of prepare, the 21,584 bytes of uniforms and the 65,536-byte output copy.
- The largest stage peak with the packed UNet is now the text encoder, 1,162.0 MB = 60.5% of the tab limit.
- In the UNet test, the peak 823,721,584 = 111,141,888 + 702,072,352 + 21,584 + 10,485,760 (the read-back copy of up_block_2).
- Bind groups: 1,146 in prepare, 404 more than the float32 UNet (742). They are the 404 widen dispatches, one per small float16 tensor.
- The packed latent error: 1.411e-4 = 9.66e-6 x 14.61, the UNet error times sigma.
- Run times on the Mac: the packed UNet stage 1.79 to 1.81 s, the float32 UNet stage 2.72 to 2.96 s.

Mutation checks. Each bug made the packed test fail, and the code was restored (`cmp` against the saved copies):

```
the packed one-row kernel reads the scale of the neighbor column:
time_embedding: first mismatch, largest error 6.03471902012825e0 is above 1e-4

widen reads only the low half of each word:
time_embedding: first mismatch, largest error 6.57806396484375e-2 is above 1e-4

the packed linear kernel reads row 0 of the block for row 1:
down_block_0: first mismatch, largest error 4.398473739624023e0 is above 1e-4

4-bit weights read as 8-bit (prepare stops):
packed weight has 204800 values, the shape needs 409600
```

Failed attempts during the work: a Python script that changed `linear.rs`, the norm reader and the binders stopped with `ValueError: substring not found`, because its first cut in `linear.rs` had already removed the text that its second cut looked for. Each file is written only at the end of its part, so no file was half changed. The script ran again without the second cut.

Not measured on the iPhone.

Conclusion: the UNet stage runs on the packed weights of `unet_packed.bin`, streamed into GPU buffers with the loader of the browser. It matches PyTorch with the same weights to 9.8e-6, and a real run needs 813.3 MB, 42.4% of the tab limit.

Next step: the WASM entry for the stages.

## 2026-11-18: WASM entry for the three stages, and the generate page

This is the second of the two steps that the user asked for in "Packed 4-bit and 8-bit UNet weights on the planned slots". A web page now runs a whole generation in the browser: it loads each weight file just before its stage, and each stage frees its file.

Files added:

- `engine/web/generate.html`: reads the golden inputs (token ids, noise, sigma, timestep), loads `text_encoder` with `weights.js` (OPFS cache), runs stage 1, loads `unet_packed`, runs stage 2, fetches `taesd`, runs stage 3, draws the image, and compares its bytes with `golden/packed/taesd_image`.
- `engine/src/pipeline/sources.rs`: replaces `unet_weights.rs`. The traits `TextWeights` and `UnetWeights` take their value, so a `GpuWeights` can move into its stage. `FromReader` (CPU reader), `Loaded` (a file on the GPU already, for the browser), and `StreamedFile` (streams a file on disk, native only).

Files changed:

- `engine/src/web/engine.rs`: the state is in an `Rc<RefCell<State>>`, so the async methods can own a handle. New methods `runTextEncoder(name, ids)`, `runUnet(name, noise, sigma, timestep)` and `runDecoder(indexText, bytes)`. Each one takes its loaded file out of the list, runs its stage, and gives back the report lines. The engine keeps the text output and the latent between the calls.
- `engine/src/pipeline/mod.rs`: `text_stage`, `unet_stage`, `decoder_stage`, `scale_input` and `euler_step` are public. `generate` calls them and takes `StageWeights` by value.
- `engine/src/pipeline/event.rs`: `Display` for `StageEvent`. The example and the page print the same lines.
- `engine/src/text/embed.rs`, `engine/src/text/mod.rs`, `engine/src/text/bind.rs`: the token rows are copied on the GPU (see below). `TextStage::prepare_loaded` binds to a loaded file. `run` no longer takes a reader.
- `engine/src/taesd/mod.rs`, `engine/src/taesd/report.rs`, `engine/src/lib.rs`: `run_taesd` takes `wanted`, and `TAESD_PARTS` lists its 9 parts.
- `engine/tests/text.rs`, `engine/tests/taesd.rs`, `engine/tests/pipeline.rs`, `engine/examples/generate.rs`: the new calls. The packed pipeline test and `generate -- --packed` stream both planned files, as the page does.
- `engine/web/pkg/`: rebuilt.

Design choices:

- JavaScript runs the stages one by one, because only JavaScript can stream a file with fetch and OPFS. So the page streams the file of a stage, then calls the stage, and the stage takes the file. Before, `generate` read files only on the Mac.
- The text encoder file (1,361,551,360 bytes) cannot be in WASM memory, so a run cannot read 77 rows from a CPU reader in the browser. The text stage now keeps the token table on the GPU and copies the 77 rows with `copy_buffer_to_buffer` into the embeddings slot. A copy needs no bind group, and it is not limited by the 128 MiB binding limit. The position table is added by a kernel in the first step, on the Mac and in the browser. The Mac path uses the GPU copy too, so the text stage has one path. Its peak grew by the token table and the position table (see below).
- The page reads TAESD whole into WASM memory (9,780,736 bytes), as `taesd.html` does.
- `run_taesd` copied every part back for its checks, also in a generation. The first page run measured 191,102,976 bytes of WASM memory. WASM memory never shrinks, so this stays in the tab budget for the life of the page. Now a run copies back only the wanted parts. A generation wants none, so the only copy is the image.

Command: `wasm-pack build --release --target web --out-dir web/pkg --no-typescript`, in `engine/`. It printed `Done in 1.20s`, and later `Done in 1.02s` after the TAESD change. `cargo build --release --target wasm32-unknown-unknown` gave no warnings.

Command: `cargo test --release --test text -- --nocapture`, in `engine/`, after the GPU row copy. Device: Mac native.

```
planned activations: 2838528 bytes (2.8 MB)
after prepare: 278 bind groups, 656 buffers
text_embeddings: largest error 0e0, mean error 0e0, worst at [0, 0, 0] (engine 0.0008400101, golden 0.0008400101), 1 submits, longest 3.9 ms, peak 1364710096 bytes
text_layer_0: largest error 9.5367431640625e-5, mean error 2.1296295446884504e-7, worst at [0, 0, 457] (engine 53.513607, golden 53.513702), 1 submits, longest 12.9 ms, peak 1364710096 bytes
text_encoder: largest error 2.288818359375e-5, mean error 2.481718759764614e-6, worst at [0, 30, 267] (engine 10.422267, golden 10.422244), 23 submits, longest 6.6 ms, peak 1364710096 bytes
run 1 (3 checks): returned output error 2.288818359375e-5, 78848 values; measured peak 1364710096 bytes (1364.7 MB); live now: activations 2838528, weights 1361551360, uniforms 4816, read-back 0; bind groups 278, buffers made in the run 4
run 2 (0 checks): returned output error 2.288818359375e-5, 78848 values; measured peak 1364710096 bytes (1364.7 MB); live now: activations 2838528, weights 1361551360, uniforms 4816, read-back 0; bind groups 278, buffers made in the run 1
```

Run 3 printed the same errors. The embeddings are still bit for bit the same as PyTorch.

Page: `http://127.0.0.1:8765/engine/web/generate.html`, served by `python3 -m http.server 8765 --bind 127.0.0.1` in the project folder, opened with Playwright. `localhost` is a secure context, so WebGPU and OPFS work. Device: Mac Chromium. The console had one error, `Failed to load resource: the server responded with a status of 404 (File not found) @ http://127.0.0.1:8765/favicon.ico:0`, which is the missing page icon.

First click on Generate (files from the network, before the TAESD change):

```
ready
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from network in 4.51 s
text_encoder: prepared; planned activations 2838528 bytes (2.8 MB); planned peak 1364710096 bytes (1364.7 MB); live 1364394704 bytes
text_encoder: ran in 1.18 s; measured peak 1364710096 bytes (1364.7 MB)
text_encoder: freed; live 0 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from network in 2.40 s
unet: prepared; planned activations 111141888 bytes (111.1 MB); planned peak 813301360 bytes (813.3 MB); live 813235824 bytes
unet: ran in 2.84 s; measured peak 813301360 bytes (813.3 MB)
unet: freed; live 0 bytes
taesd: not planned; live 0 bytes
taesd: ran in 0.54 s; measured peak 201474352 bytes (201.5 MB)
taesd: freed; live 0 bytes
total: 11.57 s
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
```

A screenshot of the canvas (`.playwright-mcp/chromium_generate.png`) shows the same cat as `golden/packed/taesd_image.png`.

Second click on the same page (files from the cache):

```
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from cache in 1.63 s
...
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from cache in 0.69 s
...
total: 5.81 s
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
```

The lines in "..." were the same as in the first click, except the times (`ran in 0.22 s` for the text encoder, `2.71 s` for the UNet, `0.49 s` for TAESD). After it, `wasm.memory.buffer.byteLength` printed `191102976`.

After the TAESD change, the page was opened again (a new page, so a new WASM memory) and Generate was clicked once (files from the cache):

```
WASM memory before 1310720, after 20185088
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from cache in 1.32 s
text_encoder: prepared; planned activations 2838528 bytes (2.8 MB); planned peak 1364710096 bytes (1364.7 MB); live 1364394704 bytes
text_encoder: ran in 0.24 s; measured peak 1364710096 bytes (1364.7 MB)
text_encoder: freed; live 0 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from cache in 0.73 s
unet: prepared; planned activations 111141888 bytes (111.1 MB); planned peak 813301360 bytes (813.3 MB); live 813235824 bytes
unet: ran in 2.71 s; measured peak 813301360 bytes (813.3 MB)
unet: freed; live 0 bytes
taesd: not planned; live 0 bytes
taesd: ran in 0.42 s; measured peak 201474352 bytes (201.5 MB)
taesd: freed; live 0 bytes
total: 5.49 s
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1

--- memory after:
weights: 0 bytes (0.0 MB)
activations: 0 bytes (0.0 MB)
uniforms: 0 bytes (0.0 MB)
read-back: 0 bytes (0.0 MB)
total: 0 bytes (0.0 MB)
peak: 201474352 bytes (201.5 MB)
```

Command: `cargo test --release`, in `engine/`, after the TAESD change. Device: Mac native. All 12 test files passed. The library tests printed `test result: ok. 40 passed`. The TAESD test printed `taesd_image: largest error 7.867813110351563e-6`, the same as in "TAESD decoder and its golden set". `cargo test --release --test pipeline -- --nocapture --test-threads 1` printed `image bytes: 547 of 1048576 differ, by at most 1` (float32) and `image bytes: 1140 of 1048576 differ, by at most 1` (packed), the same as before.

Calculations:

- Text stage in the browser: planned peak = 1,364,394,704 live + 315,392 output copy = 1,364,710,096 bytes = the measured peak. The live bytes are the whole file (1,361,551,360) + slots (2,838,528) + uniforms (4,816).
- The text stage grew by 1,364,710,096 - 1,162,019,520 = 202,690,576 bytes = the token table (202,375,168) + the position table (315,392) + the uniform of the position add (16). 1,364,710,096 / 1,920,000,000 = 71.1% of the tab limit.
- UNet stage in the browser: planned peak 813,301,360 bytes = measured peak, the same number as on the Mac. 42.4% of the tab limit.
- Image bytes: Chromium 1,199 differing bytes, the Mac native run 1,140. Both differ by at most 1. Chromium runs the WGSL through another compiler (Dawn), so the float rounding can differ a little.
- WASM memory: the 191,102,976 bytes were measured after two generations with the old code. The old code copied back all 9 TAESD parts. The two largest are 64 x 512 x 512 x 4 = 67,108,864 bytes each, and each copy is held twice for a moment: the bytes, then the f32 values. With the new code, one generation from a fresh page grew WASM memory from 1,310,720 to 20,185,088 bytes. The parts in WASM memory are now TAESD (9,780,736), the image as bytes and as f32 values (2 x 3,145,728), and the RGBA bytes (1,048,576). The sum is 17,120,768 bytes, and the rest is the allocator.
- The TAESD GPU peak did not change (201,474,352). Its peak is the activations at 512 x 512: 3 tensors of 67,108,864 bytes = 201,326,592, plus 147,760 bytes of weights and uniforms.
- Load from the network over localhost: 1,361,551,360 / 4.51 s = 302 MB/s, and 699,997,456 / 2.40 s = 292 MB/s. From the cache: 1,361,551,360 / 1.32 s = 1,031 MB/s.

Mutation checks: none new in this step. The stage code is the code of the native tests, which have their mutation checks. The page check is the byte compare with the packed golden image.

Not measured on the iPhone. Not known for the iPhone: whether Safari keeps the tab alive with 1,364.7 MB of engine buffers in the text stage (71.1% of the tab limit before the JS heap and the browser's own memory), the submit times, the load time of 2,061,548,816 bytes over Wi-Fi (the network load of "Weight loader on the iPhone, first load over the local network" ran at 7.35 MB/s), and whether OPFS can hold 2,061,548,816 bytes.

Conclusion: the browser runs a whole generation with the packed UNet in three stages on the Mac. Each stage frees its file, and the planned and measured peaks are equal. The image matches PyTorch with the same weights: 1,199 of 1,048,576 bytes differ, each by 1.

Next step: the user opens `https://10.120.1.125:8443/engine/web/generate.html` on the iPhone, taps Generate, and sends the log. If the tab dies in the text stage, the next step is to keep the token table off the GPU, or to pack the text encoder.

## 2026-11-18: Twenty images in a row

Files changed:

- `engine/src/gpu.rs`: the field `longest_submit`, and the functions `record_submit` and `take_longest_submit`.
- `engine/src/submit.rs`: `submit_within_limit` gives each submit time to `record_submit`.
- `engine/src/web/engine.rs`: two new calls for the page, `liveBytes()` and `takeLongestSubmit()`.
- `engine/web/generate.html`: the button "Make 20". `generate` takes a report function and returns `{ seconds, differing, largest }`. `runJob` holds the button and error code that both buttons share.

Design choices:

- The submits did not change. Each UNet, text encoder, and TAESD step is one submit already. `submit_within_limit` waits for onSubmittedWorkDone after each submit and stops the run if a submit takes 250 ms or more. So a group of submits is one step, and the wait comes between each pair of groups.
- The engine keeps the longest submit time, in the same way as it keeps the peak bytes. The page reads it and sets it to 0 after each image. So each line proves that every submit of that image stayed under the limit. A field on `Gpu` was chosen over a new field in each stage event, because all three stages call `submit_within_limit`.
- `liveBytes()` gives a Number, not a BigInt. A Number holds whole numbers up to 2^53 exactly.
- Each image loads both weight files again, from the cache, and each stage frees its file. This is the same path as one Generate click. So the row also shows whether a buffer leaks from one image to the next.
- The row prints one line per image and no stage lines, so that the log stays short on the phone. The line also gives the WASM memory size. WASM memory never shrinks, so growth in it from image to image means a leak in WASM memory.
- An error stops the row. The error line names the submit, and `freeWeights` frees what was loaded.

Format of one line of the row:

```
image <n> of 20: <seconds> s; longest submit <ms> ms; live <bytes> bytes; WASM memory <bytes> bytes; <count> image bytes differ, by at most <largest>
```

Commands: `wasm-pack build --release --target web --out-dir web/pkg --no-typescript`, in `engine/`. `cargo test --release --lib` printed `test result: ok. 40 passed`. `cargo test --release --test pipeline -- --test-threads 1` printed `test result: ok. 2 passed`. `cargo clippy --release --all-targets` gave no warning in the changed files.

Page: `http://127.0.0.1:8765/engine/web/generate.html`, served by `python3 -m http.server 8765 --bind 127.0.0.1` in the project folder. Device: Mac Chromium. The Playwright MCP was not available, so a Node script with `playwright@1.56.1` (in the scratchpad, not in the project) opened Chrome for Testing from `~/Library/Caches/ms-playwright/chromium-1254`, with `--enable-unsafe-webgpu`, and clicked the button. The browser profile was new and empty, so image 1 loaded both files from the network and the other images from the cache. The console had one error, `Failed to load resource: the server responded with a status of 404 (File not found)`, which is the missing page icon.

Click on "Make 20":

```
ready
image 1 of 20: 11.80 s; longest submit 135.3 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 2 of 20: 5.23 s; longest submit 140.7 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 3 of 20: 4.36 s; longest submit 135.6 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 4 of 20: 4.66 s; longest submit 138.2 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 5 of 20: 4.32 s; longest submit 131.7 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 6 of 20: 4.63 s; longest submit 133.7 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 7 of 20: 4.71 s; longest submit 131.1 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 8 of 20: 4.87 s; longest submit 134.7 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 9 of 20: 5.30 s; longest submit 143.0 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 10 of 20: 4.51 s; longest submit 130.2 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 11 of 20: 5.25 s; longest submit 130.3 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 12 of 20: 4.92 s; longest submit 138.7 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 13 of 20: 4.79 s; longest submit 129.4 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 14 of 20: 4.61 s; longest submit 130.4 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 15 of 20: 4.69 s; longest submit 136.0 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 16 of 20: 5.45 s; longest submit 141.9 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 17 of 20: 4.78 s; longest submit 140.1 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 18 of 20: 5.17 s; longest submit 129.7 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 19 of 20: 4.65 s; longest submit 135.3 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
image 20 of 20: 4.62 s; longest submit 135.3 ms; live 0 bytes; WASM memory 19922944 bytes; 1199 image bytes differ, by at most 1
20 images in 103.50 s

--- memory after:
weights: 0 bytes (0.0 MB)
activations: 0 bytes (0.0 MB)
uniforms: 0 bytes (0.0 MB)
read-back: 0 bytes (0.0 MB)
total: 0 bytes (0.0 MB)
peak: 201474352 bytes (201.5 MB)
```

Calculations:

- Longest submit of the row: 143.0 ms (image 9). The shortest of the 20 longest submits: 129.4 ms (image 13). 143.0 / 250 = 57.2% of the project limit.
- Images 2 to 20, from the cache: sum 91.52 s, mean 91.52 / 19 = 4.817 s, shortest 4.32 s, longest 5.45 s.
- Sum of the 20 image times: 103.32 s. The row took 103.50 s. The difference, 103.50 - 103.32 = 0.18 s, is the page code between the images.
- Live GPU bytes after each image: 0. WASM memory: 19,922,944 bytes after each image, so it did not grow from image 1 to image 20.
- The image bytes were the same in all 20 images: 1,199 of 1,048,576 differ from the packed PyTorch image, by at most 1, the same as in "Staged generation in the browser".

Single Generate after the change, to show that the old button still works. The first try used the same script, but in a private browser context (`browser.newPage()` of `chromium.launch`). It failed:

```
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from network in 5.46 s
text_encoder: prepared; planned activations 2838528 bytes (2.8 MB); planned peak 1364710096 bytes (1364.7 MB); live 1364394704 bytes
text_encoder: ran in 0.52 s; measured peak 1364710096 bytes (1364.7 MB)
text_encoder: freed; live 0 bytes
FAILED: The operation failed because it would cause the application to exceed its storage quota.
```

The OPFS cache write of `unet_packed.bin` failed. Chromium gives a private context a small storage quota in memory. The Mac had 23 GiB of free disk. This is a limit of the test browser, not of the engine. The batch run above was also in a private context. Its cache writes passed, but the quota of a private context is not fixed, so that result does not prove anything about the quota. The script then used `chromium.launchPersistentContext` with a profile folder in the scratchpad. Two runs of Generate followed: the first from the network, the second from the cache.

```
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from network in 4.32 s
...
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from network in 5.15 s
...
total: 12.82 s
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
```

```
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from cache in 1.08 s
text_encoder: prepared; planned activations 2838528 bytes (2.8 MB); planned peak 1364710096 bytes (1364.7 MB); live 1364394704 bytes
text_encoder: ran in 0.22 s; measured peak 1364710096 bytes (1364.7 MB)
text_encoder: freed; live 0 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from cache in 0.55 s
unet: prepared; planned activations 111141888 bytes (111.1 MB); planned peak 813301360 bytes (813.3 MB); live 813235824 bytes
unet: ran in 2.69 s; measured peak 813301360 bytes (813.3 MB)
unet: freed; live 0 bytes
taesd: not planned; live 0 bytes
taesd: ran in 0.37 s; measured peak 201474352 bytes (201.5 MB)
taesd: freed; live 0 bytes
total: 4.98 s
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
```

The lines in "..." of the first run were the same as in the second run, except the times (`ran in 0.22 s` for the text encoder, `2.64 s` for the UNet, `0.40 s` for TAESD). The planned and measured peaks are the same as in "Staged generation in the browser".

Mutation check: `record_submit` was changed to `let _ = seconds;`, so it kept no time. The WASM build was made again, and "Make 20" was clicked in the persistent profile. The first line was:

```
image 1 of 20: 5.42 s; longest submit 0.0 ms; live 0 bytes; WASM memory 19595264 bytes; 1199 image bytes differ, by at most 1
```

So the number in the line comes from `record_submit`, and a broken counter shows as 0.0 ms. The code was put back, and the WASM build was made again. The 250 ms stop rule did not change, and its mutation check is in an older entry (the 1 µs submit limit).

Not measured on the iPhone. Not known for the iPhone: the image time, the longest submit, whether Safari keeps the tab alive for 20 images, and whether the WASM memory stays the same.

Conclusion: on the Mac, the engine makes 20 images in a row with no GPU or WASM memory growth. The longest submit of all 20 images is 143.0 ms, under the 250 ms limit.

Next step: the user opens `https://10.120.1.125:8443/engine/web/generate.html` on the iPhone, taps "Make 20", and sends the log.

## 2026-11-18: Time per kernel

Files added:

- `engine/src/profile/mod.rs`: the profile (one total per kernel name), `start_profile`, `best_mode`, and `profiled_submit`.
- `engine/src/profile/timestamps.rs`: one query set per profile, and one compute pass per dispatch with a GPU timestamp at its start and its end.
- `engine/src/profile/wall_clock.rs`: one awaited submit per dispatch, timed by the CPU clock.
- `engine/src/profile/scope.rs`: `caught`, which turns a GPU error inside a profiled submit into an engine error.
- `engine/src/profile/table.rs`: the rows and the printed table.

Files changed:

- `engine/src/dispatch.rs`: each `Dispatch` has a `Work`: its kernel name and the bytes that one run must read and write at least.
- Every place that makes a dispatch gives its `Work`: `layers/mod.rs` (all float and packed layers), `kernels/` (add, concat, geglu, gelu, group_norm, layer_norm, transpose add, upsample), `attention/mod.rs`, `attention/slots.rs`, `taesd/kernels.rs`, `silu/mod.rs`, `bench/copy.rs`, `bench/math.rs`, `weights/buffers.rs` (widen).
- `engine/src/submit.rs`: `submit_within_limit` uses the profiler while a profile runs. The new `check_submit` keeps the longest submit and applies the 250 ms limit to every submit, also to each submit of the wall clock mode.
- `engine/src/gpu.rs`: the device asks for `TIMESTAMP_QUERY` when the adapter has it. New: `has_timestamps`, `take_profile`, the profile field, `query_resolve_buffer`. `read_byte_prefix` is now `pub(crate)`.
- `engine/src/error.rs`: `NoTimestamps`, `TooManyTimestamps`, `GpuError`.
- `engine/src/bench/`: `copy_benchmark` is exported, and `BenchReport::median_rate` is new.
- `engine/src/web/engine.rs`: `startProfile(wallClock)` and `profileTable()`.
- `engine/web/generate.html`: the button "Profile". The address `generate.html?clock=wall` forces the CPU clock.
- `engine/examples/generate.rs`: the flags `--profile` and `--wall-clock`.

Design choices:

- The door number is the copy speed of module 3. The profile runs the same copy benchmark (`copy_benchmark`) on the same device just before the image, and uses its median. A fixed number would be wrong on every device but one. The first table line prints the door that it used.
- Bytes per dispatch: every tensor read once and written once, the same rule as `PreparedLayer::bytes_moved` in module 4. Packed weights count as their u32 words. The attention scores are scratch, so they do not count. A kernel that reads a value twice still counts it once, so a low share can also mean extra reads.
- WebGPU allows timestamps only at the start and the end of a pass. So in the timestamp mode, each dispatch gets its own compute pass, all in one submit. WebGPU makes each pass see the writes of the passes before it.
- Without timestamps, each dispatch is its own submit, and the CPU clock runs around the submit and the wait for onSubmittedWorkDone. This time includes the submit and the wait, so small kernels look slower.
- "calls" counts dispatches. Attention has one dispatch per chunk of query rows, so it has more calls than attention layers.
- The profile has one query set of 4096 timestamps (the WebGPU maximum, 2048 dispatches per submit) and one 32,768-byte buffer for the results. The buffer counts as read-back memory while the profile lives. A new query set per submit was rejected, see below.
- Every profiled submit runs inside GPU error scopes (validation, out of memory, internal). A GPU error stops the run with `GpuError`. Without this, Chromium skipped the bad submits and the engine gave a wrong image with no error.
- The table is sorted by time, slowest first. A kernel with 0 ms on the clock shows "-" for its speed.

Table format (new):

```
timing: <GPU timestamps, one compute pass per dispatch | CPU clock around one awaited submit per dispatch>; door: <GB/s> GB/s
kernel            calls   total ms    time     GB/s    door
<name>         <calls> <ms> <share of time>% <GB/s> <share of door>%
all kernels    <calls> <ms> 100.0%
```

Commands, in `engine/`: `cargo test --release` passed all 12 test files, 55 tests in total. The library tests printed `test result: ok. 43 passed`. `cargo clippy --release --all-targets` and `cargo clippy --release --target wasm32-unknown-unknown` gave no warning in the new or changed code. The one warning in a changed file, `src/kernels/group_norm.rs:33` (too many arguments), was there before this step. `wasm-pack build --release --target web --out-dir web/pkg --no-typescript` passed.

Command: `cargo run --release --example generate -- --packed --profile`. Device: Mac native.

```
copy: 368.22 GB/s median, 374.68 GB/s best (30 submits, 256 runs per submit, 93.3 ms median submit, 102.0 ms longest)
text_encoder: ran in 0.11 s; measured peak 1364742864 bytes (1364.7 MB)
unet: ran in 1.92 s; measured peak 813334128 bytes (813.3 MB)
taesd: ran in 0.35 s; measured peak 201507152 bytes (201.5 MB)
text_encoder: largest difference from PyTorch (float32) 2.2888184e-5
unet_output: largest difference from PyTorch (packed) 9.655952e-6
final_latent: largest difference from PyTorch (packed) 1.411438e-4
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
timing: GPU timestamps, one compute pass per dispatch; door: 368.22 GB/s
kernel            calls   total ms    time     GB/s    door
conv3x3_packed       52     995.84   49.5%     0.69    0.2%
attention           235     411.52   20.4%     2.92    0.8%
linear_packed       192     262.73   13.1%     6.30    1.7%
conv3x3              35     243.17   12.1%     4.33    1.2%
linear              138      57.11    2.8%    22.57    6.1%
conv1x1_packed       14      18.30    0.9%     6.41    1.7%
group_norm           61       7.74    0.4%    46.65   12.7%
vector_packed        24       3.73    0.2%     5.25    1.4%
layer_norm           95       3.58    0.2%    85.89   23.3%
add                 139       2.58    0.1%   271.40   73.7%
geglu                16       1.88    0.1%   294.61   80.0%
taesd relu           21       1.20    0.1%   444.55  120.7%
taesd add_relu       10       1.05    0.1%   381.47  103.6%
transpose add        16       0.91    0.0%   152.61   41.4%
concat               12       0.39    0.0%   320.14   86.9%
taesd upsample        3       0.35    0.0%   313.19   85.1%
gelu                 23       0.29    0.0%   198.15   53.8%
upsample              3       0.07    0.0%   299.64   81.4%
taesd image           1       0.02    0.0%   304.44   82.7%
taesd clamp           1       0.01    0.0%    21.25    5.8%
all kernels        1091    2012.49  100.0%
```

The same command without `--profile` printed `unet: ran in 1.85 s` and the same four largest differences. Command with `--profile --wall-clock`. Device: Mac native.

```
copy: 358.66 GB/s median, 364.72 GB/s best (30 submits, 256 runs per submit, 95.8 ms median submit, 100.1 ms longest)
text_encoder: ran in 0.16 s; measured peak 1364710096 bytes (1364.7 MB)
unet: ran in 2.20 s; measured peak 813301360 bytes (813.3 MB)
taesd: ran in 0.34 s; measured peak 201474352 bytes (201.5 MB)
text_encoder: largest difference from PyTorch (float32) 2.2888184e-5
unet_output: largest difference from PyTorch (packed) 9.655952e-6
final_latent: largest difference from PyTorch (packed) 1.411438e-4
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
timing: CPU clock around one awaited submit per dispatch; door: 358.66 GB/s
kernel            calls   total ms    time     GB/s    door
conv3x3_packed       52    1042.54   40.9%     0.66    0.2%
attention           235     546.94   21.5%     2.19    0.6%
linear_packed       192     366.94   14.4%     4.51    1.3%
conv3x3              35     276.19   10.8%     3.81    1.1%
linear              138      94.80    3.7%    13.60    3.8%
add                 139      57.98    2.3%    12.09    3.4%
layer_norm           95      37.37    1.5%     8.23    2.3%
group_norm           61      30.99    1.2%    11.65    3.2%
conv1x1_packed       14      28.23    1.1%     4.16    1.2%
vector_packed        24      13.82    0.5%     1.42    0.4%
geglu                16      13.55    0.5%    40.91   11.4%
concat               12       8.33    0.3%    15.10    4.2%
taesd relu           21       7.22    0.3%    74.05   20.6%
gelu                 23       6.90    0.3%     8.41    2.3%
transpose add        16       6.63    0.3%    20.92    5.8%
taesd upsample        3       3.99    0.2%    27.57    7.7%
taesd add_relu       10       3.92    0.2%   101.88   28.4%
upsample              3       0.94    0.0%    22.64    6.3%
taesd image           1       0.57    0.0%    11.00    3.1%
taesd clamp           1       0.28    0.0%     0.47    0.1%
all kernels        1091    2548.13  100.0%
```

This wall clock run used the first timestamp design. The wall clock path did not change after that, and its peaks have no timestamp buffer.

Page: `http://127.0.0.1:8765/engine/web/generate.html`, served by `python3 -m http.server 8765 --bind 127.0.0.1` in the project folder. Device: Mac Chromium. A Node script with `playwright-core@1.56.1` (in the scratchpad, not in the project) opened Chrome for Testing from `~/Library/Caches/ms-playwright/chromium-1254` with `--enable-unsafe-webgpu` and a persistent profile folder, clicked "Profile", and printed every console message that was not a plain log. The weights came from the OPFS cache. The only console message was `Failed to load resource: the server responded with a status of 404 (File not found)`, the missing page icon.

```
ready
copy: 357.54 GB/s median, 366.31 GB/s best (30 submits, 256 runs per submit, 96.1 ms median submit, 97.5 ms longest)
timestamp-query on this device: true
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from cache in 1.48 s
text_encoder: prepared; planned activations 2838528 bytes (2.8 MB); planned peak 1364742864 bytes (1364.7 MB); live 1364427472 bytes
text_encoder: ran in 0.20 s; measured peak 1364742864 bytes (1364.7 MB)
text_encoder: freed; live 32768 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from cache in 0.58 s
unet: prepared; planned activations 111141888 bytes (111.1 MB); planned peak 813334128 bytes (813.3 MB); live 813268592 bytes
unet: ran in 2.79 s; measured peak 813334128 bytes (813.3 MB)
unet: freed; live 32768 bytes
taesd: not planned; live 32768 bytes
taesd: ran in 0.43 s; measured peak 201507152 bytes (201.5 MB)
taesd: freed; live 32768 bytes
total: 5.57 s
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
timing: GPU timestamps, one compute pass per dispatch; door: 357.54 GB/s
kernel            calls   total ms    time     GB/s    door
conv3x3_packed       52    1393.07   49.5%     0.50    0.1%
attention           235     739.43   26.3%     1.62    0.5%
conv3x3              35     280.33   10.0%     3.75    1.1%
linear_packed       192     259.29    9.2%     6.39    1.8%
linear              138     102.34    3.6%    12.60    3.5%
conv1x1_packed       14      17.66    0.6%     6.65    1.9%
group_norm           61       7.17    0.3%    50.35   14.1%
layer_norm           95       3.81    0.1%    80.71   22.6%
vector_packed        24       3.42    0.1%     5.73    1.6%
add                 139       2.77    0.1%   253.46   70.9%
geglu                16       1.86    0.1%   297.88   83.3%
taesd relu           21       1.22    0.0%   437.12  122.3%
taesd add_relu       10       1.08    0.0%   370.94  103.7%
transpose add        16       0.90    0.0%   153.34   42.9%
gelu                 23       0.41    0.0%   141.34   39.5%
concat               12       0.41    0.0%   309.87   86.7%
taesd upsample        3       0.32    0.0%   343.85   96.2%
upsample              3       0.07    0.0%   306.10   85.6%
taesd image           1       0.02    0.0%   256.80   71.8%
taesd clamp           1       0.01    0.0%    19.78    5.5%
all kernels        1091    2815.59  100.0%
```

A plain Generate in the same profile, before the fix below, printed `unet: ran in 2.68 s` and `image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1`.

Page: `http://127.0.0.1:8765/engine/web/generate.html?clock=wall`, click on "Profile". Device: Mac Chromium. Lines with tensors, prepared and freed are left out here.

```
ready
copy: 353.50 GB/s median, 359.41 GB/s best (30 submits, 128 runs per submit, 48.6 ms median submit, 49.7 ms longest)
timestamp-query on this device: true
text_encoder: ran in 0.35 s; measured peak 1364710096 bytes (1364.7 MB)
unet: ran in 2.93 s; measured peak 813301360 bytes (813.3 MB)
taesd: ran in 0.39 s; measured peak 201474352 bytes (201.5 MB)
total: 6.21 s
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
timing: CPU clock around one awaited submit per dispatch; door: 353.50 GB/s
kernel            calls   total ms    time     GB/s    door
conv3x3_packed       52    1438.50   42.3%     0.48    0.1%
attention           235     854.90   25.1%     1.40    0.4%
linear_packed       192     336.00    9.9%     4.93    1.4%
conv3x3              35     312.10    9.2%     3.37    1.0%
linear              138     229.20    6.7%     5.62    1.6%
add                 139      59.00    1.7%    11.89    3.4%
layer_norm           95      36.50    1.1%     8.43    2.4%
group_norm           61      30.50    0.9%    11.83    3.3%
conv1x1_packed       14      25.40    0.7%     4.62    1.3%
taesd add_relu       10      23.50    0.7%    17.00    4.8%
vector_packed        24      12.40    0.4%     1.58    0.4%
gelu                 23      10.10    0.3%     5.75    1.6%
geglu                16       9.30    0.3%    59.62   16.9%
taesd relu           21       8.30    0.2%    64.43   18.2%
transpose add        16       6.60    0.2%    21.00    5.9%
concat               12       4.90    0.1%    25.68    7.3%
taesd upsample        3       3.90    0.1%    28.23    8.0%
upsample              3       0.80    0.0%    26.62    7.5%
taesd clamp           1       0.50    0.0%     0.26    0.1%
taesd image           1       0.40    0.0%    15.73    4.4%
all kernels        1091    3402.80  100.0%
```

Calculations:

- Top three on the Mac native, timestamps: 995.84 + 411.52 + 262.73 = 1,670.09 ms of 2,012.49 ms = 83.0%.
- Top three on Mac Chromium, timestamps: 1,393.07 + 739.43 + 280.33 = 2,412.83 ms of 2,815.59 ms = 85.7%.
- Top three on Mac Chromium, wall clock: 1,438.50 + 854.90 + 336.00 = 2,629.40 ms of 3,402.80 ms = 77.3%.
- Extra time per dispatch of the wall clock, compared with timestamps. Mac native: the wall clock run came right after a timestamp run of the first design, which printed `all kernels        1091    2001.58  100.0%`, with the same top four rows. (2,548.13 - 2,001.58) / 1,091 = 0.50 ms. Mac Chromium: (3,402.80 - 2,815.59) / 1,091 = 0.54 ms.
- Door share of conv3x3_packed on Mac native: 0.69 / 368.22 = 0.19%. The rule for every row: GB/s = bytes / seconds / 10^9, and door = GB/s / door GB/s.
- Timestamp buffer: 4,096 x 8 = 32,768 bytes. Text encoder peak: 1,364,742,864 - 1,364,710,096 = 32,768. TAESD peak: 201,507,152 - 201,474,352 = 32,800 = 32,768 + 32. The 32 bytes are the read-back copy of the 4 timestamps of a TAESD step with 2 dispatches. After `profileTable()`, the profile and its buffer are freed.
- Cost of the timestamp mode for the UNet stage: Mac native 1.92 - 1.85 = 0.07 s, Mac Chromium 2.79 - 2.68 = 0.11 s.
- Rows above 100%: `taesd relu` 444.55 / 368.22 = 120.7% on the Mac native. The copy benchmark reads one buffer and writes another. ReLU reads and writes one buffer, right after the convolution that wrote it. A likely cause is that part of the data is still in the GPU cache. This is not measured. So a share above 100% means "at the door", not an error.

Failed design: the first version made a new query set and result buffer in every submit. Mac native passed (the first table above). In Mac Chromium, the first click on "Profile" printed:

```
text_encoder: ran in 0.89 s; measured peak 1364710096 bytes (1364.7 MB)
unet: ran in 0.70 s; measured peak 813301360 bytes (813.3 MB)
taesd: ran in 0.27 s; measured peak 201474416 bytes (201.5 MB)
total: 17.64 s
image bytes: 781775 of 1048576 differ from golden/packed/taesd_image, by at most 152
```

The table of that run had 0.00 ms for `concat`, `taesd clamp` and `upsample`, and 813.73 ms for all kernels. The script printed only console errors, so a second click printed every console message that was not a plain log. The counts of those lines:

```
 167 console: Error creating query set: Cannot allocate sample buffer
   1 console: Failed to load resource: the server responded with a status of 404 (File not found)
   1 console: WebGPU: too many warnings, no more warnings will be reported to the console for this GPUDevice.
 166 console: [Invalid CommandBuffer] is invalid due to a previous error.
 167 console: [Invalid QuerySet "kernel timestamps"] is invalid due to a previous error.
   1 image bytes: 782337 of 1048576 differ from golden/packed/taesd_image, by at most 168
   1 taesd: ran in 0.20 s; measured peak 201474416 bytes (201.5 MB)
   1 text_encoder: ran in 0.16 s; measured peak 1364710096 bytes (1364.7 MB)
   1 unet: ran in 0.74 s; measured peak 813301360 bytes (813.3 MB)
```

Cause: Chromium on the Mac could not allocate more timestamp sample buffers, so query set creation failed. Each submit with the bad query set was invalid, and Chromium skipped it. The kernels in it never ran, but the engine got no error, so it went on and made a wrong image. The fix is the design above: one query set for the whole profile, and error scopes around every profiled submit. After the fix, the image is the same as a plain Generate.

Mutation checks:

- The sort in `table.rs` was changed to fastest first. The test `rows_are_slowest_first_with_shares_of_time_and_door` failed with `left: ProfileRow { kernel: "idle", ... }` and `right: ProfileRow { kernel: "conv", ... }`. The code was put back.
- `dispatch_seconds` was changed to use the end timestamp alone. The test `pairs_start_and_end_of_each_dispatch` failed with `left: [800.0, 2800.0, 1600.0]` and `right: [600.0, 2000.0, 0.0]`. The code was put back.
- The query set was made with 4097 timestamps, one more than WebGPU allows. `cargo run --release --example generate -- --packed --profile` stopped with exit code 1:

```
generate failed: timestamp query set: the GPU reported an error: Validation Error

Caused by:
  In Device::create_query_set
    4097 is too many queries for a single QuerySet. QuerySets cannot be made more than 4096 queries.
```

The code was put back, and the WASM build was made again.

Not measured on the iPhone. Not known for the iPhone: whether Safari offers `timestamp-query`, the door number in this run, the table, and whether the timestamp mode keeps every submit under 250 ms. The page prints `timestamp-query on this device: true` or `false`. If it is false, the page uses the wall clock. The iPhone submit times in this log are all whole milliseconds (for example `90.0 ms median submit, 91.0 ms longest`), so a wall clock time of a small kernel on the iPhone can be 0 or 1 ms. In that case, only the rows with many milliseconds are useful.

Conclusion: the engine times each kernel with GPU timestamps on the Mac, natively and in Chromium, and with the CPU clock where timestamps are missing. On both Mac runtimes, `conv3x3_packed` and `attention` are the top two rows. They hold (995.84 + 411.52) / 2,012.49 = 69.9% (native) and (1,393.07 + 739.43) / 2,815.59 = 75.7% (Chromium) of the kernel time, and each runs below 1% of the door. The third row is `linear_packed` on the Mac native and `conv3x3` in Chromium.

Next step: the user opens `https://10.120.1.125:8443/engine/web/generate.html` on the iPhone, taps "Profile", and sends the log.

## 2026-11-18: Time per kernel on the iPhone

Files changed: none. This step is one measurement that the user sent.

Page: `https://10.120.1.125:8443/engine/web/generate.html`, the button "Profile". Device: iPhone. The user sent the log as text. The weights came from the OPFS cache. The user did not say whether this was the first click after the page opened.

```
ready
copy: 47.72 GB/s median, 48.81 GB/s best (30 submits, 32 runs per submit, 90.0 ms median submit, 94.0 ms longest)
timestamp-query on this device: true
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from cache in 1.02 s
text_encoder: prepared; planned activations 2838528 bytes (2.8 MB); planned peak 1364742864 bytes (1364.7 MB); live 1364427472 bytes
text_encoder: ran in 0.36 s; measured peak 1364742864 bytes (1364.7 MB)
text_encoder: freed; live 32768 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from cache in 0.41 s
unet: prepared; planned activations 111141888 bytes (111.1 MB); planned peak 813334128 bytes (813.3 MB); live 813268592 bytes
unet: ran in 4.15 s; measured peak 813334128 bytes (813.3 MB)
unet: freed; live 32768 bytes
taesd: not planned; live 32768 bytes
taesd: ran in 0.59 s; measured peak 201507152 bytes (201.5 MB)
taesd: freed; live 32768 bytes
total: 6.94 s
image bytes: 1161 of 1048576 differ from golden/packed/taesd_image, by at most 1
timing: GPU timestamps, one compute pass per dispatch; door: 47.72 GB/s
kernel            calls   total ms    time     GB/s    door
conv3x3_packed       52    2356.59   53.2%     0.29    0.6%
attention           235     825.54   18.6%     1.45    3.0%
linear_packed       192     466.83   10.5%     3.55    7.4%
conv3x3              35     402.60    9.1%     2.61    5.5%
linear              138     210.88    4.8%     6.11   12.8%
conv1x1_packed       14      88.25    2.0%     1.33    2.8%
group_norm           61      17.20    0.4%    20.99   44.0%
add                 139      12.01    0.3%    58.38  122.3%
geglu                16      10.43    0.2%    53.17  111.4%
taesd relu           21       9.37    0.2%    57.09  119.6%
taesd add_relu       10       8.10    0.2%    49.31  103.3%
layer_norm           95       7.80    0.2%    39.43   82.6%
vector_packed        24       4.01    0.1%     4.89   10.2%
taesd upsample        3       3.77    0.1%    29.21   61.2%
transpose add        16       2.52    0.1%    55.02  115.3%
concat               12       2.06    0.0%    60.94  127.7%
gelu                 23       0.86    0.0%    67.84  142.2%
upsample              3       0.27    0.0%    77.75  162.9%
taesd image           1       0.07    0.0%    89.03  186.6%
taesd clamp           1       0.01    0.0%    25.37   53.2%
all kernels        1091    4429.16  100.0%
```

Calculations:

- Safari offers `timestamp-query`, so this table uses GPU timestamps, not the wall clock. Small kernels show times of 0.01 ms and 0.07 ms, so the timestamps are finer than the whole milliseconds of the Safari clock.
- The door of this run, 47.72 GB/s median, is the same number as the copy median of "The first iPhone benchmark failed" (module 3).
- Top three rows: 2,356.59 + 825.54 + 466.83 = 3,648.96 ms of 4,429.16 ms = 82.4%. The top two: (2,356.59 + 825.54) / 4,429.16 = 71.8%.
- Door shares of the top three: `conv3x3_packed` 0.29 / 47.72 = 0.6%, `attention` 1.45 / 47.72 = 3.0%, `linear_packed` 3.55 / 47.72 = 7.4%.
- Kernel time over stage time: 4,429.16 ms / (0.36 + 4.15 + 0.59 = 5.10 s) = 86.8%. The rest is the waits between submits, the timestamp read-back, and the CPU work between the steps.
- iPhone time over Mac Chromium time (timestamp table of "Time per kernel"): `conv3x3_packed` 2,356.59 / 1,393.07 = 1.69, `attention` 825.54 / 739.43 = 1.12, `linear_packed` 466.83 / 259.29 = 1.80, `conv3x3` 402.60 / 280.33 = 1.44, `linear` 210.88 / 102.34 = 2.06, `conv1x1_packed` 88.25 / 17.66 = 5.00. All kernels: 4,429.16 / 2,815.59 = 1.57.
- Ten rows are above 100% of the door, up to 186.6% (`taesd image`). Each of these kernels moves at most a few MB per call, and most read a tensor that the kernel just before wrote. A likely cause is that the data is still in the GPU cache. This is not measured. These rows hold 0.3 + 0.2 + 0.2 + 0.2 + 0.1 + 0.0 + 0.0 + 0.0 + 0.0 = 1.0% of the time or less, so they do not matter for the next step.
- Peaks: the planned and measured peaks are the same in each stage, and they are the Mac numbers plus the 32,768-byte timestamp buffer. The text stage peak is 1,364,742,864 / 1,920,000,000 = 71.1% of the tab limit, and the tab stayed alive.
- Image: 1,161 of 1,048,576 bytes differ from the packed PyTorch image, by at most 1. Mac Chromium gave 1,199. Safari compiles the WGSL with another compiler, so the float rounding can differ a little.
- Every submit stayed under 250 ms, because `check_submit` stops the run at the first submit that does not, and the run finished. The log does not print the longest submit.

This is the first log entry with a whole generation on the iPhone. Before this, the three stages ran on the iPhone only one at a time, or not at all. Total time: 6.94 s, with 1.02 + 0.41 = 1.43 s for the cache loads. The profile itself adds time, so this is not the time of a plain Generate on the iPhone. That time is not measured.

Conclusion: on the iPhone, `conv3x3_packed` holds 53.2% of the kernel time and runs at 0.6% of the door. `attention` (18.6%, 3.0% of the door) and `linear_packed` (10.5%, 7.4% of the door) come next. The order of the top three is the same as on the Mac native. All three run far below the door, so all three are worth fixing, and `conv3x3_packed` first.

Next step: module 8.2. Propose one change for each of the top three kernels, then make one change at a time, with timing before and after, and the golden tests.

## 2026-11-18: Three proposals, and change 1: the workgroup of the packed 3x3 convolution

The user asked for one change for each of the top three kernels of "Time per kernel on the iPhone", made one at a time. This step makes change 1 only. Changes 2 and 3 are the next steps.

The three proposals:

1. `conv3x3_packed` (53.2% of the iPhone kernel time): a better block size. The workgroup was 8 x 8 x 1 invocations: 8 pixel blocks of 4 pixels, 8 rows, 1 block of 4 channels. On a 16 x 16 image a row has only 4 pixel blocks, so half of the lanes did nothing. On an 8 x 8 image, three quarters did nothing. The new workgroup is 2 x 8 x 4: 2 pixel blocks, 8 rows, 4 channel blocks. It fills every lane on every UNet image.
2. `attention` (18.6%): merge two passes into one. The kernel writes every score to a scratch buffer in its first loop and reads it back in its second loop. A running maximum and a running sum (the "online softmax") let one loop do both, so the scores never go to memory.
3. `linear_packed` (10.5%): a better block size. Each invocation computes 4 rows x 4 columns, so each unpacked weight feeds only 4 rows. A block of 8 rows x 4 columns halves the unpacking work per multiply-add.

Files added:

- `engine/src/layers/packed/conv3x3.rs`: `conv3x3_workgroups`, the grid of the packed 3x3 convolution, with its own `WORKGROUP = [2, 8, 4]`.

Files changed:

- `engine/src/layers/packed/conv3x3.wgsl`: `@workgroup_size(8, 8, 1)` became `@workgroup_size(2, 8, 4)`. The body of the shader did not change. Each invocation still computes 4 channels x 4 pixels.
- `engine/src/layers/packed/mod.rs`: `plan_packed` takes the grid of the packed 3x3 convolution from `conv3x3_workgroups`. The packed linear and 1x1 kernels keep the float grid.

How the work splits over the image sizes. The UNet has 52 3x3 convolutions, all packed. Operations = 2 x in x out x 9 x side x side, summed over the 52 layers from the UNet config, = 400.33 G operations per image. By output side: 64 x 64 has 34.0%, 32 x 32 has 32.5%, 16 x 16 has 25.0%, 8 x 8 has 8.5%. With the old workgroup, the lanes in use were 100%, 100%, 50% and 25%.

iPhone speed of the old kernel: 400.33 G operations / 2.35659 s = 169.9 G ops/s = 26.7% of the 636.29 G ops/s math rate of module 3.

Design tried and rejected (shape B): each invocation computes 4 channels x 8 pixels, with a workgroup of 1 x 8 x 8. Each unpacked weight then feeds 8 multiply-adds instead of 4. The code was a trial copy of the shader and was removed after the runs below.

Commands: `cargo run --release --example generate -- --packed --profile` (Mac native), and "Profile" on `http://127.0.0.1:8765/engine/web/generate.html` (Mac Chromium, the Playwright script of "Time per kernel"). `cargo test --release --test packed_layers -- --nocapture` (Mac native). Only the lines below were kept from each run.

Before, Mac native, two runs (the third number is from "Time per kernel"):

```
unet: ran in 1.84 s; measured peak 813334128 bytes (813.3 MB)
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
conv3x3_packed       52     984.39   49.4%     0.70    0.2%
all kernels        1091    1991.86  100.0%
unet: ran in 1.83 s; measured peak 813334128 bytes (813.3 MB)
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
conv3x3_packed       52     986.02   49.4%     0.70    0.2%
all kernels        1091    1997.26  100.0%
```

```
conv3x3_q4: 0.85 GB/s median, 0.87 GB/s best (30 submits, 4 runs per submit, 51.3 ms median submit, 54.0 ms longest), largest error 6.198883e-6
conv3x3_q8: 0.86 GB/s median, 0.89 GB/s best (30 submits, 4 runs per submit, 52.9 ms median submit, 57.5 ms longest), largest error 6.198883e-6
conv3x3_stride2_q4: 0.92 GB/s median, 0.92 GB/s best (30 submits, 8 runs per submit, 61.1 ms median submit, 61.3 ms longest), largest error 5.841255e-6
conv3x3_stride2_q8: 0.98 GB/s median, 0.98 GB/s best (30 submits, 8 runs per submit, 61.1 ms median submit, 61.4 ms longest), largest error 6.198883e-6
```

Before, Mac Chromium: `conv3x3_packed       52    1393.07   49.5%     0.50    0.1%`, from "Time per kernel".

Shape A (kept), Mac native, two runs:

```
unet: ran in 1.59 s; measured peak 813334128 bytes (813.3 MB)
text_encoder: largest difference from PyTorch (float32) 2.2888184e-5
unet_output: largest difference from PyTorch (packed) 9.655952e-6
final_latent: largest difference from PyTorch (packed) 1.411438e-4
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
conv3x3_packed       52     742.57   42.3%     0.93    0.3%
all kernels        1091    1756.18  100.0%
unet: ran in 1.58 s; measured peak 813334128 bytes (813.3 MB)
text_encoder: largest difference from PyTorch (float32) 2.2888184e-5
unet_output: largest difference from PyTorch (packed) 9.655952e-6
final_latent: largest difference from PyTorch (packed) 1.411438e-4
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
conv3x3_packed       52     741.05   41.8%     0.93    0.3%
all kernels        1091    1773.37  100.0%
```

```
conv3x3_q4: 0.64 GB/s median, 0.78 GB/s best (30 submits, 4 runs per submit, 68.9 ms median submit, 81.7 ms longest), largest error 6.198883e-6
conv3x3_q8: 0.61 GB/s median, 0.66 GB/s best (30 submits, 4 runs per submit, 74.6 ms median submit, 85.6 ms longest), largest error 6.198883e-6
conv3x3_stride2_q4: 1.71 GB/s median, 1.89 GB/s best (30 submits, 16 runs per submit, 65.6 ms median submit, 72.7 ms longest), largest error 5.841255e-6
conv3x3_stride2_q8: 1.81 GB/s median, 1.99 GB/s best (30 submits, 16 runs per submit, 66.3 ms median submit, 73.4 ms longest), largest error 6.198883e-6
```

Shape A, Mac Chromium, two clicks on "Profile":

```
unet: ran in 2.45 s; measured peak 813334128 bytes (813.3 MB)
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
conv3x3_packed       52    1046.30   42.4%     0.66    0.2%
all kernels        1091    2466.56  100.0%
unet: ran in 2.45 s; measured peak 813334128 bytes (813.3 MB)
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
conv3x3_packed       52    1031.92   42.0%     0.67    0.2%
all kernels        1091    2459.33  100.0%
```

Shape B (rejected), Mac native, two runs:

```
unet: ran in 1.76 s; measured peak 813334128 bytes (813.3 MB)
unet_output: largest difference from PyTorch (packed) 9.655952e-6
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
conv3x3_packed       52     858.12   45.1%     0.80    0.2%
all kernels        1091    1903.06  100.0%
unet: ran in 2.01 s; measured peak 813334128 bytes (813.3 MB)
unet_output: largest difference from PyTorch (packed) 9.655952e-6
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
conv3x3_packed       52     914.38   44.8%     0.75    0.2%
all kernels        1091    2039.31  100.0%
```

```
conv3x3_q4: 0.92 GB/s median, 0.96 GB/s best (30 submits, 8 runs per submit, 95.0 ms median submit, 96.4 ms longest), largest error 6.198883e-6
conv3x3_q8: 0.96 GB/s median, 0.98 GB/s best (30 submits, 8 runs per submit, 95.0 ms median submit, 98.0 ms longest), largest error 6.198883e-6
conv3x3_stride2_q4: 1.98 GB/s median, 2.01 GB/s best (30 submits, 16 runs per submit, 56.6 ms median submit, 60.1 ms longest), largest error 5.841255e-6
conv3x3_stride2_q8: 1.82 GB/s median, 1.97 GB/s best (30 submits, 16 runs per submit, 65.7 ms median submit, 72.9 ms longest), largest error 6.198883e-6
```

Shape B, Mac Chromium, two clicks:

```
unet: ran in 2.97 s; measured peak 813334128 bytes (813.3 MB)
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
conv3x3_packed       52    1214.65   45.3%     0.57    0.2%
all kernels        1091    2679.77  100.0%
unet: ran in 2.63 s; measured peak 813334128 bytes (813.3 MB)
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
conv3x3_packed       52    1194.20   45.3%     0.58    0.2%
all kernels        1091    2634.68  100.0%
```

After shape A was put back: `cargo test --release` passed all 12 test files, 55 tests. `cargo test --release --test pipeline --test unet_packed -- --nocapture --test-threads 1` printed `image bytes: 547 of 1048576 differ, by at most 1` (float32) and `image bytes: 1140 of 1048576 differ, by at most 1` (packed), the same as in "WASM entry for the three stages, and the generate page". The four largest differences of the packed run are the same numbers as before the change. `cargo clippy --release --all-targets` gave no new warning. Its two warnings in `src/layers/packed/mod.rs` (lines 123 and 129, too many arguments) were at lines 118 and 124 before this step. `wasm-pack build --release --target web --out-dir web/pkg --no-typescript` passed. One more full native run with shape A, Mac native:

```
copy: 332.62 GB/s median, 346.60 GB/s best (30 submits, 128 runs per submit, 51.7 ms median submit, 52.9 ms longest)
text_encoder: ran in 0.10 s; measured peak 1364742864 bytes (1364.7 MB)
unet: ran in 1.69 s; measured peak 813334128 bytes (813.3 MB)
taesd: ran in 0.37 s; measured peak 201507152 bytes (201.5 MB)
text_encoder: largest difference from PyTorch (float32) 2.2888184e-5
unet_output: largest difference from PyTorch (packed) 9.655952e-6
final_latent: largest difference from PyTorch (packed) 1.411438e-4
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
timing: GPU timestamps, one compute pass per dispatch; door: 332.62 GB/s
kernel            calls   total ms    time     GB/s    door
conv3x3_packed       52     798.23   43.1%     0.86    0.3%
attention           235     418.84   22.6%     2.86    0.9%
linear_packed       192     271.68   14.7%     6.10    1.8%
conv3x3              35     261.46   14.1%     4.03    1.2%
linear              138      56.76    3.1%    22.71    6.8%
conv1x1_packed       14      20.73    1.1%     5.66    1.7%
group_norm           61       8.09    0.4%    44.64   13.4%
layer_norm           95       3.84    0.2%    79.99   24.0%
vector_packed        24       3.60    0.2%     5.45    1.6%
add                 139       3.10    0.2%   226.37   68.1%
geglu                16       1.96    0.1%   283.09   85.1%
taesd relu           21       1.64    0.1%   326.59   98.2%
taesd add_relu       10       1.05    0.1%   381.47  114.7%
transpose add        16       0.92    0.0%   151.34   45.5%
gelu                 23       0.52    0.0%   110.91   33.3%
taesd upsample        3       0.36    0.0%   305.66   91.9%
concat               12       0.36    0.0%   351.36  105.6%
upsample              3       0.07    0.0%   310.37   93.3%
taesd image           1       0.02    0.0%   288.72   86.8%
taesd clamp           1       0.01    0.0%    19.78    5.9%
all kernels        1091    1853.21  100.0%
```

In this run the copy benchmark was also lower (332.62 GB/s, against 360.60 to 368.22 GB/s before), so the Mac was slower in this run in general.

Calculations:

- Mac native, `conv3x3_packed`: before, mean of 984.39, 986.02 and 995.84 = 988.75 ms. Shape A, mean of 742.57, 741.05 and 798.23 = 760.62 ms, 23.1% less. Shape B, mean of 858.12 and 914.38 = 886.25 ms, 10.4% less.
- Mac Chromium, `conv3x3_packed`: before 1,393.07 ms (one run). Shape A, mean of 1,046.30 and 1,031.92 = 1,039.11 ms, 25.4% less. Shape B, mean of 1,214.65 and 1,194.20 = 1,204.43 ms, 13.5% less.
- Shape A as math speed: Mac native 400.33 / 0.76062 = 526.3 G ops/s, Mac Chromium 400.33 / 1.03911 = 385.3 G ops/s.
- The isolated tests disagree with the whole UNet. They are 320 to 320 channels at 64 x 64 (stride 1) and at 32 x 32 output (stride 2). At 64 x 64, shape A is slower than before (0.64 against 0.85 GB/s). The stride 2 case is faster (1.71 against 0.92 GB/s). In the whole UNet, the layers at 16 x 16 and 8 x 8 gain the most, and those layers have no isolated test. So the UNet time decides, not the isolated tests.
- Shape B is faster than A in the isolated tests, but slower in the whole UNet, on both Mac runtimes. So shape A is kept.

Mutation check: `WORKGROUP` in `packed/conv3x3.rs` was changed to `[2, 8, 8]`, so the Rust grid no longer matched `@workgroup_size(2, 8, 4)`. `cargo test --release --test packed_layers -- --nocapture` failed:

```
largest error above 1e-4: ["conv3x3_q4: 2.7906196e0", "conv3x3_q8: 2.815799e0", "conv3x3_stride2_q4: 2.647364e0", "conv3x3_stride2_q8: 2.6620228e0"]
```

The grid then covers only half of the channel blocks, so half of the output stays 0. The code was put back.

Not measured on the iPhone. Not known for the iPhone: the time of `conv3x3_packed` with shape A, and whether the image still matches.

Conclusion: on both Mac runtimes, the 2 x 8 x 4 workgroup cuts the time of the packed 3x3 convolution by about a quarter, and the golden tests still pass. Shape B (8 pixels per invocation) is slower than A on the whole UNet.

Next step: the user taps "Profile" on the iPhone and sends the log. The iPhone "before" is 2,356.59 ms. Then change 2: one pass in attention.

## 2026-11-18: Change 1 on the iPhone

Files changed: none. This step is one measurement that the user sent, for change 1 of "Three proposals, and change 1: the workgroup of the packed 3x3 convolution".

Page: `https://10.120.1.125:8443/engine/web/generate.html`, the button "Profile". Device: iPhone. The user sent the log as text. The weights came from the OPFS cache.

```
ready
copy: 45.69 GB/s median, 48.26 GB/s best (30 submits, 32 runs per submit, 94.0 ms median submit, 117.0 ms longest)
timestamp-query on this device: true
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from cache in 1.22 s
text_encoder: prepared; planned activations 2838528 bytes (2.8 MB); planned peak 1364742864 bytes (1364.7 MB); live 1364427472 bytes
text_encoder: ran in 0.36 s; measured peak 1364742864 bytes (1364.7 MB)
text_encoder: freed; live 32768 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from cache in 0.47 s
unet: prepared; planned activations 111141888 bytes (111.1 MB); planned peak 813334128 bytes (813.3 MB); live 813268592 bytes
unet: ran in 3.36 s; measured peak 813334128 bytes (813.3 MB)
unet: freed; live 32768 bytes
taesd: not planned; live 32768 bytes
taesd: ran in 0.67 s; measured peak 201507152 bytes (201.5 MB)
taesd: freed; live 32768 bytes
total: 6.59 s
image bytes: 1161 of 1048576 differ from golden/packed/taesd_image, by at most 1
timing: GPU timestamps, one compute pass per dispatch; door: 45.69 GB/s
kernel            calls   total ms    time     GB/s    door
conv3x3_packed       52    1610.41   43.0%     0.43    0.9%
attention           235     851.92   22.8%     1.41    3.1%
linear_packed       192     470.31   12.6%     3.52    7.7%
conv3x3              35     465.69   12.4%     2.26    4.9%
linear              138     194.01    5.2%     6.64   14.5%
conv1x1_packed       14      61.15    1.6%     1.92    4.2%
group_norm           61      19.15    0.5%    18.85   41.3%
add                 139      14.58    0.4%    48.09  105.3%
geglu                16      12.77    0.3%    43.43   95.1%
taesd relu           21      10.35    0.3%    51.65  113.0%
layer_norm           95       9.28    0.2%    33.16   72.6%
taesd add_relu       10       8.63    0.2%    46.30  101.3%
taesd upsample        3       4.34    0.1%    25.34   55.5%
vector_packed        24       4.01    0.1%     4.88   10.7%
transpose add        16       2.65    0.1%    52.33  114.5%
concat               12       2.17    0.1%    58.04  127.0%
gelu                 23       1.01    0.0%    57.58  126.0%
upsample              3       0.30    0.0%    70.55  154.4%
taesd image           1       0.07    0.0%    84.97  186.0%
taesd clamp           1       0.02    0.0%     8.28   18.1%
all kernels        1091    3742.82  100.0%
```

Calculations, against "Time per kernel on the iPhone" (one run before, one run after):

- `conv3x3_packed`: 2,356.59 ms before, 1,610.41 ms after. 1,610.41 / 2,356.59 - 1 = 31.7% less, 746.18 ms saved.
- As math speed: 400.33 G operations / 1.61041 s = 248.6 G ops/s = 39.1% of the 636.29 G ops/s math rate. Before: 169.9 G ops/s, 26.7%.
- UNet stage: 4.15 s before, 3.36 s after, 19.0% less. All kernels: 4,429.16 ms before, 3,742.82 ms after, 15.5% less.
- The image did not change: 1,161 of 1,048,576 bytes differ, by at most 1, the same as before.
- The iPhone was slower in general in this run. The copy median fell from 47.72 to 45.69 GB/s (4.3% less). The kernels that did not change are mostly slower too: `attention` +3.2% (825.54 to 851.92 ms), `linear_packed` +0.7% (466.83 to 470.31 ms), `conv3x3` +15.7% (402.60 to 465.69 ms). `linear` is the exception, 8.0% less (210.88 to 194.01 ms). The cause is not measured. A warmer phone is one possible cause. So the gain of `conv3x3_packed` is not from a faster phone.
- The longest copy submit was 117.0 ms, under the 250 ms limit. The run finished, so every profiled submit was under the limit too.
- New top three: `conv3x3_packed` 43.0%, `attention` 22.8%, `linear_packed` 12.6%. Together (1,610.41 + 851.92 + 470.31) / 3,742.82 = 78.4%. `conv3x3` (the float TAESD convolution) is close behind, at 12.4%.

The Mac predicted the iPhone gain well: 23.1% less on the Mac native and 25.4% less in Mac Chromium, against 31.7% less on the iPhone.

Conclusion: on the iPhone, change 1 cuts the packed 3x3 convolution from 2,356.59 to 1,610.41 ms, and the image stays the same. It is still the slowest kernel, at 39.1% of the math rate.

Next step: change 2, attention in one pass (online softmax), with timing before and after and the golden tests.

## 2026-11-18: Change 1b: each input value loaded once per row in the packed 3x3 convolution

The user let the agent pick the order of the next changes. `conv3x3_packed` was still the largest row on the iPhone (1,610.41 ms, 43.0%, in "Change 1 on the iPhone"), so it got a second change before attention.

Files changed:

- `engine/src/layers/packed/conv3x3.wgsl`: a fast path for stride 1, and the helpers `load_column` and `four_weights`. The old loop stays for stride 2, unchanged.

Design choices:

- The old inner loop ran once per input channel, kernel row and kernel column. Each time, it loaded 4 input values, each with 4 bounds checks, for 16 multiply-adds. Over the 3 kernel columns, that is 12 loads for 48 multiply-adds. But with stride 1, the 3 kernel columns under 4 pixels touch only 6 input columns, x0 - 1 to x0 + 4. The fast path loads those 6 values once per input row and builds the 3 windows of 4 from them.
- The row check runs once per input row. A row above or below the image is all padding, so the loop skips it.
- Stride 1 is 49 of the 52 UNet 3x3 convolutions. The 3 stride 2 layers have 2 x (320 x 320 x 1,024 + 640 x 640 x 256 + 1,280 x 1,280 x 64) x 9 = 5.66 G operations, 1.4% of the 400.33 G. They keep the old loop, so the change stays small. The branch is on a uniform value, so all lanes take the same side.
- This is not one of the four kinds of change in the plan of module 8.2. It is "fewer loads per multiply-add" inside the same block.

Commands as in "Three proposals, and change 1". Before is shape A from that entry: Mac native mean 760.62 ms, Mac Chromium mean 1,039.11 ms.

`cargo test --release --test packed_layers -- --nocapture`, Mac native:

```
conv3x3_q4: 0.83 GB/s median, 0.90 GB/s best (30 submits, 8 runs per submit, 105.1 ms median submit, 112.2 ms longest), largest error 6.198883e-6
conv3x3_q8: 0.87 GB/s median, 0.94 GB/s best (30 submits, 4 runs per submit, 52.5 ms median submit, 57.9 ms longest), largest error 6.198883e-6
conv3x3_stride2_q4: 1.69 GB/s median, 1.76 GB/s best (30 submits, 16 runs per submit, 66.3 ms median submit, 69.9 ms longest), largest error 5.841255e-6
conv3x3_stride2_q8: 1.81 GB/s median, 1.92 GB/s best (30 submits, 16 runs per submit, 66.0 ms median submit, 68.0 ms longest), largest error 6.198883e-6
```

`cargo run --release --example generate -- --packed --profile`, Mac native, two runs:

```
copy: 351.17 GB/s median, 356.72 GB/s best (30 submits, 128 runs per submit, 48.9 ms median submit, 49.9 ms longest)
unet: ran in 1.57 s; measured peak 813334128 bytes (813.3 MB)
unet_output: largest difference from PyTorch (packed) 9.655952e-6
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
conv3x3_packed       52     620.85   37.4%     1.11    0.3%
all kernels        1091    1659.18  100.0%
copy: 337.13 GB/s median, 350.52 GB/s best (30 submits, 256 runs per submit, 101.9 ms median submit, 108.1 ms longest)
unet: ran in 1.60 s; measured peak 813334128 bytes (813.3 MB)
unet_output: largest difference from PyTorch (packed) 9.655952e-6
taesd_image: largest difference from PyTorch (packed) 6.182194e-4
conv3x3_packed       52     629.69   36.9%     1.10    0.3%
all kernels        1091    1705.74  100.0%
```

"Profile" on `http://127.0.0.1:8765/engine/web/generate.html`, Mac Chromium, two clicks:

```
copy: 347.42 GB/s median, 354.96 GB/s best (30 submits, 128 runs per submit, 49.4 ms median submit, 61.1 ms longest)
unet: ran in 2.02 s; measured peak 813334128 bytes (813.3 MB)
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
conv3x3_packed       52     482.29   24.9%     1.43    0.4%
all kernels        1091    1933.34  100.0%
copy: 349.18 GB/s median, 357.91 GB/s best (30 submits, 128 runs per submit, 49.2 ms median submit, 51.7 ms longest)
unet: ran in 1.86 s; measured peak 813334128 bytes (813.3 MB)
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
conv3x3_packed       52     486.09   25.4%     1.42    0.4%
all kernels        1091    1911.55  100.0%
```

A third click, after the final WASM build, Mac Chromium (lines with tensors, prepared and freed are left out):

```
console: Failed to load resource: the server responded with a status of 404 (File not found)
copy: 359.04 GB/s median, 365.53 GB/s best (30 submits, 128 runs per submit, 47.9 ms median submit, 54.9 ms longest)
text_encoder: ran in 0.22 s; measured peak 1364742864 bytes (1364.7 MB)
unet: ran in 1.80 s; measured peak 813334128 bytes (813.3 MB)
taesd: ran in 0.40 s; measured peak 201507152 bytes (201.5 MB)
total: 4.33 s
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
timing: GPU timestamps, one compute pass per dispatch; door: 359.04 GB/s
kernel            calls   total ms    time     GB/s    door
attention           235     713.82   37.4%     1.68    0.5%
conv3x3_packed       52     487.69   25.5%     1.41    0.4%
conv3x3              35     280.05   14.7%     3.76    1.0%
linear_packed       192     259.18   13.6%     6.39    1.8%
linear              138     118.92    6.2%    10.84    3.0%
conv1x1_packed       14      24.17    1.3%     4.86    1.4%
group_norm           61       7.06    0.4%    51.12   14.2%
vector_packed        24       6.67    0.3%     2.94    0.8%
layer_norm           95       4.52    0.2%    68.03   18.9%
add                 139       2.58    0.1%   271.44   75.6%
geglu                16       1.67    0.1%   331.38   92.3%
taesd relu           21       1.15    0.1%   463.55  129.1%
taesd add_relu       10       1.01    0.1%   397.26  110.6%
transpose add        16       0.90    0.0%   153.71   42.8%
gelu                 23       0.45    0.0%   128.12   35.7%
concat               12       0.35    0.0%   354.66   98.8%
taesd upsample        3       0.35    0.0%   312.09   86.9%
upsample              3       0.07    0.0%   307.94   85.8%
taesd image           1       0.02    0.0%   294.92   82.1%
taesd clamp           1       0.01    0.0%    19.54    5.4%
all kernels        1091    1910.65  100.0%
```

Golden tests after the change: `cargo test --release` passed all 12 test files, 55 tests. `cargo test --release --test pipeline -- --nocapture --test-threads 1` printed `image bytes: 547 of 1048576 differ, by at most 1` (float32) and `image bytes: 1140 of 1048576 differ, by at most 1` (packed), the same as before. The packed largest errors and the packed UNet differences are the same numbers as before. `cargo clippy --release --all-targets` gave no new warning. `wasm-pack build --release --target web --out-dir web/pkg --no-typescript` passed.

Calculations:

- Mac native, `conv3x3_packed`: mean of 620.85 and 629.69 = 625.27 ms, against 760.62 ms with shape A alone: 17.8% less.
- Mac Chromium: mean of 482.29, 486.09 and 487.69 = 485.36 ms, against 1,039.11 ms: 53.3% less. Against 1,393.07 ms before change 1: 65.2% less.
- As math speed: Mac native 400.33 / 0.62527 = 640.2 G ops/s, Mac Chromium 400.33 / 0.48536 = 824.8 G ops/s.
- The isolated 64 x 64 test went from 0.64 to 0.83 GB/s (q4). The stride 2 tests did not change (1.69 against 1.71 GB/s), as expected, because they use the old loop.
- Mac Chromium, all kernels: 1,910.65 ms, against the mean 2,462.95 ms of shape A: 22.4% less. In Mac Chromium, `attention` is now the top row (37.4%).

Mutation check: in the fast path, `w_middle` read the weights of kernel column 2 (`index + 2u`) instead of column 1. `cargo test --release --test packed_layers` failed:

```
largest error above 1e-4: ["conv3x3_q4: 2.88949e0", "conv3x3_q8: 2.8684285e0"]
```

Only the stride 1 cases failed, so the stride 2 cases do not use the fast path. The code was put back.

Not measured on the iPhone. Not known for the iPhone: the time of `conv3x3_packed` with this change, and whether the image still matches.

Conclusion: on both Mac runtimes, loading each input value once per row cuts the packed 3x3 convolution again, by 17.8% natively and by 53.3% in Chromium. The golden tests still pass.

Next step: the user taps "Profile" on the iPhone and sends the log. The iPhone "before" is 1,610.41 ms. Then attention (change 2).

## 2026-11-18: Change 1b on the iPhone

Files changed: none. This step is one measurement that the user sent, for "Change 1b: each input value loaded once per row in the packed 3x3 convolution".

Page: `https://10.120.1.125:8443/engine/web/generate.html`, the button "Profile". Device: iPhone. The user sent the log as text. The weights came from the OPFS cache.

```
ready
copy: 47.46 GB/s median, 48.26 GB/s best (30 submits, 32 runs per submit, 90.5 ms median submit, 94.0 ms longest)
timestamp-query on this device: true
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from cache in 1.13 s
text_encoder: prepared; planned activations 2838528 bytes (2.8 MB); planned peak 1364742864 bytes (1364.7 MB); live 1364427472 bytes
text_encoder: ran in 0.23 s; measured peak 1364742864 bytes (1364.7 MB)
text_encoder: freed; live 32768 bytes
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from cache in 0.61 s
unet: prepared; planned activations 111141888 bytes (111.1 MB); planned peak 813334128 bytes (813.3 MB); live 813268592 bytes
unet: ran in 2.75 s; measured peak 813334128 bytes (813.3 MB)
unet: freed; live 32768 bytes
taesd: not planned; live 32768 bytes
taesd: ran in 0.60 s; measured peak 201507152 bytes (201.5 MB)
taesd: freed; live 32768 bytes
total: 6.15 s
image bytes: 1161 of 1048576 differ from golden/packed/taesd_image, by at most 1
timing: GPU timestamps, one compute pass per dispatch; door: 47.46 GB/s
kernel            calls   total ms    time     GB/s    door
conv3x3_packed       52     978.59   33.5%     0.70    1.5%
attention           235     838.71   28.7%     1.43    3.0%
linear_packed       192     469.73   16.1%     3.53    7.4%
conv3x3              35     410.64   14.0%     2.56    5.4%
linear              138      93.65    3.2%    13.76   29.0%
conv1x1_packed       14      49.56    1.7%     2.37    5.0%
group_norm           61      18.54    0.6%    19.47   41.0%
add                 139      12.79    0.4%    54.82  115.5%
geglu                16      10.97    0.4%    50.53  106.5%
taesd relu           21      10.73    0.4%    49.86  105.1%
taesd add_relu       10       8.14    0.3%    49.09  103.4%
layer_norm           95       7.07    0.2%    43.51   91.7%
taesd upsample        3       4.44    0.2%    24.79   52.2%
vector_packed        24       3.89    0.1%     5.04   10.6%
transpose add        16       3.00    0.1%    46.14   97.2%
concat               12       2.17    0.1%    58.01  122.2%
gelu                 23       0.48    0.0%   119.68  252.2%
upsample              3       0.29    0.0%    72.53  152.8%
taesd image           1       0.07    0.0%    84.45  177.9%
taesd clamp           1       0.01    0.0%    11.12   23.4%
all kernels        1091    2923.48  100.0%
```

Calculations, against "Change 1 on the iPhone" (one run before, one run after):

- `conv3x3_packed`: 1,610.41 ms before, 978.59 ms after. 978.59 / 1,610.41 - 1 = 39.2% less, 631.82 ms saved. Against 2,356.59 ms before change 1: 58.5% less.
- As math speed: 400.33 G operations / 0.97859 s = 409.1 G ops/s = 64.3% of the 636.29 G ops/s math rate. It was 26.7% before change 1 and 39.1% after it.
- UNet stage: 3.36 s before, 2.75 s after, 18.2% less. Against 4.15 s before change 1: 33.7% less.
- All kernels: 3,742.82 ms before, 2,923.48 ms after, 21.9% less. Against 4,429.16 ms before change 1: 34.0% less.
- The image did not change: 1,161 of 1,048,576 bytes differ, by at most 1.
- The copy median is 47.46 GB/s, 3.9% above the 45.69 GB/s of the run before. The kernels that did not change: `attention` -1.6% (851.92 to 838.71 ms), `linear_packed` -0.1% (470.31 to 469.73 ms), `conv3x3` -11.8% (465.69 to 410.64 ms), and `linear` -51.7% (194.01 to 93.65 ms). The `linear` kernel and its shader did not change. Its three iPhone values are 210.88, 194.01 and 93.65 ms, and the text encoder stage went from 0.36 s to 0.23 s. The cause of this large spread is not known and not measured. So a change of a few percent in one iPhone run is not proof of anything. The 39.2% of `conv3x3_packed` is much larger than that.
- New top three: `conv3x3_packed` 33.5%, `attention` 28.7%, `linear_packed` 16.1%. Together (978.59 + 838.71 + 469.73) / 2,923.48 = 78.2%.
- Total 6.15 s, of which 1.13 + 0.61 = 1.74 s were the cache loads. The profile adds time, so this is not the time of a plain Generate.

Conclusion: on the iPhone, loading each input value once per row cuts the packed 3x3 convolution from 1,610.41 to 978.59 ms, and the image stays the same. It now runs at 64.3% of the math rate. It is still the top row, but attention (838.71 ms) is close behind.

Next step: change 2, attention in one pass (online softmax), with timing before and after and the golden tests. The iPhone "before" is 838.71 ms.

## 2026-11-18: Benchmark page: this engine against the onnxruntime-web example

The user asked for a benchmark page that runs the onnxruntime-web example and the engine with the same prompt, and records cold time, warm time, peak memory and successes out of 10 runs, with a table to export. The user also asked for a README draft with the results and the steps to reproduce them.

Files added:

- `benchmark/index.html`, `benchmark/index.ts`: the page and its run protocol.
- `benchmark/gpu-memory.ts`: the counter of live WebGPU buffer bytes and their peak.
- `benchmark/engine-runner.ts`, `benchmark/onnx-runner.ts`, `benchmark/runners.ts`: the two runtimes behind one shape, `open()` that gives a function that makes one image.
- `benchmark/inputs.ts`, `benchmark/image.ts`, `benchmark/paths.ts`: the golden inputs, the image check, and the URLs.
- `benchmark/record.ts`, `benchmark/table.ts`: the state in `localStorage`, the summary, and the Markdown and CSV exports.
- `benchmark/results/2026-11-18-mac-chromium.md` and two PNG images: the Mac Chromium result below.
- `onnx-pipeline.ts`: the models, session options and scheduler math of the onnxruntime-web example, moved out of `index.ts`.
- `engine/web/golden.js`: `fetchBytes`, `readGolden` and `readGoldenInputs`, moved out of `generate.html`.
- `engine/web/stages.js`: `runStages`, the three stages of one image, moved out of `generate.html`.

Files changed:

- `index.ts`: imports the moved parts from `onnx-pipeline.ts`. `scaleModelInputs` and `step` take sigma as a parameter. `index.ts` passes its own 14.6146, so its behavior did not change.
- `engine/web/generate.html`: imports `golden.js` and `stages.js` in place of its own copies.
- `webpack.config.ts`: the entry `dist/benchmark`. `tsconfig.json`: the folder `benchmark`.
- `README.md`: replaced. The old file was the README of the upstream onnxruntime-web example. The new file describes this project, the results, and the steps. It keeps the link to the upstream demo and to the ONNX models.

Design choices:

- Same input: both runtimes get the 77 golden token ids of "a photo of a cat sitting on a wooden table" (`PROMPT` in `dump_golden.py`), the golden noise, sigma 14.614646911621094 and timestep 999. The onnxruntime-web example normally gets its ids from a tokenizer on a CDN and draws random noise. Golden inputs remove both differences. The ONNX latent is the golden noise times sigma, as the example does with its random noise.
- Same code: the benchmark imports the settings of the baseline from `onnx-pipeline.ts` and the engine stages from `engine/web/stages.js`. So it runs what `index.ts` and `generate.html` run, not a copy.
- One run is one fresh page. The page runs one job, saves the result, and reloads itself. Before a job, it saves a "pending" mark. If a page starts and finds the mark, the run never ended, and the page records a failure ("the page ended during this run"). This counts a tab that iOS kills.
- Run 0 of each runtime is a warm-up that fills the caches. The table leaves it out, so all counted runs load their weights from the cache.
- Cold: from the start of the run (after the golden inputs are read) to the first image on the CPU. Warm: each of the next images on the same page.
- The engine loads its weights again for every image, from OPFS, because each stage frees its file. This is the engine's normal path, so its warm time includes the loads.
- Peak memory: the page wraps `GPUDevice.createBuffer` and `GPUBuffer.destroy` before any device exists. A `FinalizationRegistry` stops counting a buffer that the garbage collector frees without `destroy`. This counts both runtimes in the same way, without help from either. JavaScript and WASM memory are not in the number.
- Success: no error, and the image is 512 x 512 and not one flat color. Each run has a time limit of 15 minutes.
- The table exports as Markdown (also shown on the page, to copy on the phone) and as CSV.

Finding while building the counter: in wgpu 30, `impl Drop for WebBuffer` in `src/backend/webgpu.rs` is a no-op. So when the engine drops a buffer in the browser, wgpu does not call `destroy()`. The browser frees the buffer only after the garbage collector frees its JavaScript wrapper. The comment in `engine/src/memory/mod.rs` says "in the browser, wgpu calls destroy()". That comment is wrong. The engine's counter (`live 0 bytes` after each stage) counts only its own books. This step does not change the engine.

Checks after the refactor:

- `npm run typecheck` and `npm run build` passed. The first `npm run build` of this step, before any change, wrote `3f8fcd97177ed35d4f32.mjs` and `ae86c793c46ed060a6f3.wasm` in the project folder. `dist/index.js`, `dist/index.min.js` and `dist/benchmark.js` refer to them. They are output of the existing build with onnxruntime-web 1.31.1, not of this change.
- Generate on `http://127.0.0.1:8765/engine/web/generate.html`, Mac Chromium, after the move of the stage code:

```
text_encoder: ran in 0.20 s; measured peak 1364710096 bytes (1364.7 MB)
unet: ran in 1.73 s; measured peak 813301360 bytes (813.3 MB)
taesd: ran in 0.40 s; measured peak 201474352 bytes (201.5 MB)
total: 4.53 s
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
```

Page: `http://127.0.0.1:8765/benchmark/index.html`, served by `python3 -m http.server 8765 --bind 127.0.0.1` in the project folder. Device: Mac Chromium. A Node script with `playwright-core@1.56.1` (in the scratchpad, not in the project) opened Chrome for Testing from `~/Library/Caches/ms-playwright/chromium-1254` with `--enable-unsafe-webgpu` and the persistent profile of "Time per kernel". It pressed Reset, set the plan, pressed Start, and read the log every 5 s through the reloads.

Smoke test, engine only, 1 run, 1 warm image:

```
| this engine (Rust + WGSL) | 1 of 1 | 3.37 | 3.41 | 2179.3 |
| engine | 0 (warm-up) | yes | 3.62 | 3.45 | 2002.8 |  |
| engine | 1 | yes | 3.37 | 3.41 | 2179.3 |  |
```

Failed script run: the first onnxruntime-web smoke test stopped after 5 s. The log of the page still had "done." from the engine plan, so the script ended at once and closed the browser during run 0. The next page found the pending mark and printed `onnx run 0: FAILED: the page ended during the run`. This is the path that a killed tab takes, so it worked once by accident. The script now looks for "done." on the last log line only.

Smoke test, onnxruntime-web only, 1 run, 1 warm image. Run 0 downloaded the models from Hugging Face:

```
| onnxruntime-web example | 1 of 1 | 10.19 | 1.77 | 5545.7 |
| onnx | 0 (warm-up) | yes | 511.70 | 3.32 | 5545.7 |  |
| onnx | 1 | yes | 10.19 | 1.77 | 5545.7 |  |
```

Full plan: both runtimes, 10 runs, 3 warm images per run. The whole report, as the page exported it:

```
Browser: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36
Plan: 10 runs per runtime, 3 warm images per run

| runtime | successes | cold s (median) | warm s (median) | peak GPU buffers MB (max) |
| --- | --- | --- | --- | --- |
| this engine (Rust + WGSL) | 9 of 10 | 3.30 | 3.29 | 4134.6 |
| onnxruntime-web example | 10 of 10 | 7.73 | 1.58 | 5545.7 |

| runtime | run | ok | cold s | warm s | peak MB | error |
| --- | --- | --- | --- | --- | --- | --- |
| engine | 0 (warm-up) | yes | 3.64 | 3.27 3.35 3.22 | 2112.1 |  |
| engine | 1 | no | 3.30 | 3.34 3.27 | 3711.9 | text embeddings: one submit took 292.7 ms, the limit is 250.0 ms |
| engine | 2 | yes | 3.29 | 3.27 3.28 3.30 | 2179.3 |  |
| engine | 3 | yes | 3.32 | 3.32 3.28 3.27 | 2037.5 |  |
| engine | 4 | yes | 3.34 | 3.28 3.24 3.33 | 3779.1 |  |
| engine | 5 | yes | 3.28 | 3.32 3.31 3.32 | 2770.2 |  |
| engine | 6 | yes | 3.29 | 3.30 3.28 3.29 | 2770.2 |  |
| engine | 7 | yes | 3.29 | 3.29 3.30 3.31 | 2770.2 |  |
| engine | 8 | yes | 3.30 | 3.36 3.29 3.27 | 3644.7 |  |
| engine | 9 | yes | 3.33 | 3.32 3.33 3.26 | 4134.6 |  |
| engine | 10 | yes | 3.32 | 3.29 3.27 3.30 | 4115.0 |  |
| onnx | 0 (warm-up) | yes | 9.43 | 2.54 1.59 1.60 | 5545.7 |  |
| onnx | 1 | yes | 9.45 | 2.60 1.56 1.55 | 5545.7 |  |
| onnx | 2 | yes | 8.39 | 2.44 1.55 1.56 | 5545.7 |  |
| onnx | 3 | yes | 6.78 | 1.99 1.59 1.55 | 5545.7 |  |
| onnx | 4 | yes | 7.71 | 2.91 1.57 1.56 | 5545.7 |  |
| onnx | 5 | yes | 9.40 | 2.34 1.57 1.57 | 5545.7 |  |
| onnx | 6 | yes | 8.29 | 2.58 1.57 1.57 | 5545.7 |  |
| onnx | 7 | yes | 7.75 | 2.09 1.59 1.58 | 5545.7 |  |
| onnx | 8 | yes | 7.34 | 2.42 1.57 1.58 | 5545.7 |  |
| onnx | 9 | yes | 7.54 | 2.11 1.74 1.55 | 5545.7 |  |
| onnx | 10 | yes | 7.40 | 2.14 1.59 1.57 | 5545.7 |  |
```

The console of the full plan had these lines (counted by `uniq -c`, timestamps removed). The onnxruntime-web lines are warnings that it prints at session creation:

```
  22 console error: [0;93m[W:onnxruntime:, constant_folding.cc:588 ApplyImpl] Could not find a CPU kernel and hence can't constant fold Sqrt node '/decoder/mid_block/attentions.0/Sqrt'[m
  22 console error: [0;93m[W:onnxruntime:, session_state.cc:1598 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.[m
  22 console error: [0;93m[W:onnxruntime:, session_state.cc:1600 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.[m
   1 console error: Failed to load resource: the server responded with a status of(File not found)
```

The script saved the canvas after the first warm image of run 1 of each runtime. Both images show the same cat in the same pose, on a wooden table. The engine image is sharper, because TAESD decodes it. The onnxruntime-web image comes from the full VAE decoder.

Calculations:

- Cold, median of the successful counted runs: engine 3.30 s, onnxruntime-web 7.73 s. 3.30 / 7.73 - 1 = 57.3% less time for the engine. 7.73 / 3.30 = 2.34.
- Warm, median of all warm images of the successful runs: engine 3.29 s, onnxruntime-web 1.58 s. 3.29 / 1.58 = 2.08. The engine loads its weights from OPFS for each image. "Time per kernel" measured these loads at 1.08 s (text encoder) and 0.55 s (UNet) in another run.
- The first warm image of onnxruntime-web is slower than the next two in every run (1.99 to 2.91 s, against 1.55 to 1.74 s).
- Successes: engine 9 of 10, onnxruntime-web 10 of 10. The engine run 1 failed in its third warm image: `text embeddings: one submit took 292.7 ms, the limit is 250.0 ms`. That is the engine's own stop at 250 ms. Its cause is not measured.
- Peak of live GPU buffers: onnxruntime-web 5545.7 MB in every run. Engine from 2037.5 MB (run 3) to 4134.6 MB (run 9). The engine's own counter gives at most 1364.7 MB per stage. The difference is buffers that the engine dropped but the browser had not freed yet (the wgpu finding above). It depends on when the garbage collector runs, so it changes from run to run.
- Both peaks are above the 2048 MB per-tab limit of the iPhone (`crash-record/2026-11-15-iphone16promax.md`). For the engine, this shows a risk on the iPhone: the text encoder's 1,361,551,360 bytes can still be live when the UNet's 699,997,456 bytes load. The sum is 2,061,548,816 bytes.

Not measured on the iPhone. Not known for the iPhone: every number of the table, whether the engine's dropped buffers are freed in time in Safari, and how long the onnxruntime-web runs take before iOS stops them.

Conclusion: the page runs both runtimes with the same prompt and noise, ten fresh pages each, and exports the table. In Mac Chromium, the engine makes the first image in 3.30 s against 7.73 s, and onnxruntime-web makes the next images in 1.58 s against 3.29 s. The page found that the engine's dropped GPU buffers stay in the browser until the garbage collector runs.

Next step: the user runs the page on the iPhone (`https://10.120.1.125:8443/benchmark/index.html`, both runtimes, 10 runs, 3 warm images) and sends the Markdown export. A later step can make `GpuBuffer` call `destroy()` when it drops, and fix the comment in `engine/src/memory/mod.rs`.

## 2026-11-18: Benchmark page on the iPhone

Files changed:

- `benchmark/results/2026-11-18-iphone.md`: the Markdown export that the user sent, copied unchanged.
- `README.md`: the iPhone section now has this result in place of "not measured".

Page: `https://10.120.1.125:8443/benchmark/index.html`, both runtimes, 10 runs, 3 warm images per run. Device: iPhone. The user sent the file `sd-turbo-benchmark.md` that the page exported, and wrote: "OMNX always killed". Earlier, the user also reported that the page reloads after "engine 1 warm 3". That reload is the end of a run, as designed. The export:

```
Browser: Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.6.2 Mobile/15E148 Safari/604.1
Plan: 10 runs per runtime, 3 warm images per run

| runtime | successes | cold s (median) | warm s (median) | peak GPU buffers MB (max) |
| --- | --- | --- | --- | --- |
| this engine (Rust + WGSL) | 0 of 10 | - | - | 1608.2 |
| onnxruntime-web example | 0 of 2 | - | - | 0.0 |

| runtime | run | ok | cold s | warm s | peak MB | error |
| --- | --- | --- | --- | --- | --- | --- |
| engine | 0 (warm-up) | yes | 5.79 | 5.04 5.44 5.38 | 1608.2 |  |
| engine | 1 | no | 6.67 |  | 1450.9 | up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention: one submit took 265.0 ms, the limit is 250.0 ms |
| engine | 2 | no | - |  | 1412.1 | up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention: one submit took 268.0 ms, the limit is 250.0 ms |
| engine | 3 | no | 6.47 |  | 1376.5 | up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention: one submit took 251.0 ms, the limit is 250.0 ms |
| engine | 4 | no | 6.11 | 7.82 | 1608.2 | up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention: one submit took 252.0 ms, the limit is 250.0 ms |
| engine | 5 | no | 5.94 |  | 1412.1 | up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention: one submit took 253.0 ms, the limit is 250.0 ms |
| engine | 6 | no | 6.07 |  | 1446.4 | up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention: one submit took 250.0 ms, the limit is 250.0 ms |
| engine | 7 | no | 6.09 |  | 1452.5 | up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention: one submit took 254.0 ms, the limit is 250.0 ms |
| engine | 8 | no | 6.20 |  | 1412.1 | up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention: one submit took 271.0 ms, the limit is 250.0 ms |
| engine | 9 | no | 6.41 |  | 1412.1 | up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention: one submit took 298.0 ms, the limit is 250.0 ms |
| engine | 10 | no | 6.34 |  | 1412.1 | up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention: one submit took 258.0 ms, the limit is 250.0 ms |
| onnx | 0 (warm-up) | no | - |  | - | the page ended during this run: the tab was killed or reloaded |
| onnx | 1 | no | - |  | - | the page ended during this run: the tab was killed or reloaded |
| onnx | 2 | no | - |  | 0.0 | Load failed |
```

Calculations:

- The browser string says `iPhone OS 18_7` and `Version/26.6.2`. The crash record names iOS 26.7.1 and Safari 26.6.2 on this phone. The cause of the different iOS number is not checked.
- Engine, counted runs: 0 of 10. All 10 failures are the same: the step `up_blocks.3.attentions.0.transformer_blocks.0.attn1 attention` broke the 250 ms limit. Its times: 250, 251, 252, 253, 254, 258, 265, 268, 271 and 298 ms. Median (254 + 258) / 2 = 256 ms. The longest, 298 ms, is 298 / 500 = 59.6% of the time where iOS loses the device.
- This step holds only the self attention kernel at 4,096 tokens (`engine/src/unet/transformer.rs`), all its chunks in one submit. "Attention on the iPhone" measured 145 ms per run of this attention.
- Where the runs stopped: run 2 in its first image (no cold time). Run 4 in its second warm image (one warm time, 7.82 s). Every other run in its first warm image.
- The same step passed in every earlier iPhone run of this log, and in the warm-up run here (4 images). The counted runs come after it, with the GPU busy for minutes without a break. A hotter, slower GPU is one possible cause of the creep from about 145 ms to 250 ms or more. This is not measured.
- Cold images that finished, in failed runs 1 and 3 to 10: 5.94, 6.07, 6.09, 6.11, 6.20, 6.34, 6.41, 6.47 and 6.67 s. Median 6.20 s. The table shows "-", because it counts only successful runs.
- Warm-up run 0 of the engine, which passed: cold 5.79 s, warm 5.04, 5.44 and 5.38 s.
- Engine peak of live GPU buffers: 1,376.5 to 1,608.2 MB. 1,608.2 / 2,048 = 78.5% of the per-tab limit. The engine's own counter gives at most 1,364.7 MB per stage, so 1,608.2 - 1,364.7 = 243.5 MB more were live in Safari. In Mac Chromium, the peak was 2,037.5 to 4,134.6 MB. So Safari freed the dropped buffers much sooner than Chromium did in that run.
- onnxruntime-web: run 0 (warm-up) and run 1 ended with the tab killed or reloaded. Run 2 failed with "Load failed", the message that Safari gives for a failed `fetch`. Its peak 0.0 MB means that no GPU buffer existed yet. The export has no runs after 2, so the plan stopped there. The user did not say whether Stop was pressed.

Not known for the iPhone: the cause of the slow attention submits, and the warm time of the engine with all 3 warm images in counted runs.

Conclusion: on the iPhone, onnxruntime-web made no image, as in the crash record. The engine stayed inside the memory limit, but its self attention step at 4,096 tokens is too close to the 250 ms limit to survive ten runs in a row. That limit is the engine's own safety stop at half the device-loss time, not a crash.

Next step: split the self attention step into submits of fewer chunks, so each submit has room under 250 ms. Then run the benchmark again on the iPhone, engine only.

## 2026-11-18: Self attention in submits of 1,024 query rows

The fix for "Benchmark page on the iPhone": there, the self attention step at 4,096 tokens took 250 to 298 ms in one submit, past the 250 ms limit, in all 10 counted engine runs.

Files changed:

- `engine/src/attention/slots.rs`: the new `bind_attention_groups` makes the same chunks as before, but puts them into groups of at most `rows_per_group` query rows. Each group is one `BoundKernel` with its dispatches and their sizes buffers. `bind_attention_into` is now the case with one group.
- `engine/src/attention/mod.rs`: exports `bind_attention_groups`.
- `engine/src/unet/transformer.rs`: `SELF_ATTENTION_ROWS_PER_SUBMIT = 1024`. The self attention gets one step per group, named for example `...attn1 attention 2 of 4`. The new helper `bind_on_attention_slots` holds the five tensor names of an attention, which `attention` and `self_attention_groups` both use.
- `benchmark/results/2026-11-18-mac-chromium-split-attention.md`: the Mac Chromium benchmark below.

Design choices:

- The kernel, the chunks (256 rows, from `unet/plan/transformer.rs`), the sizes buffers and the bind groups do not change. Only the steps change: the chunks of one self attention run in several submits in place of one. So the math is the same, and the plan and the bind group count are the same.
- 1,024 rows per submit: 4 submits at 64 x 64 (4,096 rows), 1 submit at 32 x 32 (1,024 rows) and below. At the worst time of the iPhone benchmark, 298 ms for 4,096 rows, 1,024 rows take about 298 / 4 = 74.5 ms, if the time is even over the rows.
- Cross attention (77 keys) and the text encoder attention (77 rows) keep one submit.
- Each new submit adds one wait for onSubmittedWorkDone. At 64 x 64 there are 5 self attentions, so 5 x 3 = 15 more submits per image.

Commands, in `engine/`. Device: Mac native.

`cargo test --release --test unet_packed -- --nocapture`, before the change (the mean error and worst position are left out of each line here):

```
after prepare: 1146 bind groups, 2232 buffers, weights 702072352 bytes (2074896 added by prepare)
down_block_0: largest error 2.4318695068359375e-5, 19 submits, longest 68.4 ms, peak 818478704 bytes
up_block_3: largest error 2.751350402832031e-4, 30 submits, longest 67.5 ms, peak 823721584 bytes
```

After the change:

```
after prepare: 1146 bind groups, 2232 buffers, weights 702072352 bytes (2074896 added by prepare)
down_block_0: largest error 2.4318695068359375e-5, 25 submits, longest 23.8 ms, peak 818478704 bytes
down_block_1: largest error 1.7261505126953125e-4, 19 submits, longest 10.4 ms, peak 818478704 bytes
down_block_2: largest error 1.773834228515625e-4, 19 submits, longest 10.6 ms, peak 818478704 bytes
down_block_3: largest error 1.6117095947265625e-4, 4 submits, longest 7.2 ms, peak 818478704 bytes
mid_block: largest error 1.583099365234375e-4, 11 submits, longest 7.2 ms, peak 818478704 bytes
up_block_0: largest error 1.125335693359375e-4, 10 submits, longest 14.0 ms, peak 818478704 bytes
up_block_1: largest error 5.5694580078125e-4, 31 submits, longest 38.3 ms, peak 818478704 bytes
up_block_2: largest error 5.106925964355469e-4, 31 submits, longest 36.9 ms, peak 823721584 bytes
up_block_3: largest error 2.751350402832031e-4, 39 submits, longest 30.7 ms, peak 823721584 bytes
```

The largest errors of all nine blocks are the same numbers as before the change. The other blocks had the same submit counts before.

`cargo test --release` passed all 12 test files, 55 tests. `cargo test --release --test pipeline -- --nocapture --test-threads 1` printed `image bytes: 547 of 1048576 differ, by at most 1` (float32) and `image bytes: 1140 of 1048576 differ, by at most 1` (packed), the same as before. `cargo clippy --release --all-targets` gave no new warning. `wasm-pack build --release --target web --out-dir web/pkg --no-typescript` passed.

Page: `http://127.0.0.1:8765/benchmark/index.html`, engine only, 10 runs, 3 warm images. Device: Mac Chromium, with the script and profile of "Benchmark page: this engine against the onnxruntime-web example":

```
Browser: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36
Plan: 10 runs per runtime, 3 warm images per run

| runtime | successes | cold s (median) | warm s (median) | peak GPU buffers MB (max) |
| --- | --- | --- | --- | --- |
| this engine (Rust + WGSL) | 10 of 10 | 3.33 | 3.32 | 4134.6 |

| runtime | run | ok | cold s | warm s | peak MB | error |
| --- | --- | --- | --- | --- | --- | --- |
| engine | 0 (warm-up) | yes | 3.96 | 3.36 3.32 3.41 | 3140.9 |  |
| engine | 1 | yes | 3.32 | 3.30 3.45 3.28 | 3342.4 |  |
| engine | 2 | yes | 3.33 | 3.31 3.31 3.29 | 2770.2 |  |
| engine | 3 | yes | 3.34 | 3.93 3.28 3.35 | 3846.3 |  |
| engine | 4 | yes | 3.30 | 3.31 3.31 3.29 | 2770.2 |  |
| engine | 5 | yes | 3.32 | 3.29 3.28 3.28 | 2770.2 |  |
| engine | 6 | yes | 3.33 | 3.29 3.44 3.29 | 3565.0 |  |
| engine | 7 | yes | 3.33 | 4.17 3.53 3.35 | 4134.6 |  |
| engine | 8 | yes | 3.35 | 3.33 3.44 3.97 | 3825.3 |  |
| engine | 9 | yes | 3.70 | 4.33 3.85 3.52 | 4131.8 |  |
| engine | 10 | yes | 3.34 | 3.30 3.54 3.34 | 3376.0 |  |
```

Calculations:

- `down_block_0`: 19 + 2 x 3 = 25 submits, as expected for 2 transformers at 64 x 64. Longest submit 68.4 to 23.8 ms. `up_block_3`: 30 + 3 x 3 = 39 submits. Longest submit 67.5 to 30.7 ms. Its longest submit is now a step other than the attention.
- Mac Chromium, engine: 10 of 10, against 9 of 10 in the first benchmark. The failure there was another step (`text embeddings`, 292.7 ms). It did not come back in this run, but one run does not prove that it is gone.
- Cold median 3.33 s against 3.30 s, warm median 3.32 s against 3.29 s: 0.03 s more, which is within the spread of single runs (cold 3.30 to 3.70 s here). The 15 extra submits cost little on the Mac.
- The peak of live GPU buffers is 2,770.2 to 4,134.6 MB, as before. The change does not touch memory.

Mutation check: `bind_attention_groups` was changed to drop the first full group of query rows. `cargo test --release --test unet_packed -- --nocapture` failed:

```
down_block_0: largest error 1.165020203590393e1, mean error 5.430525296915999e-1, worst at [0, 181, 2, 13] (engine 11.669307, golden 0.01910472), 23 submits, longest 27.4 ms, peak 818477424 bytes
down_block_0: first mismatch, largest error 1.165020203590393e1 is above 1e-4
```

The first try of this check used a wrong condition and failed with a panic before it printed a block line, so it proved nothing. The second try above shows the missing rows. The code was put back.

Not measured on the iPhone. Not known for the iPhone: whether 1,024 rows per submit keep the self attention under 250 ms in ten runs in a row, and the new success count.

Conclusion: the self attention at 4,096 tokens now runs in 4 submits, with the same numbers. On the Mac native, its longest submit fell from 68.4 to 23.8 ms. The engine made 10 of 10 runs in Mac Chromium.

Next step: the user runs `https://10.120.1.125:8443/benchmark/index.html` on the iPhone, engine only, 10 runs, 3 warm images, and sends the Markdown export.

## 2026-11-18: Benchmark on the iPhone after the attention split

Files changed:

- `benchmark/results/2026-11-18-iphone-split-attention.md`: the Markdown export that the user sent, copied unchanged.
- `README.md`: the iPhone section has the engine row of this run. The onnxruntime-web row stays from "Benchmark page on the iPhone".

Page: `https://10.120.1.125:8443/benchmark/index.html`, engine only, 10 runs, 3 warm images per run. Device: iPhone. The user sent the file `sd-turbo-benchmark 2.md` that the page exported:

```
Browser: Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.6.2 Mobile/15E148 Safari/604.1
Plan: 10 runs per runtime, 3 warm images per run

| runtime | successes | cold s (median) | warm s (median) | peak GPU buffers MB (max) |
| --- | --- | --- | --- | --- |
| this engine (Rust + WGSL) | 10 of 10 | 5.36 | 5.64 | 1608.2 |

| runtime | run | ok | cold s | warm s | peak MB | error |
| --- | --- | --- | --- | --- | --- | --- |
| engine | 0 (warm-up) | no | - |  | 1364.4 | no GPU adapter: No suitable graphics adapter found; noop support not compiled in, vulkan support not compiled in, metal support not compiled in, dx12 support not compiled in, gl support not compiled in, webgpu found no adapters |
| engine | 1 | yes | 5.92 | 5.05 4.94 5.15 | 1450.9 |  |
| engine | 2 | yes | 5.06 | 5.18 5.35 5.38 | 1450.9 |  |
| engine | 3 | yes | 5.43 | 5.50 5.51 5.51 | 1608.2 |  |
| engine | 4 | yes | 5.52 | 5.38 5.47 5.67 | 1608.2 |  |
| engine | 5 | yes | 5.36 | 5.49 5.57 5.61 | 1608.2 |  |
| engine | 6 | yes | 5.42 | 5.51 5.82 6.64 | 1452.5 |  |
| engine | 7 | yes | 5.25 | 6.42 6.64 6.70 | 1608.2 |  |
| engine | 8 | yes | 5.22 | 6.71 6.70 7.16 | 1608.2 |  |
| engine | 9 | yes | 5.36 | 6.70 6.69 6.64 | 1608.2 |  |
| engine | 10 | yes | 5.24 | 6.56 6.61 6.61 | 1608.2 |  |
```

Calculations:

- Successes: 10 of 10, against 0 of 10 before the split ("Benchmark page on the iPhone"). No counted run stopped at the 250 ms limit.
- Cold, median of 10 runs: 5.36 s (5.06 to 5.92 s). Warm, median of 30 images: 5.64 s.
- Warm images of runs 1 to 5: median 5.47 s. Runs 6 to 10: median 6.64 s. 6.64 / 5.47 - 1 = 21.4% slower. The cold images did not get slower over the runs (5.22 to 5.43 s in runs 6 to 10). The page waits 3 s and loads a new page before each cold image, so the GPU gets a short pause. A hotter GPU is one possible cause of the slower warm images. This is not measured.
- Against Mac Chromium after the split ("Self attention in submits of 1,024 query rows"): cold 5.36 / 3.33 = 1.61, warm 5.64 / 3.32 = 1.70.
- Peak of live GPU buffers: 1,450.9 to 1,608.2 MB. 1,608.2 / 2,048 = 78.5% of the per-tab limit, the same peak as before the split.
- The warm-up run 0 failed with `no GPU adapter: ... webgpu found no adapters`, with a peak of 1,364.4 MB. 1,364.4 MB is the live size of the text stage without a profile (1,364,394,704 bytes, "WASM entry for the three stages, and the generate page"). So in that page, an engine had already opened the GPU and loaded the text encoder, and the error came from a second `Engine.open` in the same page. The page can do this: Start calls `continuePlan`, and `nextJob` does not look at the `pending` mark. So a press on Start while a run is going schedules the same run a second time. The user did not say whether Start was pressed twice. This is the likely cause, not a proven one. Run 0 is the warm-up, so the counted runs are not affected.

Conclusion: with self attention in 4 submits, the engine made 10 of 10 runs on the iPhone, 4 images each, inside the memory limit. On the iPhone, the engine is the only one of the two runtimes that makes an image.

Next step: make Start do nothing while a run is going (check the `pending` mark), so one page never runs two jobs.

## 2026-11-18: Prompt tokenizer in the engine

Before this step, the engine had no tokenizer. The 77 token ids came from `golden/float32/text_input_ids.safetensors`, so the engine could draw only the golden prompt.

Files added:

- `export_tokenizer.py`: copies `vocab.json` and `merges.txt` of `stabilityai/sd-turbo` (subfolder `tokenizer`) to `weights/tokenizer_vocab.json` and `weights/tokenizer_merges.txt`.
- `engine/testdata/tokenizer/make_cases.py` and `cases.json`: 27 prompts and the 77 ids that transformers gives for each, with the call of StableDiffusionPipeline (`padding="max_length", max_length=77, truncation=True`).
- `engine/src/tokenizer/mod.rs`: `Tokenizer::new(vocab, merges)` and `Tokenizer::encode(prompt)`, which gives 77 ids.
- `engine/src/tokenizer/normalize.rs`: NFC, one space for each run of whitespace, lowercase.
- `engine/src/tokenizer/special.rs`: `split_specials` cuts the text at special tokens, longest match first.
- `engine/src/tokenizer/bpe.rs`: the GPT-2 byte alphabet and the byte pair merges.
- `engine/tests/tokenizer.rs`: compares the engine with all cases of `cases.json`.
- `benchmark/results/2026-11-18-mac-chromium-prompt-car.png`: the image of the new prompt below.

Files changed:

- `engine/Cargo.toml`: `regex` (only the features `std`, `unicode-gencat` and `unicode-perl`) and `unicode-normalization`.
- `engine/src/error.rs`: `TokenizerFiles`. `engine/src/lib.rs`: exports `Tokenizer` and `CONTEXT_IDS`.
- `engine/src/web.rs`: the class `PromptTokenizer` for JavaScript.
- `engine/web/generate.html`: a prompt box. Generate, "Make 20" and Profile tokenize the prompt in the box. The page prints the ids and says whether they are the golden ids. Only the golden ids have a golden image, so for another prompt the page prints "no golden image for this prompt, so no byte compare". The noise, sigma and timestep stay the golden ones.
- `README.md`: `uv run export_tokenizer.py` in step 1.

Design choices:

- The reference is the tokenizer that transformers 5.23.0 loads for `stabilityai/sd-turbo` (`CLIPTokenizer`, built on the tokenizers library). Its source in the uv cache gives the rules: the normalizer is NFC, then `\s+` to one space, then lowercase. The pre-tokenizer keeps the matches of `'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+` (and the two special tokens), then maps each byte to the byte-level alphabet. BPE adds `</w>` to the last symbol of each piece. The post-processor adds `<|startoftext|>` (49406) and `<|endoftext|>` (49407).
- The pad token "!" (id 0) is a special token with `normalized: false`. The tokenizers library finds such tokens in the raw text before the normalizer. So every "!" in a prompt becomes id 0, not a BPE piece. The engine does the same. `<|startoftext|>` and `<|endoftext|>` have `normalized: true`, so the engine finds them after the normalizer, also in capitals.
- The ByteLevel pre-tokenizer of the reference splits again with the GPT-2 pattern. That split changes nothing on the CLIP pieces, because a CLIP piece has no whitespace and a punctuation run has no letters. So the engine leaves it out. The cases check this with contractions and punctuation.
- BPE merges the pair with the lowest rank, the leftmost one on a tie, one pair at a time. The ranks are keyed by the merge line itself ("left right"), because no symbol holds a space (the space byte maps to "Ġ").
- A symbol that is not in the vocabulary gets the unknown id 49407, the `unk_token` of the reference.
- The tokenizer files load at run time from `weights/`, not inside the .wasm file. The page reads them once at start.

Commands, in `engine/` unless named. Device: Mac, Python through `uv run`, and Mac native for the Rust tests.

`uv run export_tokenizer.py`, in the project folder:

```
tokenizer_vocab.json: 1059962 bytes
tokenizer_merges.txt: 524619 bytes
```

`uv run make_cases.py`, in `engine/testdata/tokenizer/`, the summary line of the final run (27 prompts). The first run had 26 prompts, without the car prompt below:

```
'A red sports car on a mountain road at s': [49406, 320, 736, 2054, 1615, 525, 320, 3965, 1759, 536, 3424, 0, 49407, 0]
27 cases, transformers 5.23.0
```

The first case, the golden prompt, gives `[49406, 320, 1125, 539, 320, 2368, 4919, 525, 320, 9057, 2175, 49407, 0, ...]`, the same ids as `golden/float32/text_input_ids.safetensors`.

`cargo test --release --test tokenizer -- --nocapture`:

```
tokenizer: 27 of 27 prompts match transformers 5.23.0, 77 ids each
```

The 26-prompt run before had printed `tokenizer: 26 of 26 prompts match transformers 5.23.0, 77 ids each` on the first try. `cargo test --release --lib tokenizer` passed the 4 new unit tests. `cargo test --release` passed all 13 test files, 60 tests. `cargo clippy --release --all-targets` and the WASM clippy gave no new warning. `wasm-pack build --release --target web --out-dir web/pkg --no-typescript` passed.

Size of `web/pkg/engine_bg.wasm`: 700,010 bytes before, 1,272,976 bytes after. 1,272,976 - 700,010 = 572,966 bytes more, for the Unicode tables of `regex` and `unicode-normalization` and the tokenizer code.

Page: `http://127.0.0.1:8765/engine/web/generate.html`, Generate with the prompt in the box. Device: Mac Chromium, a Node script with `playwright-core@1.56.1` and the persistent profile of the earlier steps. Golden prompt:

```
ready
prompt "a photo of a cat sitting on a wooden table": ids [49406, 320, 1125, 539, 320, 2368, 4919, 525, 320, 9057, 2175, 49407]; the same as the golden ids
text_encoder: ran in 0.23 s; measured peak 1364710096 bytes (1364.7 MB)
unet: ran in 1.83 s; measured peak 813301360 bytes (813.3 MB)
taesd: ran in 0.46 s; measured peak 201474352 bytes (201.5 MB)
total: 4.89 s
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
```

New prompt:

```
ready
prompt "A red sports car on a mountain road at sunset!": ids [49406, 320, 736, 2054, 1615, 525, 320, 3965, 1759, 536, 3424, 0, 49407]; not the golden ids
text_encoder: ran in 0.25 s; measured peak 1364710096 bytes (1364.7 MB)
unet: ran in 1.81 s; measured peak 813301360 bytes (813.3 MB)
taesd: ran in 0.42 s; measured peak 201474352 bytes (201.5 MB)
total: 3.76 s
no golden image for this prompt, so no byte compare
```

The ids of the new prompt are the same as the transformers ids of the 27-prompt run above. The image shows a red sports car from behind, on a mountain road, in orange evening light. The "!" became id 0, as in transformers.

Mutation checks:

- The "!" pre-split was removed (`split_specials(prompt, &[])`). The test printed `tokenizer: 23 of 26 prompts match transformers 5.23.0, 77 ids each`, with the failures `"hello!"`, `"wow!!! amazing!"` and `"a!b"`. The code was put back.
- NFC was removed from `normalize` (the text went on unchanged). The test printed `tokenizer: 25 of 26 prompts match transformers 5.23.0, 77 ids each`, with the failure `"cafe\u{301} with a decomposed accent"`. The code was put back.

Not measured on the iPhone. Not known for the iPhone: the load time of the two tokenizer files (1,584,581 bytes) and of the larger .wasm file, and whether the page works with a typed prompt.

Conclusion: the engine turns any prompt into the same 77 ids as transformers, for all 27 test prompts, and the generate page draws the typed prompt. The golden prompt still gives the golden ids and the same image bytes.

Next step: the user opens `https://10.120.1.125:8443/engine/web/generate.html` on the iPhone, types a prompt, taps Generate, and sends the log and a screenshot.

## 2026-11-18: Where the quality of a new prompt is lost

The user tried "dog" on the iPhone, which looked fine, then "a photo of girl with watermelon", which looked poor. The user asked whether the base model is also poor for that prompt.

Files added:

- `compare_prompt.py`: makes the image of one prompt three ways in PyTorch, with the golden noise of the engine. `1_original.png` is SD-Turbo as published (float32 UNet, full VAE decoder). `2_taesd.png` is the same float32 UNet with the TAESD decoder. `3_engine_math.png` is the packed UNet of `weights/unet_packed.bin` with TAESD, which is what the engine computes.
- `benchmark/results/2026-11-18-girl-watermelon-grid.png`: the four images below in one grid.

Files changed:

- `pack_weights.py`: `read_packed_file`, `entry_values` and `INDEX_BITS`, moved out of `dump_packed_golden.py`, so `compare_prompt.py` can read the packed file too. It returns numpy arrays, so the format module still needs only numpy.
- `dump_packed_golden.py`: `packed_state` now calls `read_packed_file` and makes torch tensors of its arrays.

Check of the move: `uv run dump_packed_golden.py`, in the project folder. Device: Mac, Python through `uv run`, CPU. `md5 -q golden/packed/*.safetensors | md5 -q` printed `e5f46b621cb5214ff07b7a8dc72053b2` before and after, so all 16 files of the packed golden set are the same bytes. The script printed the same lines as before, for example `unet_output: packed vs float32 weights largest difference 1.259e-01`.

Command: `uv run compare_prompt.py "a photo of girl with watermelon" --out <scratchpad>/girl`, in the project folder. Device: Mac, Python through `uv run`, CPU:

```
ids: [49406, 320, 1125, 539, 1611, 593, 19889, 49407]
```

Page: Generate on `http://127.0.0.1:8765/engine/web/generate.html` with the same prompt. Device: Mac Chromium, the script of "Prompt tokenizer in the engine":

```
prompt "a photo of girl with watermelon": ids [49406, 320, 1125, 539, 1611, 593, 19889, 49407]; not the golden ids
total: 4.03 s
no golden image for this prompt, so no byte compare
```

The ids of the engine and of the pipeline are the same.

Failed first compare: the script first saved the engine image as a screenshot of the canvas. The screenshot was 512 x 513 pixels, so the compare resized it, and printed `3_engine_math vs 4_engine: mean absolute byte difference 5.11, largest 243`. The script now saves the canvas pixels with `toDataURL`, which gives 512 x 512. A second Generate (`total: 3.75 s`) gave the image of the compare below.

A small script in the scratchpad put the four images in a grid and compared them:

```
1_original vs 2_taesd: mean absolute byte difference 10.56, largest 144
2_taesd vs 3_engine_math: mean absolute byte difference 11.96, largest 236
3_engine_math vs 4_engine: mean absolute byte difference 0.00, largest 1
```

What the grid shows:

- `1_original`: a smiling girl in a green dress holds a watermelon slice. More slices float in the air around her, and her fingers under the slice are stiff. These faults are in the published model, with one step at 512 x 512.
- `2_taesd`: almost the same image, with a little more contrast in the eyes and the skin.
- `3_engine_math` and `4_engine`: the same layout, but the face is worse. The eyes are distorted, and the grin is wider, with odd teeth. The hands are a little worse.
- `4_engine` differs from `3_engine_math` by at most 1 in each byte. So the engine computes the packed model correctly, and the loss comes from the packed weights.

Calculations, from `quant_eval.txt` of "Mixed 4-bit and 8-bit plan, and a 20-image test": the two prompts with people have the largest LPIPS of the mixed plan, `a portrait of an old fisherman with a grey beard, natural light` (0.1594, the worst of 20) and `a young woman reading a book in a sunny cafe` (0.0897). The mean is 0.0654. So faces lose the most to the 54 layers at 4 bits. The mixed plan holds 699,997,456 bytes.

Not measured on the iPhone. The user saw the iPhone image, but did not send it. The engine on the iPhone gave the golden cat within 1 byte in earlier steps, so the iPhone image is very likely the same as `4_engine`. This is not checked.

Conclusion: for this prompt, the base model has faults of its own (floating slices, stiff hands), and the engine adds a worse face. The face loss comes from the 4-bit layers of the packed UNet, not from a bug, the tokenizer, or TAESD.

Next step, if the user wants it: a packed UNet with more 8-bit layers (for example all 8-bit) and a compare with `compare_prompt.py`. The UNet stage peak is 813.3 MB, and the text stage peak 1,364.7 MB is the largest, so a larger UNet file can fit under the text stage peak.

## 2026-11-18: An 8-bit UNet file, side by side with the mixed plan

The user asked to try the fix from "Where the quality of a new prompt is lost": more 8-bit layers in the packed UNet.

Files added:

- `quant_plan_8bit.json`: the plan of `uv run quant_plan.py 900 --out quant_plan_8bit.json`.
- `weights/unet_q8.bin` and `weights/unet_q8.json`: the packed UNet with that plan.
- `quant_eval_8bit.txt`: the 20-image test with the new plan as a fourth version.
- `quant_eval/grid_mixed_only.png`: a copy of the old grid of "Mixed 4-bit and 8-bit plan, and a 20-image test", made before the new run wrote `quant_eval/grid.png` with four columns.
- `benchmark/results/2026-11-18-girl-watermelon-8bit.png`: the original, the engine with the mixed file, and the engine with the 8-bit file, in one row.

Files changed:

- `quant_plan.py`: the option `--out` (default `quant_plan.json`).
- `convert_weights.py`: the option `--stem`, the name of the output files, only with one network. Without it, the names are as before.
- `compare_prompt.py`: the option `--packed` (default `unet_packed`).
- `quant_eval.py`: the fourth version, "8-bit plan", from `quant_plan_8bit.json`.
- `engine/web/stages.js`: `runStages` takes `unetFile`. `DEFAULT_UNET_FILE` is `unet_packed`.
- `engine/web/generate.html`: `generate.html?unet=unet_q8` loads that UNet file. The byte compare runs only for the golden prompt with the default file.

Design choices:

- The new file has its own name. The OPFS cache of `weights.js` knows a file only by its name. A new file under the old name would load from the old cache copy on the iPhone, with no warning.
- The default stays `unet_packed` until the iPhone runs the new file. The benchmark page and the native tests still use the mixed file.
- The limit of 900 MB is above the all-8-bit size, so the solver could choose 8 bits for every layer. It kept one layer at 4 bits: `mid_block.attentions.0.transformer_blocks.0.attn2.to_q` (1,638,400 weights). Its measured error is 4.155e-03 at 4 bits and 4.282e-03 at 8 bits (`quant_sensitivity.txt`), so 8 bits does not help it.

Command: `uv run quant_plan.py 900 --out quant_plan_8bit.json`, in the project folder. Device: Mac, Python through `uv run`, CPU.

```
limit 900000000 bytes, plan 866172176 bytes, all 4-bit 434257936 bytes, all 8-bit 866991376 bytes
8-bit layers 281 of 282, 8-bit weights 863828480 of 865466880
predicted rms: all 4-bit 1.369e+00, plan 1.123e-01, all 8-bit 1.123e-01
wrote quant_plan_8bit.json
```

Command: `uv run convert_weights.py unet --plan quant_plan_8bit.json --dtype float16 --stem unet_q8`:

```
unet_q8: 968 tensors, 866195840 bytes, float16, packed 1 int4, 281 int8, largest up_blocks.0.resnets.0.conv1.weight 29491200 bytes, read back equal
```

`weights/unet_packed.bin` did not change: same size (700,021,120 bytes) and time stamp as before.

Command: `uv run compare_prompt.py "a photo of girl with watermelon" --out <scratchpad>/girl_q8 --packed unet_q8`, then Generate on `http://127.0.0.1:8765/engine/web/generate.html?unet=unet_q8` with the same prompt. Device: Mac CPU, then Mac Chromium:

```
prompt "a photo of girl with watermelon": ids [49406, 320, 1125, 539, 1611, 593, 19889, 49407]; not the golden ids
text_encoder: ran in 0.12 s; measured peak 1364710096 bytes (1364.7 MB)
unet_q8: 968 tensors in 968 GPU buffers, 866172176 bytes, largest buffer 29491200 bytes; from network in 2.56 s
unet: prepared; planned activations 111141888 bytes (111.1 MB); planned peak 979476080 bytes (979.5 MB); live 979410544 bytes
unet: ran in 1.81 s; measured peak 979476080 bytes (979.5 MB)
unet: freed; live 0 bytes
taesd: ran in 0.41 s; measured peak 201474352 bytes (201.5 MB)
total: 5.96 s
no golden image for this prompt and unet_q8, so no byte compare
```

The compare script of the earlier step, on the four images of this folder:

```
1_original vs 2_taesd: mean absolute byte difference 10.56, largest 144
2_taesd vs 3_engine_math: mean absolute byte difference 6.87, largest 211
3_engine_math vs 4_engine: mean absolute byte difference 0.00, largest 1
```

In the row image, the 8-bit face has natural eyes and a normal smile, close to the original. The mixed-plan face has distorted eyes and a wide grin with odd teeth.

Command: `uv run quant_eval.py > quant_eval_8bit.txt`. Device: Mac, Python through `uv run`, PyTorch on the Mac GPU ("mps"), LPIPS on the CPU:

```
image     all 4-bit PSNR   all 4-bit LPIPS    mixed plan PSNR  mixed plan LPIPS    8-bit plan PSNR  8-bit plan LPIPS  prompt
    0              11.08            0.8007              29.22            0.0470              34.29            0.0223  a photo of a cat sitting on a wooden table
    1              11.23            0.7146              21.13            0.1594              22.58            0.1228  a portrait of an old fisherman with a grey beard, natural light
    2              13.27            0.6336              29.49            0.0326              31.90            0.0201  a red sports car parked on a wet city street at night
    3              10.53            0.7276              25.99            0.0497              29.25            0.0270  a bowl of ramen with an egg and green onions, top view
    4              10.87            0.6869              26.25            0.0584              28.99            0.0360  a snowy mountain lake at sunrise, reflections on the water
    5              12.88            0.6586              33.32            0.0216              36.03            0.0159  a golden retriever puppy running on green grass
    6              10.67            0.7532              21.00            0.1362              25.57            0.0665  a cozy living room with a fireplace and bookshelves
    7              12.12            0.7164              28.96            0.0540              32.42            0.0223  a watercolor painting of a lighthouse on a cliff
    8              13.22            0.8380              29.94            0.0319              31.61            0.0201  a close-up photo of a hummingbird drinking from a flower
    9              10.89            0.7406              22.86            0.0885              28.33            0.0337  a busy market street in Marrakech with colorful spices
   10              11.30            0.7089              26.67            0.0516              29.32            0.0320  a futuristic city skyline with flying cars, digital art
   11              10.33            0.7727              24.05            0.0740              28.76            0.0358  a sign that says OPEN on a coffee shop door
   12              10.89            0.7595              23.99            0.0618              26.39            0.0384  a pile of fresh oranges and lemons in a wooden crate
   13              10.89            0.7222              25.33            0.0688              29.48            0.0379  an astronaut riding a horse on the moon
   14              10.65            0.6662              23.00            0.0897              27.86            0.0377  a young woman reading a book in a sunny cafe
   15              11.50            0.7632              25.81            0.0632              31.02            0.0248  a pencil sketch of an owl on a branch
   16              11.39            0.7487              25.19            0.0705              29.26            0.0291  a tropical beach with palm trees and turquoise water
   17              10.83            0.6807              23.09            0.0825              27.27            0.0374  a plate of chocolate chip cookies next to a glass of milk
   18              13.87            0.8753              35.10            0.0368              37.99            0.0251  an old castle in the fog, oil painting
   19              11.82            0.6433              27.82            0.0291              30.81            0.0182  a macro photo of dew drops on a spider web
 mean              11.51            0.7305              26.41            0.0654              29.96            0.0352
worst              10.33            0.8753              21.00            0.1594              22.58            0.1228
wrote quant_eval/grid.png
```

The default path after the change, golden prompt on `generate.html` without `?unet=`, Mac Chromium:

```
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from cache in 0.54 s
unet: ran in 1.80 s; measured peak 813301360 bytes (813.3 MB)
image bytes: 1199 of 1048576 differ from golden/packed/taesd_image, by at most 1
```

Calculations:

- The "all 4-bit" and "mixed plan" columns are the same numbers as in `quant_eval.txt`, so the test repeats exactly.
- 8-bit plan against the mixed plan, over 20 images: mean PSNR 26.41 to 29.96 dB (+3.55 dB), mean LPIPS 0.0654 to 0.0352 (0.0352 / 0.0654 - 1 = 46.2% less). LPIPS is lower on all 20 images.
- The two prompts with people: the woman in a cafe 0.0897 to 0.0377, the old fisherman 0.1594 to 0.1228. The fisherman is still the worst of the 20 in both plans.
- Girl with watermelon: distance of the engine math from the float32 UNet with TAESD, 11.96 (mixed) to 6.87 (8-bit) mean per byte.
- File: 866,195,840 - 700,021,120 = 166,174,720 bytes more. UNet stage peak: 979,476,080 - 813,301,360 = 166,174,720 bytes more, planned = measured. The text stage peak, 1,364,710,096 bytes, is still the largest stage.
- UNet time in Mac Chromium: 1.81 s with the 8-bit file, 1.80 s with the mixed file. 8-bit weights use the same kernels.
- In "Benchmark on the iPhone after the attention split", the largest page peak was 1,608.2 MB, 243.5 MB above the engine counter. If that overlap stays, 1,608.2 + 166.2 = 1,774.4 MB, 86.6% of the 2,048 MB tab limit. This is a guess from two numbers, not a measurement.

Not measured on the iPhone. Not known for the iPhone: the first load time of the 866 MB file, the peak of the page, and whether Safari keeps the tab alive.

Conclusion: the 8-bit UNet halves the mean LPIPS of the mixed plan over 20 images and fixes the face of the watermelon prompt, at the same speed, for 166 MB more file and UNet stage memory. The text stage stays the largest.

Next step: the user opens `https://10.120.1.125:8443/engine/web/generate.html?unet=unet_q8` on the iPhone, generates the watermelon prompt, then taps "Make 20", and sends the log. If the tab lives, the next step makes `unet_q8` the default and runs the benchmark page again.

## 2026-11-19: The 8-bit UNet on the iPhone, user report

Files changed: none.

Page: `https://10.120.1.125:8443/engine/web/generate.html?unet=unet_q8`. Device: iPhone. The user wrote: "It is better now." The user sent no log, no image and no "Make 20" result.

Not measured on the iPhone: the load time of `unet_q8`, the page peak, the longest submit, and whether the tab lives through "Make 20". So `unet_q8` is not the default yet.

Next step: the open items of "An 8-bit UNet file, side by side with the mixed plan" stay open. The user asked what is left before the engine is top-notch.

## 2026-11-19: Random seed, the same noise as torch.randn, and a guard on Start

The user asked to fix item 1 of the list in "The 8-bit UNet on the iPhone, user report": the engine had no seed and always used the golden noise, and the benchmark page could start a second job during a run. The third part of item 1, the 8-bit UNet as the default, waits for a "Make 20" run on the iPhone.

Files added:

- `engine/src/noise/mt19937.rs`: the Mersenne Twister MT19937, as PyTorch's `at::mt19937`: the same seeding (`init_with_uint32`), twist and tempering.
- `engine/src/noise/mod.rs`: `starting_noise(seed)`, 16,384 values, as PyTorch's `normal_fill`: one uniform number per value from the low 24 bits of each random number, then Box-Muller on each block of 16 (value j with value j + 8, the angle in f64 and rounded to f32).
- `engine/testdata/noise/make_noise.py` and `noise.safetensors`: `torch.randn((1, 4, 64, 64), generator=torch.Generator("cpu").manual_seed(seed))` for the seeds 0, 1, 42, 1234 and 4,294,967,295.
- `engine/tests/noise.rs`: compares the engine noise with the file, tolerance 1e-5.
- `benchmark/results/2026-11-18-girl-watermelon-seeds.png`: the watermelon prompt with seed 42 and seed 1234.

Files changed:

- `engine/src/lib.rs`: exports `starting_noise` and `NOISE_VALUES`. `engine/src/web.rs`: `startingNoise(seed)` for JavaScript.
- `engine/web/generate.html`: a seed field (default 42) and a "New seed" button, which puts a random 32-bit seed from `crypto.getRandomValues` in the field. The page logs the seed of each image. The byte compare runs only for the golden prompt, seed 42 and the default UNet file. At start, the page prints the largest difference between the noise of seed 42 and `golden/starting_noise.safetensors`.
- `compare_prompt.py`: the option `--seed`, which makes the noise with `torch.randn` and that seed.
- `benchmark/index.ts`: Start does nothing while the `pending` mark is set. It prints "a run is going on. Wait for the page to reload."

Design choices:

- The noise copies PyTorch, not a new generator. So seed N gives the same starting noise in the engine, in diffusers and in `compare_prompt.py`, and any engine image can be checked against PyTorch.
- `torch.randn` with seed 42 is the golden noise, so seed 42 keeps every golden check of the earlier steps.
- `torch.Generator.manual_seed` uses only the low 32 bits of the seed (`state[0] = seed & 0xffffffff`), so the engine takes a u32 seed.

Commands. Device: Mac, Python through `uv run`, CPU, then Mac native.

`uv run make_noise.py`, in `engine/testdata/noise/`. The first try failed with `ModuleNotFoundError: No module named 'numpy'`, because `safetensors.torch.save_file` needs numpy. The script now lists numpy:

```
seed_0: [-1.12584, -1.15236, -0.250579, -0.433879]
seed_1: [-1.525596, -0.750232, -0.653981, -1.609485]
seed_42: [1.926915, 1.487284, 0.900717, -2.105521]
seed_1234: [-0.111719, -0.49659, 0.163074, -0.881688]
seed_4294967295: [-2.664298, -1.091049, 0.451447, 0.228361]
torch 2.15.0
```

A Python check in the shell compared `seed_42` with `golden/starting_noise.safetensors` and printed `golden vs seed_42 largest diff 0.0`.

`cargo test --release --lib noise` passed `ten_thousandth_number_matches_the_cpp_standard`: the 10,000th number of seed 5489 is 4,123,659,995, as the C++ standard requires for `std::mt19937`.

`cargo test --release --test noise -- --nocapture`:

```
seed 0: largest difference from torch.randn 0e0
seed 1: largest difference from torch.randn 0e0
seed 1234: largest difference from torch.randn 0e0
seed 42: largest difference from torch.randn 0e0
seed 4294967295: largest difference from torch.randn 0e0
```

`cargo test --release` passed all 14 test files, 62 tests. `npm run typecheck` and `npm run build` passed. Clippy gave no new warning. `wasm-pack build --release --target web --out-dir web/pkg --no-typescript` passed.

Mutation check: `box_muller` wrote the sine first and the cosine second. The noise test failed, for example `seed 42: largest difference from torch.randn 5.7072554e0`. The code was put back.

Page: `http://127.0.0.1:8765/engine/web/generate.html`, golden prompt, seed 42. Device: Mac Chromium:

```
seed 42 noise: largest difference from golden/starting_noise 2.384185791015625e-7
seed 42
image bytes: 1185 of 1048576 differ from golden/packed/taesd_image, by at most 1
```

Page: `generate.html?unet=unet_q8`, prompt "a photo of girl with watermelon", seed 1234. Device: Mac Chromium:

```
seed 42 noise: largest difference from golden/starting_noise 2.384185791015625e-7
seed 1234
total: 3.77 s
no golden image for this prompt, seed and unet_q8, so no byte compare
```

Then `uv run compare_prompt.py "a photo of girl with watermelon" --out <scratchpad>/girl_1234 --packed unet_q8 --seed 1234`, and the compare script of the scratchpad:

```
1_original vs 2_taesd: mean absolute byte difference 8.48, largest 162
2_taesd vs 3_engine_math: mean absolute byte difference 4.03, largest 186
3_engine_math vs 4_engine: mean absolute byte difference 0.00, largest 1
```

Seed 1234 shows another picture: a girl in a pink top in front of trees, with two watermelon pieces. The engine image and the PyTorch original have the same layout.

Page: `http://127.0.0.1:8765/benchmark/index.html`, engine only, 1 run, 0 warm images. A Playwright script pressed Start again while run 0 went on. The log right after the second Start, and the table at the end:

```
next: this engine (Rust + WGSL), run 0 of 1, in 3 s. Press Stop to pause.
engine run 0 (warm-up, fills the caches, not counted): started
a run is going on. Wait for the page to reload.
```

```
| this engine (Rust + WGSL) | 1 of 1 | 3.57 | - | 1441.3 |
| engine | 0 (warm-up) | yes | 3.73 |  | 1537.5 |  |
| engine | 1 | yes | 3.57 |  | 1441.3 |  |
```

Calculations:

- WASM noise against the golden noise: largest difference 2.38e-7. The native build gives 0. The WASM build has its own `ln`, `sin` and `cos`, so a few values differ in the last bits. Near 2.0, one step of f32 is 2^-22 = 2.38e-7, so this is one rounding step.
- Golden image with seed 42 in Chromium: 1,185 differing bytes now, 1,199 before ("Prompt tokenizer in the engine"), by at most 1 in both. The page used the golden noise file before, and the WASM noise now. The last-bit differences of the noise change the rounding of 14 more bytes. The largest difference stays 1.
- The second Start made no second job: run 0 and run 1 each have one row, and run 0 has no "no GPU adapter" error, unlike run 0 of "Benchmark on the iPhone after the attention split".

Not measured on the iPhone. Not known for the iPhone: the noise difference of seed 42 in Safari's WASM, the golden image bytes with seed 42, and images with new seeds.

Conclusion: the engine makes the noise of any 32-bit seed, bit for bit the same as `torch.randn` natively and within one f32 step in the browser. Seed 42 still gives the golden image. Start no longer starts a second job during a run.

Next step: the user opens `https://10.120.1.125:8443/engine/web/generate.html?unet=unet_q8` on the iPhone, taps "New seed" and Generate a few times, then "Make 20", and sends the log. After that, `unet_q8` can become the default.

## 2026-11-19: Plan of the remaining work

Files added: `plan.md`. It lists the steps left before the engine is top-notch, in order, each with its reason, its numbers from this log, and the iPhone measurement that ends it. Step 1 is the 8-bit UNet as the default, after "Make 20" on the iPhone.

No command ran, and no number is new.

## 2026-11-19: "Make 20" with the 8-bit UNet on the iPhone, and the 8-bit UNet as the default

Page: `https://10.120.1.125:8443/engine/web/generate.html?unet=unet_q8`, "Make 20". Device: iPhone. The user sent the log. The user did not name the address, but the "-" in each byte compare fits `unet_q8`, which had no golden image then. The prompt and the seed are not in the log.

```
seed 42 noise: largest difference from golden/starting_noise 2.384185791015625e-7
ready
image 1 of 20: 5.31 s; longest submit 72.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 2 of 20: 5.23 s; longest submit 82.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 3 of 20: 5.53 s; longest submit 86.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 4 of 20: 5.86 s; longest submit 90.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 5 of 20: 8.44 s; longest submit 186.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 6 of 20: 9.40 s; longest submit 198.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 7 of 20: 9.34 s; longest submit 193.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 8 of 20: 8.78 s; longest submit 181.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 9 of 20: 8.45 s; longest submit 180.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 10 of 20: 8.27 s; longest submit 175.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 11 of 20: 8.24 s; longest submit 174.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 12 of 20: 8.30 s; longest submit 175.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 13 of 20: 8.38 s; longest submit 180.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 14 of 20: 8.62 s; longest submit 179.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 15 of 20: 8.55 s; longest submit 182.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 16 of 20: 8.60 s; longest submit 186.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 17 of 20: 8.80 s; longest submit 187.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 18 of 20: 8.68 s; longest submit 187.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 19 of 20: 8.70 s; longest submit 187.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
image 20 of 20: 9.02 s; longest submit 190.0 ms; live 0 bytes; WASM memory 24117248 bytes; - image bytes differ, by at most -
20 images in 160.58 s
```

Calculations:

- 20 of 20 images. Live GPU bytes 0 after each image. WASM memory 24,117,248 bytes after each image, so it did not grow.
- Safari's WASM noise of seed 42 differs from the golden noise by 2.38e-7, the same as Mac Chromium ("Random seed, the same noise as torch.randn, and a guard on Start").
- Images 1 to 4: mean (5.31 + 5.23 + 5.53 + 5.86) / 4 = 5.48 s. Images 5 to 20: mean 8.66 s, median 8.61 s. 8.66 / 5.48 = 1.58.
- Longest submit: 72 to 90 ms in images 1 to 4, 174 to 198 ms in images 5 to 20. The largest, 198 ms (image 6), is 198 / 250 = 79.2% of the limit.
- The step from image 4 to image 5 is sudden and stays. A slower GPU from heat fits this. It is not measured.
- The page has no page-wide GPU peak in this mode, so the peak of the 8-bit UNet on the iPhone is still not measured.

Files changed after the run:

- `dump_packed_golden.py`: an optional file name on the command line (`uv run dump_packed_golden.py unet_q8`). `unet_packed` keeps `golden/packed`. Any other file writes `golden/<name>`.
- `golden/unet_q8/`: the golden set of the 8-bit UNet.
- `engine/web/stages.js`: `DEFAULT_UNET_FILE` is `unet_q8`.
- `engine/web/generate.html`: the byte compare uses the golden set of the UNet file in use, for the golden prompt and seed 42.
- `README.md`: the commands for `unet_q8`, and a note that the benchmark results used `unet_packed`.
- `plan.md`: step 1 is done except the new benchmark run. The heat item has the new numbers.

The benchmark page takes the default of `stages.js`, so it now runs `unet_q8`. The native tests still use `unet_packed` and `golden/packed`.

Command: `uv run dump_packed_golden.py unet_q8`, in the project folder. Device: Mac, Python through `uv run`, CPU:

```
unet_output: float32 vs float64 largest difference 7.124e-06, largest value 4.200
unet_output: packed vs float32 weights largest difference 4.409e-02
```

The same line for `unet_packed` is `1.259e-01` ("Where the quality of a new prompt is lost"). `md5 -q golden/packed/*.safetensors | md5 -q` still printed `e5f46b621cb5214ff07b7a8dc72053b2`.

Page: `http://127.0.0.1:8765/engine/web/generate.html`, golden prompt, seed 42, then the same with `?unet=unet_packed`. Device: Mac Chromium:

```
unet_q8: 968 tensors in 968 GPU buffers, 866172176 bytes, largest buffer 29491200 bytes; from cache in 0.51 s
unet: ran in 1.79 s; measured peak 979476080 bytes (979.5 MB)
image bytes: 1064 of 1048576 differ from golden/unet_q8/taesd_image, by at most 1
```

```
unet_packed: 968 tensors in 968 GPU buffers, 699997456 bytes, largest buffer 29491200 bytes; from cache in 0.55 s
unet: ran in 1.75 s; measured peak 813301360 bytes (813.3 MB)
image bytes: 1185 of 1048576 differ from golden/packed/taesd_image, by at most 1
```

Conclusion: the 8-bit UNet made 20 of 20 images in a row on the iPhone with no memory growth, and it is now the default of the pages. In a long run, the iPhone slows down by 1.58 times after about 4 images, and the longest submit reaches 198 ms.

Next step: the user runs the benchmark page on the iPhone again (engine only, 10 runs, 3 warm images), now with `unet_q8`, and sends the Markdown export.

## 2026-11-19: The model moved to models/sd-turbo

The repository is now a Bun monorepo, and this model is `models/sd-turbo/`. The full entry, with the moves and the outputs, is "2026-11-19: Bun monorepo, step 1 of the restructure" in `PROGRESS.md` at the repository root. Paths in older entries are relative to the old project folder.

The commands changed:

| Old | New |
|---|---|
| `python3 -m http.server 8765 --bind 127.0.0.1` | `bun run serve`, in the repository folder |
| `python3 -m http.server 8443 ... --tls-cert ... --tls-key ...` | `bun run serve:lan`, in the repository folder |
| `npm run build` | `bun run build`, in the repository folder |
| `uv run dump_golden.py` (and the other scripts) | `uv run python/dump_golden.py`, in `models/sd-turbo/` |
| `--plan quant_plan.json` | `--plan quant/quant_plan.json` |

The pages moved too. For example, `engine/web/generate.html` is now at `/models/sd-turbo/engine/web/generate.html` on the server, and the benchmark page is at `/models/sd-turbo/benchmark/index.html`.

Page: `http://127.0.0.1:8765/models/sd-turbo/engine/web/generate.html`, golden prompt, seed 42. Device: Mac Chrome (see the root log).

```
image bytes: 1064 of 1048576 differ from golden/unet_q8/taesd_image, by at most 1
```

Not measured on the iPhone.

## 2026-11-19: Client, tests page, demo and harness

The full entry is "2026-11-19: Client, design system, site and iPhone harness, steps 2 to 5 of the restructure" in `PROGRESS.md` at the repository root.

The pages of `engine/web/` are gone. Their code is in `client/` (TypeScript), and the tests run on `/sd-turbo/dev/`. The old pages and their new places:

| Old page | New place |
|---|---|
| `engine/web/index.html` (SiLU) | test `silu` |
| `engine/web/layers.html` | test `layers` |
| `engine/web/attention.html` | test `attention` (256 query rows per submit) |
| `engine/web/taesd.html` | test `taesd` |
| `engine/web/bench.html` | test `bench` |
| `engine/web/weights.html` | test `weights` (files `taesd` and `unet_q8`) |
| `engine/web/generate.html`, golden prompt and seed | test `golden-image`; any prompt: the demo at `/sd-turbo/` |
| `engine/web/generate.html`, Make 20 | test `make-20` |
| `engine/web/generate.html`, Profile | test `profile` |
| `benchmark/index.html` | `/sd-turbo/benchmark/` |

The output lines of the tests are the same as on the old pages. wasm-pack now writes `engine/pkg/`, with TypeScript types.

The harness runs a test on the iPhone from the Mac: `bun run phone sd-turbo:golden-image`. The steps are in `README.md`.

Page: `http://127.0.0.1:8765/sd-turbo/dev/?test=all&run=chrome-all-1`. Device: Mac Chrome.

```
golden-image: image bytes: 1064 of 1048576 differ from golden/unet_q8/taesd_image, by at most 1
```

Not measured on the iPhone.

## 2026-11-19: Golden image on the iPhone, through the harness

The full entry is "2026-11-19: The first harness run on the iPhone" in `PROGRESS.md` at the repository root. Command: `bun run phone sd-turbo:golden-image --timeout 15`. Device: iPhone.

```
text_encoder: ran in 0.37 s; measured peak 1364710096 bytes (1364.7 MB)
unet: ran in 2.54 s; measured peak 979476080 bytes (979.5 MB)
taesd: ran in 0.73 s; measured peak 201474352 bytes (201.5 MB)
 556.1 s  golden-image: image bytes: 1073 of 1048576 differ from golden/unet_q8/taesd_image, by at most 1
```

The stages took 0.37 + 2.54 + 0.73 = 3.64 s. Both weight files came from the network (4.23 MB/s and 3.82 MB/s), so the total of 554.42 s is not an image time.

## 2026-11-19: The entry of SD-Turbo on the infer site

The full entry is "2026-11-19: infer, the dictionary redesign, and the rename" in `PROGRESS.md` at the repository root. The demo of this model is now its entry page at `/sd-turbo/`. `client/entry.ts` holds the definition, the parts and the measured times of the iPhone run of 2026-11-19. The test tensors in `engine/testdata/` are no longer in git. `README.md` has the `uv run` commands that make them again.

Not measured on the iPhone.
