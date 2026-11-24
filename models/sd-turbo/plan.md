# Plan: what is left before the engine is top-notch

Goal: the best SD-Turbo in a browser on iPhone. That means any prompt and seed, quality close to the published model, and a fast warm image. It also means 10 of 10 runs and 20 images in a row without a crash. A native app can use the Apple Neural Engine (the iPhone's AI chip), and a web page cannot. So this plan does not try to beat native apps.

Every number below is from `PROGRESS.md`. Each step ends with a measurement on the iPhone. The step numbers stay the same, because the log names them.

## Where the engine is now (2026-11-24, iPhone 16 Pro Max, Safari 26.6.2)

| Measure | Value | Source in `PROGRESS.md` |
| --- | --- | --- |
| Download of the demo | 1219.8 MB (was 2239.3 MB) | Step 2, part 2 |
| Largest memory peak | 974.2 MB, the UNet stage (was 1364.7 MB, the text stage) | Step 2, part 2 |
| Warm image on a cool phone, "Make 20" images 2 to 5 | 4.105 s mean (was 4.838 s on 2026-11-24 before the decoder change) | Step 2b, part 3 |
| Warm image, "Make 20" images 2 to 20 | 5.56 s mean on 2026-11-20, 5.835 s on 2026-11-24 with a hot phone | Step 2, part 2, and step 2b, part 3 |
| Stage times of one image | text encoder 0.45 s, UNet 2.33 s, TAESD 0.46 s | Step 2, part 2 |
| Kernel time of one image | about 2.46 s (was 2.79 s) | Step 5 |
| Longest submit in "Make 20" | 131 ms on 2026-11-20, but 204 ms (82% of the limit) on 2026-11-24 with a hot phone | Step 2, part 2, and step 2b, part 3 |
| Image quality | 8-bit UNet: mean LPIPS 0.0352. 8-bit text encoder adds mean LPIPS 0.0057. | Step 2, part 1 |

The limits are in `CLAUDE.md`: iOS kills the tab at about 1920 MB, and one submit must stay under 250 ms.

The largest kernels of one image on the iPhone, after step 5:

| Kernel | Time per image |
| --- | --- |
| `conv3x3_packed` (UNet) | about 980 ms |
| `attention` | about 605 ms |
| `linear_packed` | about 355 ms |
| `conv3x3` (TAESD) | about 255 ms |

## Next steps, in order

1. Step 7, smaller submits: a warm phone brought the longest submit to 204 ms on 2026-11-24, 82% of the 250 ms limit.
2. Step 2b: decide on both loaded models (1.02 s per image, about 345 MB more).
3. Ship the work: `bun run deploy`, then one image on the live page on the iPhone.
4. Step 10: a faster `conv3x3_packed`, the largest kernel.
5. Step 11: register blocking in `attention`.
6. Step 1: the benchmark run with the 8-bit files.
7. Steps 8 and 9.

## Steps

### 1. Make the 8-bit UNet the default (done, except the new benchmark run)

- Done: `unet_q8` is the default on all pages. "Make 20" passed on the iPhone, 20 of 20. The golden set is `golden/unet_q8`.
- Done: the random seed gives the same noise as `torch.randn`. The iPhone profile run printed `seed 42 noise: largest difference from golden/starting_noise 2.384185791015625e-7`.
- Done: Start does nothing during a benchmark run.
- Open: the benchmark page (`benchmark/index.html`) has not run with the 8-bit files. When the iPhone benchmark shows 10 of 10 with them, this step is done.

### 2. Small text encoder, and both models stay loaded

Part a, the 8-bit text encoder (done):

- `text_encoder_q8`: every matrix at 8 bits, 342.1 MB in place of 1361.6 MB. Over 20 prompts, mean LPIPS 0.0057 against the float32 text encoder.
- The text stage peak fell from 1364.4 MB to 344.9 MB. The golden image, "Make 20" and the Mac tests pass. The file is on Hugging Face at commit `eb09c74f3b310cc507a645857a8ebf7317bbdf21`, and `apps/site/deploy.ts` reads it.

Part b, both models stay loaded (not done):

- Why: each image reads both weight files from the phone's cache and prepares both stages again. With both models on the GPU, every image after the first skips the reads and the preparations.
- Cost: today only one model is on the GPU at a time, and the largest peak is 974.2 MB. With both kept, the peak during the UNet becomes about 344.9 + 974.2 = 1319.1 MB. TAESD adds about 200 MB, and JavaScript and WASM add more. That is an estimate. The tab limit is about 1920 MB.
- Measured (2026-11-24, "Make 20" on the iPhone): the two loads and the two preparations cost 1.02 s of the 5.23 s of a warm image. The decoder stage lost another 0.91 s per image, nearly all of it to a new fetch of its 9.8 MB file. Since 2026-11-24, the page fetches that file once. A cool warm image now takes 4.105 s, against 4.838 s before (images 2 to 5 of "Make 20").
- If the user agrees after that, keep the text encoder and the UNet loaded for the whole page.
- Done when: the iPhone warm time drops, the golden compare passes, and "Make 20" passes under the tab limit.

### 3. TAESD 3x3 convolution: load each input value once per row (done)

- iPhone `conv3x3`: 410.64 ms to 253.21 ms per image, 38.3% less. The TAESD golden test passes.

### 4. Attention in one pass (done)

- An online softmax (a running maximum and sum) over tiles of 16 keys in workgroup memory. Each score now needs one `exp` in place of 16. The scores scratch left the activation plan.
- iPhone `attention`: 832.02 ms to 602.55 ms per image, 27.6% less. On the Mac GPU it is not faster, so the limit there is the loads of workgroup memory (see step 11).

### 5. Packed linear layer: 8 rows per invocation (done)

- iPhone `linear_packed`: 468.25 ms to 354.55 ms per image, 24.3% less.

### 6. Float16 math (measured, then dropped)

- Done: the iPhone has `shader-f16`, and it has no `subgroups` (the capabilities report in `packages/probes/crumbs.jsonl`). If the GPU has `shader-f16`, the engine now asks for it. The benchmark has a `math_f16` line.
- Result: on the iPhone, float16 math is the same speed as float32: 648.30 against 636.29 G ops/s.
- Decision: float16 can then help only a kernel that is limited by bytes. The largest kernels are not: on the iPhone, `conv3x3_packed` moves 0.84 GB/s, `attention` 1.98 GB/s and `linear_packed` 4.77 GB/s, against a copy speed of 44.74 GB/s. So the kernels stay float32. Steps 10 and 11 attack the real limits instead.

### 7. Run well for a long time

- If Rust drops a GPU buffer, destroy it at once. wgpu 30 does not call `destroy()` in the browser, so freed buffers stay until the garbage collector runs. Fix the wrong comment in `engine/src/memory/mod.rs` at the same time.
- Wait less often: the engine waits after each of about 150 submits per image. Waiting once per group of submits that stays under 250 ms recovers part of the time between submits.
- Heat: three "Make 20" runs in a row on 2026-11-24 made the UNet run grow to 5.59 s, and the longest submit to 214 ms (86% of the limit). After an 18-minute rest, the next run slowed from image 6 on, and its longest submit was 204 ms (82%). Smaller submits are now more urgent. Before steps 3 to 5, it was 1.58 times slower, with 198 ms (79%).
- Find the cause of the 292.7 ms `text embeddings` submit on the Mac.
- `bun run phone` reads the weights from the network for the first image of each run, and from the cache after that (2026-11-24). Find why the cache of an earlier run is not used.

### 8. Quality options

- 2 to 4 steps: SD-Turbo supports them. They give better hands and layouts, for more time. This can be a "quality" mode.
- Full VAE decoder as an option, for finer detail than TAESD.

### 9. Know the gap to native apps

- The user makes one SD-Turbo image in a native app, for example Draw Things, on the same iPhone, and reports the time. This gives a real target. It is not guessed.

### 10. A faster `conv3x3_packed` (new)

- Why: it is the largest kernel, about 980 ms of the 2.46 s of kernel time per image on the iPhone. It runs at 0.84 GB/s, so it is limited by its own work, not by memory speed. For each input row, it unpacks 12 integers one at a time, with `extractBits` and a conversion each.
- Do these two changes one at a time, each with a profile before and after.
  - The 8-bit UNet stores 4 weights per word. The built-in `unpack4xI8` can unpack the 4 weights at once.
  - Each workgroup loads its input tile once into workgroup memory. Then neighbor invocations do not load the same values again.
- Done when: the iPhone profile shows a lower `conv3x3_packed` row, and the packed layer and UNet golden tests pass.

### 11. Register blocking in `attention` (new)

- Why: after step 4, the attention kernel makes about one workgroup memory load for each 2 multiply-adds. That is why it got no faster on the Mac GPU.
- Do: each invocation computes more query rows or more output values, so each loaded key and value feeds more multiply-adds.
- Done when: the iPhone profile shows a lower `attention` row, and the attention and UNet golden tests pass.

## How each step is checked

- Each change runs the golden tests and a deliberate-bug (mutation) check on the Mac.
- Each speed change shows a profile table before and after, on the iPhone.
- Each step goes into `PROGRESS.md` with its outputs, as `AGENTS.md` says.
