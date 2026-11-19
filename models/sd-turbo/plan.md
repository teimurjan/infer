# Plan: what is left before the engine is top-notch

Goal: the best SD-Turbo in a browser on iPhone. That means any prompt and seed, quality close to the published model, and a fast warm image. It also means 10 of 10 runs and 20 images in a row without a crash. A native app can use the Apple Neural Engine (the iPhone's AI chip), and a web page cannot. So this plan does not try to beat native apps.

Every number below is from `PROGRESS.md`. Each step ends with a measurement on the iPhone.

## Status of item 1

| Part | Status |
| --- | --- |
| Random seed, the same noise as `torch.randn` | Done on the Mac. Not measured on the iPhone. |
| Start does nothing during a benchmark run | Done on the Mac. |
| The 8-bit UNet (`unet_q8`) as the default | Done. "Make 20" passed on the iPhone, 20 of 20. |

## Steps, in order

### 1. Make the 8-bit UNet the default (done, except the new benchmark run)

- Why: LPIPS measures how different two images look. Over 20 prompts, the 8-bit UNet halves the mean LPIPS of the mixed plan, from 0.0654 to 0.0352. It also fixes the faces.
- Risk: the iPhone page peak can grow from 1608.2 MB to about 1774 MB. That is an estimate, not a measurement. The tab limit is 2048 MB.
- Do: the user runs `generate.html?unet=unet_q8` with "Make 20" on the iPhone. If all 20 images finish, `unet_q8` becomes the default on all pages. A golden set for it is made, and the benchmark runs again.
- Done when: the iPhone benchmark shows 10 of 10 with the 8-bit UNet.

### 2. Small text encoder, and both models stay loaded

- Why: the engine loads its weights again for every image. On the iPhone, this takes about 1.0 s to 1.2 s for the text encoder and 0.4 s to 0.6 s for the UNet. The warm image takes 5.64 s. The float32 text encoder is 1364.7 MB at its peak, the largest stage. It does not fit next to the UNet.
- Do: pack the text encoder at 8 bits (about 340 MB), with the same sensitivity and image test as the UNet. Then keep the text encoder and the UNet on the GPU for the whole page.
- Also: about 1 GB less to download, and a much lower peak.
- Done when: the iPhone warm time drops, the golden compare passes, and the page peak stays under the tab limit.

### 3. TAESD 3x3 convolution: load each input value once per row

- Why: the float 3x3 convolution of TAESD takes 410.64 ms per image on the iPhone. The same change cut the packed 3x3 convolution by 39.2% on the iPhone.
- Done when: the iPhone profile shows a lower `conv3x3` row, and the TAESD golden test passes.

### 4. Attention in one pass

- Why: attention takes 838.71 ms per image on the iPhone. It runs at about a quarter of the math rate of the phone. Each score goes to memory once and comes back 16 times, with 16 copies of the same `exp`.
- Do: an online softmax (a running maximum and sum) in one loop. The scores never go to memory. The scores scratch leaves the activation plan.
- Done when: the iPhone profile shows a lower `attention` row, and the attention and UNet golden tests pass.

### 5. Packed linear layer: 8 rows per invocation

- Why: `linear_packed` takes 469.73 ms per image on the iPhone. Each unpacked weight feeds only 4 rows.
- Done when: the iPhone profile shows a lower `linear_packed` row, and the golden tests pass.

### 6. Float16 math

- Why: float16 halves the bytes that each kernel moves. The Apple GPU can also do float16 math faster. The onnxruntime-web baseline loads only after the phone reports `shader-f16`, and on this iPhone it got to the download. So the phone very likely has this feature. This is not checked directly.
- Do: first check `shader-f16` on the iPhone. Then move one kernel at a time to float16 between tensors, with a precision check against the golden files.
- Done when: the iPhone time per image drops, and every golden test stays within its tolerance.

### 7. Run well for a long time

- If Rust drops a GPU buffer, destroy it at once. wgpu 30 does not call `destroy()` in the browser, so freed buffers stay until the garbage collector runs. Fix the wrong comment in `engine/src/memory/mod.rs` at the same time.
- Wait less often: the engine waits after each of about 150 submits per image. Waiting once per group of submits that stays under 250 ms recovers part of the time between submits.
- Heat: in "Make 20" with the 8-bit UNet, the iPhone was 1.58 times slower from image 5 on, and the longest submit rose from 72 ms to 198 ms (79% of the 250 ms limit). Smaller submits keep a margin under the limit. Less work per image (steps 2 to 6) also makes less heat.
- Find the cause of the 292.7 ms `text embeddings` submit on the Mac.

### 8. Quality options

- 2 to 4 steps: SD-Turbo supports them. They give better hands and layouts, for more time. This can be a "quality" mode.
- Full VAE decoder as an option, for finer detail than TAESD.

### 9. Know the gap to native apps

- The user makes one SD-Turbo image in a native app, for example Draw Things, on the same iPhone, and reports the time. This gives a real target. It is not guessed.

## How each step is checked

- Each change runs the golden tests and a deliberate-bug (mutation) check on the Mac.
- Each speed change shows a profile table before and after, on the iPhone.
- Each step goes into `PROGRESS.md` with its outputs, as `AGENTS.md` says.
