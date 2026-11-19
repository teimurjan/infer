# SD-Turbo in Safari on iPhone

This project runs the SD-Turbo image model in a web browser with WebGPU. The engine is written in Rust with wgpu, and every GPU kernel is hand-written WGSL. The target is Safari on an iPhone 16 Pro Max. There, one tab gets about 2 GB of memory for everything: JavaScript, WASM and GPU buffers.

The onnxruntime-web example of SD-Turbo is the baseline. It came from Microsoft's [ort-webgpu example](https://guschmue.github.io/ort-webgpu/sd-turbo/index.html), and its code is in `baseline/index.ts` and `baseline/onnx-pipeline.ts`.

## Results

The benchmark page (`/sd-turbo/benchmark/` on the server, code in `apps/site/src/sd-turbo/benchmark/` and `benchmark/`) runs both runtimes with the same input. The input comes from the golden run of `python/dump_golden.py`. It is the 77 token ids of the prompt "a photo of a cat sitting on a wooden table". Both runtimes also get the same starting noise, sigma 14.614646911621094 and timestep 999.

### Mac Chromium

Device: MacBook Pro with Apple M1 Max (24-core GPU), macOS 26.6. Browser: Chrome for Testing 158 (Playwright build `chromium-1254`), with `--enable-unsafe-webgpu`. Date: 2026-11-18. Plan: 10 counted runs per runtime, 3 warm images per run.

| runtime | successes | cold s (median) | warm s (median) | peak GPU buffers MB (max) |
| --- | --- | --- | --- | --- |
| this engine (Rust + WGSL) | 9 of 10 | 3.30 | 3.29 | 4134.6 |
| onnxruntime-web example | 10 of 10 | 7.73 | 1.58 | 5545.7 |

Every run, with its error, is in `benchmark/results/2026-11-18-mac-chromium.md`. The two images are next to it.

| this engine | onnxruntime-web example |
| --- | --- |
| ![The cat from this engine](benchmark/results/2026-11-18-mac-chromium-engine.png) | ![The cat from the onnxruntime-web example](benchmark/results/2026-11-18-mac-chromium-onnx.png) |

What these numbers show:

- The engine makes the first image in 3.30 s, against 7.73 s for onnxruntime-web. That is 57.3% less time.
- onnxruntime-web makes the next images in 1.58 s, against 3.29 s for the engine. The engine loads its weights again for every image, from the browser's private file storage (OPFS). It does this because its text encoder and UNet together do not fit in the iPhone tab budget.
- The engine failed once. In run 1, one GPU submit took 292.7 ms. The engine stops any submit at 250 ms, because the iPhone loses the GPU device near 500 ms.
- Both runtimes hold more GPU memory than one iPhone tab can use (2048 MB). The engine's own counter never goes above 1364.7 MB in one stage. The page counter goes higher, because wgpu does not destroy a dropped buffer in the browser. The browser frees such a buffer only after a garbage collection. So the peak of the engine changes from run to run, from 2037.5 MB to 4134.6 MB.

### iPhone 16 Pro Max, Safari

Device: iPhone 16 Pro Max, Safari 26.6.2. Date: 2026-11-18. Plan: 10 counted runs per runtime, 3 warm images per run. The engine row is from the second run of the page, after the fix below. The onnxruntime-web row is from the first run, which stopped after run 2.

| runtime | successes | cold s (median) | warm s (median) | peak GPU buffers MB (max) |
| --- | --- | --- | --- | --- |
| this engine (Rust + WGSL) | 10 of 10 | 5.36 | 5.64 | 1608.2 |
| onnxruntime-web example | 0 of 2 | - | - | 0.0 |

Every run, with its error, is in `benchmark/results/2026-11-18-iphone-split-attention.md` (engine) and `benchmark/results/2026-11-18-iphone.md` (both runtimes, before the fix).

What these numbers show:

- On the iPhone, only the engine makes an image. onnxruntime-web never did. In runs 0 and 1, iOS stopped the tab. Run 2 failed with "Load failed", a failed download. On 2026-11-15, the same example died during the download of its UNet file. The evidence is in `../../crash-record/2026-11-15-iphone16promax.md`.
- The engine's peak of live GPU buffers stayed at or under 1608.2 MB. That is 78.5% of the 2048 MB tab limit.
- The warm images got slower over the runs. In runs 1 to 5, their median is 5.47 s. In runs 6 to 10, it is 6.64 s. The cold image stayed between 5.06 s and 5.92 s.
- The first page run had 0 of 10 engine successes. The self attention at 4096 tokens ran in one submit, and it took 250 ms to 298 ms. The engine stops any submit at 250 ms, because the iPhone loses the GPU device near 500 ms. The fix runs it in 4 submits of 1024 query rows.

## What the columns mean

- Run: one fresh page. The page runs one job, saves the result in `localStorage`, and reloads itself for the next job. So no run starts with anything loaded.
- Warm-up: run 0 of each runtime fills the browser caches. The table leaves it out. In every counted run, all weights come from the cache, not from the network.
- Cold: the time from the start of the run to the first image on the CPU. It includes loading the runtime, its weights from the cache, the shader compiles, and the first image. It does not include reading the golden inputs, which is the same for both runtimes.
- Warm: the time of one more image on the same page. The value is the median of all warm images of all successful runs.
- Peak GPU buffers: the largest number of bytes in live WebGPU buffers during the run. The page wraps `GPUDevice.createBuffer` and `GPUBuffer.destroy`, so it counts both runtimes in the same way. If the page destroys a buffer, or the garbage collector frees it, the buffer stops counting. JavaScript and WASM memory are not in this number.
- Success: the run made all of its images without an error, and no image is one flat color. A tab that the browser kills is a failure. The next page finds the mark of the unfinished run and records it.

The two runtimes do not run the same model files:

| part | this engine | onnxruntime-web example |
| --- | --- | --- |
| text encoder | float32, 1361.6 MB | float16 ONNX, about 1700 MB |
| UNet | 4-bit and 8-bit packed, 700.0 MB | float16 ONNX, about 640 MB |
| image decoder | TAESD, float32, 9.8 MB | SD-Turbo VAE decoder, float16 ONNX, about 95 MB |

## How to reproduce the results

### What you need

- A Mac with Apple silicon.
- Rust 1.97.1 with the `wasm32-unknown-unknown` target, and wasm-pack 0.13.1.
- uv 0.5.9, for the Python scripts. Each script names its own Python version and packages.
- Bun 1.3.10, for the pages, the build and the web server.
- For the iPhone: mkcert 1.4.4, and the iPhone on the same Wi-Fi network as the Mac.
- About 10 GB of free disk space for the model downloads and the weight files.

### 1. Make the engine weights

Run these commands in this folder, `models/sd-turbo`. The first run downloads SD-Turbo and TAESD from Hugging Face.

```sh
uv run python/convert_weights.py text_encoder
uv run python/convert_weights.py taesd
uv run python/convert_weights.py unet --plan quant/quant_plan.json --dtype float16
uv run python/quant_plan.py 900 --out quant/quant_plan_8bit.json
uv run python/convert_weights.py unet --plan quant/quant_plan_8bit.json --dtype float16 --stem unet_q8
uv run python/export_tokenizer.py
```

The folder `weights/` then holds `text_encoder.bin`, `taesd.bin`, `unet_packed.bin` and `unet_q8.bin`, each with its `.json` index. The pages use `unet_q8` (8 bits) since 2026-11-19. The benchmark results above used `unet_packed` (mixed 4 bits and 8 bits). The native tests still use `unet_packed`. It also holds `tokenizer_vocab.json` and `tokenizer_merges.txt`, which the engine's tokenizer reads. The benchmark page does not use the tokenizer. It feeds the golden token ids to both runtimes.

### 2. Make the golden inputs

Run this command in this folder, `models/sd-turbo`. It runs SD-Turbo in PyTorch on the CPU and writes `golden/`.

```sh
uv run python/dump_golden.py
```

The benchmark page reads four files from it: `golden/float32/text_input_ids.safetensors`, `golden/starting_noise.safetensors`, `golden/float32/sigma.safetensors` and `golden/float32/timestep.safetensors`.

### 2b. Make the test data of the kernels

The Rust tests and the tests page read float32 files in `engine/testdata/`. Git does not keep them, because these scripts make them again. Run them in this folder:

```sh
uv run engine/testdata/silu/make_data.py
uv run engine/testdata/make_layers.py
uv run engine/testdata/make_packed_layers.py
uv run engine/testdata/make_attention.py
uv run engine/testdata/noise/make_noise.py
```

### 3. Build the engine and the pages

Run these commands in the repository folder, two levels up.

```sh
rustup target add wasm32-unknown-unknown
bun install
bun run build
```

`bun run build` runs wasm-pack for the engine, which writes `engine/pkg/`. Then `build.ts` writes `dist/baseline.js` and copies the onnxruntime-web `.wasm` files into `dist/`. Last, the site in `apps/site/` builds its pages into `apps/site/dist/`.

### 4. Run on the Mac

1. Start the web server in the repository folder: `bun run serve`.
2. Open `http://127.0.0.1:8765/sd-turbo/benchmark/` in a Chromium browser with WebGPU.
3. Keep both runtimes selected. Set "runs" to 10 and "warm images per run" to 3.
4. Press Start.
5. Wait until the log says "done". The page reloads itself after each run, so do not close it.
6. Press Download Markdown or Download CSV.

The warm-up run of onnxruntime-web downloads about 2.4 GB from Hugging Face. On the Mac above, this took 511.70 s. The `404` for `favicon.ico` in the console is the missing page icon.

### 5. Run on the iPhone

The iPhone needs HTTPS for WebGPU and OPFS. These steps use a local certificate from mkcert. `<mac-ip>` is the Wi-Fi address of the Mac, which `bun run serve:lan` prints.

1. Make the certificate on the Mac, in the repository folder. It names the Wi-Fi address of the Mac now. If that address changes, run it again.

   ```sh
   bun run cert
   ```

2. Install `rootCA.pem` from the folder that `mkcert -CAROOT` prints on the iPhone. Then turn on full trust for it in Settings, General, About, Certificate Trust Settings. Do not copy `rootCA-key.pem`. It stays on the Mac.
3. Start the HTTPS server in the repository folder. It reads the two files of step 1 and prints the address of the Mac.

   ```sh
   bun run serve:lan
   ```

4. Set Auto-Lock to Never on the iPhone.
5. Open `https://<mac-ip>:8443/sd-turbo/benchmark/` in Safari.
6. Press Start, and keep Safari in front until the log says "done". iOS stops GPU work of an app in the background.
7. If iOS stops the tab, Safari reloads the page, and the page records a failed run. If Safari shows an error page, tap Reload. The page then goes on with the next run.
8. Press Download Markdown. Safari saves the file in the Files app.

The onnxruntime-web part can take a long time on the iPhone. Each run that dies starts the 2.4 GB download again, because the files never finish in the cache. To measure only the engine, clear the "onnxruntime-web" box before you press Start.

### Run on the iPhone from the Mac, with the harness

The harness drives Safari on the iPhone over the cable, so nobody has to tap the phone. It needs three settings once:

1. On the Mac, run `sudo safaridriver --enable`.
2. On the iPhone, turn on Settings, Apps, Safari, Advanced, Web Inspector.
3. On the same screen, turn on Remote Automation.

Then connect the iPhone with a cable, unlock it, and run one of these commands in the repository folder:

```sh
bun run phone sd-turbo:golden-image
bun run phone sd-turbo:all
bun run phone 'sd-turbo:benchmark?runs=10&warm=3&runtimes=engine' --timeout 120
```

The command prints each line of the page, and `PASSED` or `FAILED` at the end. The events of the run are saved in `runs/<run>.jsonl`. The test names are on the tests page: `silu`, `layers`, `attention`, `taesd`, `bench`, `weights`, `golden-image`, `make-20` and `profile`. With `--mac`, the harness drives Safari on the Mac instead. That needs "Allow remote automation" in the Developer settings of Safari.

### Other controls

- Stop: stops after the run that goes on now. Start goes on with the runs that have no result yet.
- Reset: deletes all results.
- `?model=<url>` on the page address: loads the ONNX models from another place, for example a local copy.

## Pages

The server serves the site of `apps/site/` at its root. These are the pages of this model:

| Address | Page |
| --- | --- |
| `/sd-turbo/` | The public demo: the model download, a prompt, and the live steps of each image. |
| `/sd-turbo/dev/` | The tests. Each test sends its lines to the Mac, into `runs/<run>.jsonl`. |
| `/sd-turbo/benchmark/` | The benchmark of the engine against onnxruntime-web. |
| `/models/sd-turbo/baseline/` | The onnxruntime-web example, unchanged. |

The engine code that these pages share is in `client/`: the WASM loader, the weight cache, the three-stage pipeline and the tests.

## The onnxruntime-web baseline

The baseline page is `baseline/index.html`. It loads the ONNX models from [schmuell/sd-turbo-ort-web](https://huggingface.co/schmuell/sd-turbo-ort-web) on Hugging Face. Build it with `bun run build`, serve it with `bun run serve`, and open `http://127.0.0.1:8765/models/sd-turbo/baseline/`.
