# Progress log of the repository

This file records the steps that change the shared parts of the repository: `packages/`, the workspace files and the layout. Each model keeps its own log, for example `models/sd-turbo/PROGRESS.md`. The rules for entries are in `AGENTS.md`. The device names are in the table at the top of `models/sd-turbo/PROGRESS.md`, plus one new name:

| Name in this log | Device | Runtime |
|---|---|---|
| Mac Chrome | the same Mac as "Mac native" | the user's Google Chrome, WebGPU, the WASM build, driven by the Claude in Chrome extension |

## 2026-11-19: Bun monorepo, step 1 of the restructure

The user asked for five things: an automatic harness for the iPhone, a Bun monorepo with SD-Turbo as the first model, a design system for the demo pages, every script on Bun or uv, and a clean layout. This step does the monorepo, the scripts and the layout. The other parts are later steps.

Moves:

| Old path | New path |
|---|---|
| `engine/`, `compare/`, `benchmark/`, `weights/`, `golden/` | `models/sd-turbo/` (same names) |
| `README.md`, `plan.md`, `PROGRESS.md` | `models/sd-turbo/` |
| `index.html`, `index.ts`, `onnx-pipeline.ts`, `globals.d.ts`, `reference_output.png` | `models/sd-turbo/baseline/` |
| `*.py` | `models/sd-turbo/python/` |
| `quant_plan.json`, `quant_plan_8bit.json`, `quant_sensitivity.txt`, `quant_eval.txt`, `quant_eval_8bit.txt`, `quant_eval/` | `models/sd-turbo/quant/` |
| `dom.ts`, `errors.ts`, `breadcrumbs.ts`, `probe-store.ts` | `packages/browser/src/` |
| `memory/`, `watchdog/`, `capabilities/`, `crumbs.jsonl` | `packages/probes/` |
| `crumb-sink.ts` | `packages/server/src/crumbs.ts` |

Deleted, because a build makes them again: `dist/`, `node_modules/`, `package-lock.json`, `__pycache__/`, and the two webpack chunks at the root, `3f8fcd97177ed35d4f32.mjs` and `ae86c793c46ed060a6f3.wasm`. Deleted, because Bun replaces them: `webpack.config.ts`, `tsconfig.json`, `tsconfig.node.json`.

New files:

- `package.json`: the Bun workspaces `packages/*` and `models/*`, and the root commands `serve`, `serve:lan`, `build`, `typecheck`.
- `tsconfig.base.json`: the compiler options that every workspace extends.
- `.gitignore`: build output, `node_modules/`, and the model data in `models/*/weights/` and `models/*/golden/`.
- `packages/server/src/main.ts`, `files.ts`, `tls.ts`: the web server. It replaces `python3 -m http.server` and the webpack dev server.
- `models/sd-turbo/build.ts`: builds `dist/baseline.js` and `dist/benchmark.js` with `Bun.build`, and copies the onnxruntime-web `.wasm` and `.mjs` files into `dist/`.
- `package.json` and `tsconfig.json` in each workspace.

Changed files:

- `CLAUDE.md`: the new layout, the commands, and the rule to run scripts with Bun or uv.
- `AGENTS.md`: which log gets an entry.
- `models/sd-turbo/README.md`: the new commands and paths.
- `models/sd-turbo/benchmark/index.ts`, `onnx-runner.ts`, `paths.ts`, `models/sd-turbo/baseline/index.ts`, `index.html`, the three probe pages: the new import paths. The baseline page now loads `../dist/baseline.js`.
- `models/sd-turbo/python/quant_plan.py`, `quant_eval.py`: the quant files are in `quant/`.
- `models/sd-turbo/engine/testdata/make_packed_layers.py`: it imports `pack_weights.py` from `python/`.

Design choices:

- `engine/`, `weights/` and `golden/` stay next to each other in `models/sd-turbo/`. So every relative path in the Rust tests and in the engine pages stays the same. No Rust file changed.
- The Python scripts run from `models/sd-turbo/`, for example `uv run python/dump_golden.py`. Their paths `weights/` and `golden/` are relative to that folder, so they did not change.
- The server serves the whole repository folder, as `python3 -m http.server` did. It refuses each path with a part that starts with a dot, so `.git` is not served. It sends `cache-control: no-cache`, so a new build always reaches the page.
- The server has no time limit per request (`idleTimeout: 0`). The default of Bun is 10 s, and a 1 GB weight file takes longer on Wi-Fi.
- The engine stays its own Cargo crate. A root Cargo workspace is not needed with one Rust model.
- The empty `apps/` folder was not kept. The demo site gets it in a later step.

Command: `bun install`, in the repository folder. Device: Mac.

```
+ @types/bun@1.3.10
+ @webgpu/types@0.1.74
+ typescript@7.0.2

46 packages installed [7.57s]
```

Command: `bun run typecheck`, in the repository folder. Device: Mac.

```
@inference/browser typecheck: Exited with code 0
@inference/server typecheck: Exited with code 0
@inference/probes typecheck: Exited with code 0
@inference/sd-turbo typecheck: Exited with code 0
```

Mutation check: a file `packages/browser/src/mutation-check.ts` with `const broken: number = 'text';` made the check fail. The file was deleted.

```
@inference/browser typecheck: src/mutation-check.ts(1,7): error TS2322: Type 'string' is not assignable to type 'number'.
@inference/browser typecheck: Exited with code 1
```

Command: `bun run build:web`, in `models/sd-turbo/`. Device: Mac.

```
built dist/baseline.js, dist/benchmark.js, dist/baseline.js.map, dist/benchmark.js.map
```

Command: `bun run build:wasm`, in `models/sd-turbo/`. Device: Mac.

```
[INFO]: ✨   Done in 3.94s
[INFO]: 📦   Your wasm pkg is ready to publish at /Users/teimurgasanov/Projects/inference/models/sd-turbo/engine/web/pkg.
```

Command: `cargo test --release --test silu --test tokenizer --test noise -- --nocapture`, in `models/sd-turbo/engine/`. Device: Mac native.

```
SiLU: 100003 values, largest error 9.536743e-7
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.17s
```

Command: `cargo test --release --no-fail-fast`, in `models/sd-turbo/engine/`. Device: Mac native. Three test files failed, each on the 250 ms submit limit. All other files passed, among them `packed_layers`, `pipeline`, `stream_weights`, `taesd`, `text` and `unet_packed`.

```
attention_self: attention_self: timed submit 15 submit with 1 runs took 890.7 ms, the limit is 250.0 ms
linear: linear: timed submit 17 submit with 128 runs took 250.5 ms, the limit is 250.0 ms
up_blocks.2.attentions.2 proj_in: one submit took 392.5 ms, the limit is 250.0 ms
```

Reruns of the three files, one at a time. Device: Mac native.

```
attention_cross: attention_cross: timed submit 28 submit with 32 runs took 803.4 ms, the limit is 250.0 ms
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 5.08s
linear: linear: timed submit 20 submit with 128 runs took 307.6 ms, the limit is 250.0 ms
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.76s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 14.62s
```

Then `cargo test --release --test attention --test layers`, 3 times:

```
attention_self: attention_self: timed submit 22 submit with 1 runs took 609.4 ms, the limit is 250.0 ms test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.57s 
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.73s test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.84s 
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.53s test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.05s 
```

The failures are timing failures, not path failures: each test read its files and ran. No Rust file changed. The model log records the same attention failure on the 250 ms limit before this step ("the attention test failed 2 times out of 4 runs"). During these runs, other programs on the Mac used the CPU: two `bun server.ts` processes of the Telegram plugin at 98% each. Their GPU use is not measured.

Command: `uv run engine/testdata/silu/make_data.py`, in `models/sd-turbo/`. Device: Mac, Python through uv. The md5 of the two files was the same before and after:

```
9e8c1af4d36f7d4e427031c203672b72
f88de652352765e204b29371aca8df18
```

Command: `uv run engine/testdata/make_packed_layers.py`, in `models/sd-turbo/`. This checks the new import path of `pack_weights.py`. The md5 of each `*_q4` and `*_q8` folder was the same before and after. Last lines:

```
conv3x3_stride2_q8 (down_blocks.0.downsamplers.0.conv): input (1, 320, 64, 64) -> output (1, 320, 32, 32), levels -127 to 127
packed testdata unchanged
```

Command: `bun run serve`, in the repository folder, then `curl` for each path. Device: Mac.

```
serving /Users/teimurgasanov/Projects/inference at http://127.0.0.1:8765/
200 text/html;charset=utf-8 13461 /models/sd-turbo/engine/web/generate.html
200 application/wasm 1275237 /models/sd-turbo/engine/web/pkg/engine_bg.wasm
200 application/json;charset=utf-8 119668 /models/sd-turbo/weights/unet_q8.json
200 application/octet-stream 65624 /models/sd-turbo/golden/starting_noise.safetensors
200 text/html;charset=utf-8 1639 /models/sd-turbo/benchmark/
200 text/javascript;charset=utf-8 189302 /models/sd-turbo/dist/benchmark.js
200 text/html;charset=utf-8 950 /packages/probes/memory/
404 text/plain;charset=utf-8 9 /.git/config
404 text/plain;charset=utf-8 9 /models/../.git/HEAD
404 text/plain;charset=utf-8 9 /models/sd-turbo/missing
200 866195840 0.426235s unet_q8.bin
866195840
204
```

The last two lines: the size of `unet_q8.bin` on disk is the same as the bytes served, and `POST /crumb` returned 204. The test breadcrumb was then removed from `crumbs.jsonl`.

Page: `http://127.0.0.1:8765/models/sd-turbo/engine/web/index.html`. Device: Mac Chrome.

```
SiLU: 100003 values, largest error 9.536743e-7
```

Page: `http://127.0.0.1:8765/models/sd-turbo/engine/web/generate.html`, golden prompt, seed 42, Generate. Device: Mac Chrome. The weights came from the network, because this browser had no OPFS copy yet.

```
text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from network in 5.47 s
text_encoder: ran in 0.43 s; measured peak 1364710096 bytes (1364.7 MB)
unet_q8: 968 tensors in 968 GPU buffers, 866172176 bytes, largest buffer 29491200 bytes; from network in 3.21 s
unet: ran in 2.04 s; measured peak 979476080 bytes (979.5 MB)
taesd: ran in 0.47 s; measured peak 201474352 bytes (201.5 MB)
image bytes: 1064 of 1048576 differ from golden/unet_q8/taesd_image, by at most 1
```

The image compare line is the same as the Mac Chromium line before the move.

The benchmark page, the baseline page and the three probe pages each loaded their script with HTTP 200, and the console had no errors. Their runs were not started. Device: Mac Chrome.

Not measured on the iPhone. `bun run serve:lan` was not started in this step.

Conclusion: the repository is a Bun monorepo. The engine, the tests and the pages give the same results from the new paths.

Next step: the iPhone harness in `packages/harness/`. It drives Safari on the iPhone over USB with `safaridriver`, opens a page, and saves the result of the page on the Mac.

## 2026-11-19: Client, design system, site and iPhone harness, steps 2 to 5 of the restructure

The user asked to finish all parts of the restructure in one prompt. This entry covers the four parts that were left: the typed client of the engine, the design system, the site with the public demo and the developer pages, and the iPhone harness.

Files added:

- `models/sd-turbo/client/`: the typed TypeScript client of the engine. `engine.ts` (WASM load and GPU), `weights.ts` (stream a weight file into the engine), `cache.ts` and `download.ts` (OPFS cache), `golden.ts`, `image.ts`, `pipeline.ts` (one image in three stages, with events), `schedule.ts`, `noise.ts`, `urls.ts`, `fetch.ts`, `engine-types.ts`.
- `models/sd-turbo/client/tests/`: the test suite of SD-Turbo: `silu`, `layers`, `attention`, `taesd`, `bench`, `weights`, `golden-image`, `make-20` and `profile`.
- `packages/ui/`: the design system. `tokens.css`, `base.css`, `layout.css`, `components.css`, `dom.ts`, `format.ts`, and the components `button`, `badge`, `progress`, `log`, `steps`, `meter` and `topbar`.
- `packages/browser/src/run-event.ts`, `report.ts`, `test-suite.ts`: the events that a test page sends to the Mac, and the shape of a test suite.
- `packages/server/src/runs.ts`, `server.ts`, `cert.ts`: the `POST /report` route, the server as a function, and `bun run cert`.
- `packages/harness/`: `bun run phone`. `device.ts`, `safaridriver.ts`, `webdriver.ts`, `args.ts`, `print.ts`, `run.ts`, `main.ts`.
- `apps/site/`: the site. Pages `/`, `/sd-turbo/`, `/sd-turbo/dev/` and `/sd-turbo/benchmark/`.

Files deleted:

- `models/sd-turbo/engine/web/`: the pages `index.html`, `layers.html`, `attention.html`, `taesd.html`, `bench.html`, `weights.html`, `generate.html`, and the modules `weights.js`, `golden.js`, `stages.js`. Their code is now in `client/` and in the tests page. wasm-pack now writes `engine/pkg/`, with TypeScript types.
- `models/sd-turbo/benchmark/index.ts` and `index.html`: the page is now `apps/site/src/sd-turbo/benchmark/`. The logic files stay in `models/sd-turbo/benchmark/`.
- `toRgba` in `models/sd-turbo/benchmark/image.ts`: it was a copy of the same function in `client/image.ts`.

Changed files: `package.json` (workspace `apps/*`, scripts `phone` and `cert`), `tsconfig.base.json` (lib `es2023`, for `findLastIndex`), `.gitignore` (`runs/`), `packages/server/src/files.ts` (it serves `apps/site/dist/` first), `packages/server/src/tls.ts` (it picks the Mac address that the certificate names), `models/sd-turbo/benchmark/engine-runner.ts`, `inputs.ts`, `paths.ts` (they use the client), `models/sd-turbo/build.ts` (it builds only the baseline), `CLAUDE.md`, `AGENTS.md`, `README.md`, `models/sd-turbo/README.md`.

Design choices:

- The design system follows the Orca app, which the user named. Its CSS in `/Applications/Orca.app` (version 1.4.262) has neutral grays (`#0a0a0a`, `#171717`, `#f0f0f0`, `#e5e5e5`), a radius of 10 px (`--radius: .625rem`), one blue (`#1447e6`), the system font, and SF Mono. The tokens take those values. Only values were taken, no code and no logo.
- No UI framework. The pages build their elements with a 20-line `h()` function. On the iPhone, the JS heap shares the 1920 MB tab budget with the GPU, so a small page leaves more room for the model.
- A test page sends each line, number and result with `navigator.sendBeacon`. A beacon still goes out when iOS kills the tab right after it. The server checks each event with `parseRunEvent` and saves it in `runs/<run>.jsonl`. Separate beacons can arrive out of order, so each event has a `seq` number, and a block of many lines goes as one event.
- The harness starts the server in its own process. So it gets the events of its run at once, with no polling of files.
- The harness uses safaridriver, the WebDriver server of Safari. WebDriver can drive Safari on an iPhone that is connected with a cable. The harness checks the session every 15 s, so a lost window ends the run with a reason.
- A test page that loads a second time with the same run id reports a failure and stops. iOS reloads a tab after it kills it, and without this mark the test would start again in a loop.
- The demo downloads the model to OPFS only after a tap on "Download", because 2.24 GB must never start by itself on a phone. The download writes to storage only, with no GPU memory.
- The demo uses `client/schedule.ts` (sigma 14.614646911621094, timestep 999) and not the golden files, because a public site will not have `golden/`. The golden-image test checks that the two agree.
- A failed write to the cache no longer stops a weight load. The load goes on from the network, and the result names the cache error. Before, a full storage stopped the image.
- In the browser tests, a kernel test passes if the engine does not throw, as on the old pages. SiLU also checks the native bound of 1e-4. The golden image passes with a byte difference of at most 1, the result on the Mac before the restructure.
- `bun run phone` builds the site first. A site bundle that is older than the event format sends events that the server refuses, which happened once in this step (see below).

Command: `bun run typecheck`, in the repository folder. Device: Mac.

```
@inference/browser typecheck: Exited with code 0
@inference/ui typecheck: Exited with code 0
@inference/probes typecheck: Exited with code 0
@inference/server typecheck: Exited with code 0
@inference/sd-turbo typecheck: Exited with code 0
@inference/harness typecheck: Exited with code 0
@inference/site typecheck: Exited with code 0
```

Command: `bun run build`, in the repository folder. Device: Mac.

```
@inference/probes build: Exited with code 0
@inference/sd-turbo build: built dist/baseline.js, dist/baseline.js.map
@inference/sd-turbo build: Exited with code 0
@inference/site build: built 4 pages, 13 files
@inference/site build: Exited with code 0
```

Command: `curl` for each address, with `bun run serve`. Device: Mac. The `/report` line sends a run id with a space and a `!`.

```
200 text/html;charset=utf-8 /
200 text/html;charset=utf-8 /sd-turbo/
200 text/html;charset=utf-8 /sd-turbo/dev/
200 text/html;charset=utf-8 /sd-turbo/benchmark/
200 application/wasm /models/sd-turbo/engine/pkg/engine_bg.wasm
HTTP/1.1 200 OK
content-length: 866195840
run must match [A-Za-z0-9_-]{1,80} 400
```

Page: `http://127.0.0.1:8765/sd-turbo/dev/?test=all&run=chrome-all-1`. Device: Mac Chrome. The result lines from `runs/chrome-all-1.jsonl`:

```
silu: PASSED: largest error 9.536743e-7, bound 0.0001
layers: FAILED: FAILED: conv1x1: timed submit 15 submit with 512 runs took 258.8 ms, the limit is 250.0 ms
attention: PASSED: 2 cases ran
bench: PASSED: benchmarks ran
weights: PASSED: live 0 bytes after free
golden-image: total: 8.24 s
golden-image: image bytes: 1064 of 1048576 differ from golden/unet_q8/taesd_image, by at most 1
golden-image: PASSED: image bytes: 1064 of 1048576 differ from golden/unet_q8/taesd_image, by at most 1
make-20: image 1 of 20: 6.83 s; longest submit 101.4 ms; live 0 bytes; WASM memory 186384384 bytes; 1064 image bytes differ, by at most 1
make-20: image 2 of 20: 5.86 s; longest submit 90.9 ms; live 0 bytes; WASM memory 186384384 bytes; 1064 image bytes differ, by at most 1
make-20: image 3 of 20: 7.46 s; longest submit 93.6 ms; live 0 bytes; WASM memory 186384384 bytes; 1064 image bytes differ, by at most 1
make-20: image 4 of 20: 7.02 s; longest submit 130.1 ms; live 0 bytes; WASM memory 186384384 bytes; 1064 image bytes differ, by at most 1
make-20: image 5 of 20: 5.54 s; longest submit 85.1 ms; live 0 bytes; WASM memory 186384384 bytes; 1064 image bytes differ, by at most 1
make-20: FAILED: FAILED: text_model.encoder.layers.0: one submit took 338.6 ms, the limit is 250.0 ms
profile: FAILED: FAILED: copy: calibration submit with 1 runs took 296.4 ms, the limit is 250.0 ms
done false 6 of 9 tests passed
```

The golden image is byte for byte the same as before the restructure: 1064 of 1048576 bytes differ, by at most 1. The three failures are submits over 250 ms on the Mac. In the same run, the UNet took 3.99 s, against 2.04 s in the run of the entry before this one, so the GPU of the Mac was busy with other work. The cause is not measured. The doubled "FAILED: FAILED:" came from the runner and the test, and the runner now writes it once.

Failed run, and the fix: `?test=profile&run=chrome-profile-1`, on a fresh page. Device: Mac Chrome.

```
profile: FAILED: Cannot read properties of undefined (reading '__wbindgen_malloc')
```

The tokenizer was made before the WASM module had loaded, because `loadTokenizer` did not wait for it. The public demo had the same race. `client/engine.ts` now loads the WASM module once (`loadWasm`), and `loadTokenizer` and `openSession` both wait for it. The same page then got past the tokenizer (`run=chrome-profile-2`):

```
profile: FAILED: text embeddings: one submit took 429.5 ms, the limit is 250.0 ms
```

This is the slow "text embeddings" submit on the Mac that `models/sd-turbo/plan.md` step 7 already lists.

Page: `http://127.0.0.1:8765/sd-turbo/`, the example prompt "a lighthouse on a cliff at sunset, oil painting", seed 42, Generate. Device: Mac Chrome. The model card showed "Ready", because the weights were in OPFS from the tests. The steps of the run card:

```
done Read the prompt | 12 of 77 token slots used. ;; done Text encoder | Turns the ids into 77 vectors of 1024 numbers. | 278 ms ;; done Denoise, one step | The UNet turns random noise into a 4 x 64 x 64 latent image. | 1.84 s ;; done Decode | TAESD turns the latent into 512 x 512 pixels. | 448 ms ##ARIA Done. Change the prompt or the seed for a new image.
```

The image showed a lighthouse on a cliff, and the caption said 4.55 s. The console had no errors. A screenshot also showed a bug: the hidden "Weights" bar was visible, because a `display: flex` class beats the `hidden` attribute. `base.css` now has `[hidden] { display: none !important; }`.

Page: `http://127.0.0.1:8765/sd-turbo/benchmark/?run=chrome-bench-1&runs=1&warm=1&runtimes=engine`. Device: Mac Chrome. The plan started by itself and reloaded the page for each run:

```
engine run 0: cold 4.60 s
engine run 0: warm 1 4.14 s
engine run 1: cold 4.32 s
engine run 1: warm 1 7.43 s
done true 2 of 2 runs succeeded
```

In this run, the table lines arrived in a mixed order, because separate beacons overtook each other. The table now goes as one event. The cold time no longer includes a separate fetch of `engine.js`, because the engine JavaScript is now bundled into the page.

Width check: each of the four site pages in an iframe 390 px wide. Device: Mac Chrome. `scroll` equal to `inner` means no sideways scroll.

```
{"/":{"inner":386,"scroll":386,"wide":[]},"/sd-turbo/":{"inner":386,"scroll":386,"wide":[]},"/sd-turbo/dev/":{"inner":386,"scroll":386,"wide":[]},"/sd-turbo/benchmark/":{"inner":386,"scroll":386,"wide":[]}}
```

Command: `bun run phone sd-turbo:silu`. Device: Mac, with the iPhone on the cable.

```
device: Teimur’s iPhone, iOS 26.7.1, 00008140-0010650421BB001C
FAILED: the certificate names 10.120.1.125, 127.0.0.1, but the Mac has 10.120.1.165, 100.124.252.28, 192.168.64.1, 192.168.139.3, 192.168.158.0, 192.168.156.0, 169.254.134.91: make a new certificate with mkcert
```

The Wi-Fi address of the Mac changed since the certificate was made. Command: `bun run cert`. Device: Mac.

```
The certificate is at "/Users/teimurgasanov/.config/inference-lan/lan-cert.pem" and the key at "/Users/teimurgasanov/.config/inference-lan/lan-key.pem" ✅
DNS:localhost, IP Address:10.120.1.165, IP Address:127.0.0.1
```

The second line is the subject of the new certificate, read with `X509Certificate`. Then `bun run phone sd-turbo:silu --timeout 2`:

```
device: Teimur’s iPhone, iOS 26.7.1, 00008140-0010650421BB001C
FAILED: cannot listen on port 8443: Failed to start server. Is port 8443 in use?. Stop the other server first.
```

Port 8443 belongs to a `python3 -m http.server 8443` that ran for 18 hours. It serves the old layout. It was not stopped, because it is the user's process. A direct `safaridriver` session for the iPhone, before the harness, gave:

```
Could not create a session: Some devices were found, but could not be used:\n- Teimur’s iPhone (00008140-0010650421BB001C): Remote Automation is turned off (turn it on via Settings > Safari > Advanced > Remote Automation)
```

Command: `bun run phone sd-turbo:silu --mac --timeout 1`. Device: Mac.

```
FAILED: Could not create a session: You must enable 'Allow remote automation' in the Developer section of Safari Settings to control Safari via WebDriver.
```

After each failed run, `pgrep -fl safaridriver` printed nothing, and nothing listened on port 8765. So the harness stops what it started.

The event path of the harness without WebDriver: a script started the server with `startServer` and printed each event with `eventText`, as `run.ts` does. Chrome opened `/sd-turbo/dev/?test=silu&run=harness-listen-2`. The first try printed nothing, because the site bundle was older than the `seq` field, and the server refused each event with 400. Nothing on the page showed it. The server now prints each refused event, and `bun run phone` builds the site first. After a new build:

```
   0.1 s  page loaded
   0.1 s  silu: started
   0.1 s  silu: SiLU: 100003 values, largest error 9.536743e-7
   0.1 s  metric silu: silu largest error = 9.5367431640625e-7
   0.1 s  silu: PASSED: largest error 9.536743e-7, bound 0.0001
   0.1 s  PASSED: largest error 9.536743e-7, bound 0.0001
```

Not measured on the iPhone. The harness reached the phone, but no page ran there yet.

Conclusion: the four parts are built, and each one ran on the Mac. The demo makes images with live steps, the tests page records every run in `runs/`, and the harness finds the iPhone and stops with a clear reason at each missing setting.

Next step: the user does three things. Stop the old Python server on port 8443. Turn on Remote Automation on the iPhone. Then run `bun run phone sd-turbo:golden-image`.

## 2026-11-19: The first harness run on the iPhone

The user turned on Remote Automation and stopped the old server on port 8443. Their first `bun run phone` printed the lines below. The reason after the colon was not in the message that the user sent.

```
device: Teimur’s iPhone, iOS 26.7.1, 00008140-0010650421BB001C
FAILED: Could not create a session: Some devices were found, but could not be used:
```

A direct `safaridriver` session a minute later opened (`"safari:deviceName":"Teimur’s iPhone","safari:platformVersion":"26.7.1","browserVersion":"26.6.2"`). The cause of the first failure is not known.

Command: `bun run phone sd-turbo:golden-image --timeout 15`, in the repository folder. Device: iPhone, driven by the harness over the cable. Events: `runs/20261119072352-sd-turbo-dev-test-golden-image-hi29.jsonl`.

```
   1.7 s  golden-image: prompt "a photo of a cat sitting on a wooden table": the same as the golden ids
   1.7 s  golden-image: seed 42 noise: largest difference from golden/starting_noise 2.384185791015625e-7
   1.7 s  golden-image: schedule: sigma 14.614646911621094, timestep 999; the same as client/schedule.ts
 323.7 s  golden-image: text_encoder: 372 tensors in 372 GPU buffers, 1361551360 bytes, largest buffer 202375168 bytes; from network in 321.98 s
text_encoder: ran in 0.37 s; measured peak 1364710096 bytes (1364.7 MB)
 551.0 s  golden-image: unet_q8: 968 tensors in 968 GPU buffers, 866172176 bytes, largest buffer 29491200 bytes; from network in 226.74 s
unet: ran in 2.54 s; measured peak 979476080 bytes (979.5 MB)
taesd: ran in 0.73 s; measured peak 201474352 bytes (201.5 MB)
 556.1 s  golden-image: total: 554.42 s
 556.1 s  golden-image: image bytes: 1073 of 1048576 differ from golden/unet_q8/taesd_image, by at most 1
PASSED: image bytes: 1073 of 1048576 differ from golden/unet_q8/taesd_image, by at most 1
```

Calculations:

- Download rate of the text encoder: 1361551360 / 1e6 / 321.98 = 4.23 MB/s. Of the UNet: 866172176 / 1e6 / 226.74 = 3.82 MB/s.
- Time in the three stages: 0.37 + 2.54 + 0.73 = 3.64 s. The downloads took 321.98 + 226.74 = 548.72 s of the 554.42 s total.

The weights came from the network, not from the OPFS cache that the user's own Safari tabs filled. So the automated Safari session has its own storage. Whether that storage keeps the weights for the next harness run is not measured.

The iPhone image differs from the PyTorch image in 1073 bytes, and the Mac image in 1064 bytes. Both are at most 1 apart, which passes.

After the run, `pgrep -fl safaridriver` printed nothing, and nothing listened on port 8443.

Conclusion: the harness ran a test on the iPhone with no tap on the phone, and the golden image passed there.

Next step: run `bun run phone sd-turbo:weights`. Its first line says "from cache" or "from network", which shows whether the harness keeps the weights between runs.

## 2026-11-19: infer, the dictionary redesign, and the rename

The user asked for an alternative, experimental design of the design system and the site, with no logo and the word "infer" set in type. They named typesafe.ai, pauladealvaro.com and glossaryoftime.com as designs they like, and asked for a landing with every available inference. Then they asked to rename the folder to `infer`, to fix `.gitignore`, to write a short README with facts only, and to push the repository to a new GitHub repo named `infer`.

Product facts, from three questions to the user: the landing lists only models that run today, the audience is engineers, and the site is named "infer" everywhere. They are in `apps/site/PRODUCT.md`.

Direction: the impeccable skill rolled a direction and showed five sketches on a decision page. The user chose "The Dictionary of Inferences". "infer" is a headword, each model is an entry that you can run, and each phone limit is a usage note. The contract is in `apps/site/.impeccable/surfaces/pages-index-html.md`, and the system is in `apps/site/DESIGN.md`.

Files added:

- `apps/site/PRODUCT.md`, `apps/site/DESIGN.md`, `apps/site/.impeccable/` (the decision sketches, the surface brief, `design.json`).
- `packages/ui/src/fonts.css`, `paper.css`, `components/running-head.ts`, `thumb-index.ts`, `show-through.ts`, `cite.ts`. The fonts Archivo and Source Serif 4 come from `@fontsource-variable` packages, and the site serves them from `/fonts/`.
- `packages/browser/src/entry.ts`: the shape of a dictionary entry. `models/sd-turbo/client/entry.ts`: the entry of SD-Turbo, with the measured times and their source run.
- `apps/site/src/shell/entries.ts`, `notes.ts`: the list of entries and the usage notes.

Files changed: the tokens, base and components of `packages/ui`, every page of `apps/site`, the benchmark view, and `CLAUDE.md`. `packages/ui/src/layout.css`, `components/topbar.ts` and `components/stat.ts` were deleted.

Design choices:

- Ultramarine `#1d2bd9` marks only what starts a run: the run buttons and the thumb tab of each entry. Everything at rest is black ink on `#fafaf7` paper. A failure is red.
- The landing prompt goes to the entry page after `#`, so the browser does not send it to the server.
- The entry page shows its parts at once. The GPU check only turns the run button on or off.
- Each number on a page has a mark that points to its source run.
- The reverse page is the real usage text, mirrored, at 0.028 opacity, in two grid columns on a phone and three on a wide screen.

Rename: the folder is now `/Users/teimurgasanov/Projects/infer`, the workspace scope is `@infer/`, and the root package is `infer`. Paths in older entries name the old folder.

`.gitignore` now also leaves out the test tensors (`models/*/engine/testdata/**/*.bin` and `*.safetensors`), `models/*/quant/quant_eval/`, `.env` files, `*.pem` and `*.key`. Before this change, the files that git would add were 213576 KB. After it, they were 8036 KB. The largest file was then `models/sd-turbo/benchmark/results/2026-11-18-girl-watermelon-grid.png`, at 1336 KB. A search for key and token patterns in those files found nothing.

Commands after the rename, in `/Users/teimurgasanov/Projects/infer`. Device: Mac.

```
50 packages installed [3.16s]
```

`bun run typecheck` printed "Exited with code 0" 7 times. `bun run build`:

```
@infer/sd-turbo build: built dist/baseline.js, dist/baseline.js.map
@infer/site build: built 4 pages, 13 files, 5 fonts
```

`cargo test --release --test silu --test tokenizer --test pipeline`, in `models/sd-turbo/engine/`. Device: Mac native.

```
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.30s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

The landing prompt in Mac Chrome, before the rename: "a red fox in fresh snow, wildlife photo" went to `/sd-turbo/#prompt=…`, and the run started by itself. Device: Mac Chrome.

```
{"steps":["done ","done 301 ms","done 1.88 s","done 475 ms"],"caption":"fig. 1 “a red fox in fresh snow, wildlife photo”, seed 42. 4.99 s on this device.","hint":"Done. Change the prompt or the seed for a new image.","scroll":2320,"inner":2320}
```

Review: a separate reviewer agent checked screenshots at 1440 px and 390 px against the contract. Round 1 found 8 material fixes, with the disposition "fix". Round 2 scored 6 of them resolved and 2 partial. It also found 2 regressions: the store button was blue, and the reverse page showed inside the empty figure. Round 3 left one partial item, the reverse page at 390 px. The layer was then rebuilt as grid columns. Measured in Mac Chrome in a 390 × 844 frame, the two columns were at x 46 to 195 and 219 to 368, each 750 px high. No reviewer saw this last change.

Failed captures: headless Chrome cannot be narrower than about 500 px, so the 390 px captures used a 390 px iframe. Headless Chrome also stopped the entry page at the OPFS or WebGPU check, so the entry page at 1440 px came from Mac Chrome. The headless captures misplaced the reverse-page columns. The measurement above replaced them.

Not measured on the iPhone.

Conclusion: the site and the design system are the dictionary world, the folder and the packages are named infer, and the repository is ready to push.

Push: `gh repo create infer --private --source . --remote origin --push` created https://github.com/teimurjan/infer (private) and pushed commit d2eae64 to `main`.

Next step: the user opens the site on the iPhone with `bun run serve:lan` and looks at the landing, the entry page and the tests page.
