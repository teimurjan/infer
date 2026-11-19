# infer

AI models that run on the GPU of an iPhone, inside one Safari tab. Each engine is Rust compiled to WASM, with every GPU kernel written by hand in WGSL. No ML framework runs inside an engine.

The first model is SD-Turbo: a text prompt in, a 512 × 512 image out, in one diffusion step.

## Results

All numbers come from logged runs on an iPhone 16 Pro Max in Safari 26.6.2. The full logs are in `models/sd-turbo/PROGRESS.md`.

| Run | Result |
| --- | --- |
| Golden image, 8-bit UNet, 2026-11-19 | text encoder 0.37 s, UNet 2.54 s, TAESD 0.73 s |
| Same image against PyTorch | 1073 of 1048576 bytes differ, by at most 1 |
| 20 images in a row, 8-bit UNet | 20 of 20, 0 live GPU bytes after each image |
| Benchmark, 4-bit/8-bit UNet, 2026-11-18 | 10 of 10 runs, cold 5.36 s, warm 5.64 s, peak 1608.2 MB |
| onnxruntime-web example, same benchmark | 0 of 2 runs: iOS closed the tab, then the download failed |

The engine works inside these iPhone limits, which `crash-record/2026-11-15-iphone16promax.md` measured:

- One tab gets about 1920 MB for the JS heap, WASM memory and GPU buffers together.
- One GPU buffer holds at most about 1073 MB.
- A GPU submit that runs past 0.5 s loses the device. The engine keeps every submit under 0.25 s.

## How it fits

- The weights are 2.24 GB: a float32 text encoder (1.36 GB), an 8-bit UNet (866 MB) and TAESD (10 MB). They download once into OPFS, the private storage of the page.
- The text encoder and the UNet do not fit in one tab together. Each stage streams its weights to the GPU, runs, and frees them before the next stage loads.
- Golden tensors from PyTorch check every kernel, every stage and the final image.

## Layout

| Path | Contents |
| --- | --- |
| `models/sd-turbo/` | The engine (`engine/`), a typed client and tests (`client/`), Python scripts, the onnxruntime-web baseline and the benchmark |
| `apps/site/` | The website: a landing with every model that runs today, an entry page per model, test and benchmark pages |
| `packages/ui/` | The design system of the site, with no UI framework |
| `packages/harness/` | Runs a test page in Safari on an iPhone on a cable, through `safaridriver` |
| `packages/server/` | The development server. It saves every test run in `runs/` |
| `packages/probes/` | The pages that measured the iPhone limits |

## Run it

You need Bun, uv, Rust with the `wasm32-unknown-unknown` target, and wasm-pack. The steps that make the weights and the golden files are in `models/sd-turbo/README.md`.

```sh
bun install
bun run build
bun run serve
```

Then open `http://127.0.0.1:8765/` in a browser with WebGPU.

For the iPhone, the Mac serves HTTPS with a local certificate. Run `bun run cert` once, then `bun run serve:lan`.

The harness runs a test on an iPhone on a cable, with no tap on the phone. Turn on Remote Automation in Safari on the phone, then run:

```sh
bun run phone sd-turbo:golden-image
```

The command prints each line of the page. If the test passed, it exits with 0.
