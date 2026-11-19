# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Engineers: web, GPU and ML engineers. They open the site, most often in Safari on an iPhone. They want to see a real model run on the phone's own GPU, and to understand how. They care about the mechanism (Rust, hand-written WGSL kernels, WebGPU) and about measured numbers. The developer of the project uses the same site for test and benchmark pages.

## Product Purpose

infer is a collection of AI model engines that run inside Safari on iPhone with WebGPU. Each model is a separate engine and gets a public demo page plus developer pages (tests, benchmark). The landing lists every inference that is available. Success: a visitor runs a model on their own phone and leaves knowing how it fits in the browser's limits.

## Positioning

No ML framework inside the engines. Each engine is Rust compiled to WASM. Every GPU kernel is written by hand in WGSL. The design follows the measured limits of one iPhone tab. The math runs on the visitor's device, not on a server.

## Operating Context

- Visitors open the site in Safari on iPhone, often over Wi-Fi. A model downloads once (SD-Turbo: about 2.24 GB) into the private storage of the browser (OPFS), and it stays there.
- The developer runs tests on an iPhone on a cable with `bun run phone`. Every run is saved on the Mac.

## Capabilities and Constraints

- Available inferences: only models that run on the phone are listed. Today that is SD-Turbo (text to 512 x 512 image, one step). Planned models are never announced before they ship.
- iPhone 16 Pro Max limits: about 1920 MB per tab for JS, WASM and GPU together. One GPU buffer can be at most about 1073 MB. Every GPU submit must stay under 0.25 s.
- Pages are vanilla TypeScript and CSS with no UI framework, to keep the JS heap small inside the tab budget.
- Must work in Safari on iPhone at 390 px width.

## Brand Commitments

- The name is "infer", lowercase, everywhere: tab titles read like "infer / sd-turbo".
- No logo. The wordmark is the word "infer" set in type.
- The user named three experimental sites they like: typesafe.ai, pauladealvaro.com, glossaryoftime.com.

## Evidence on Hand

- Measured results are in `models/sd-turbo/PROGRESS.md` and `models/sd-turbo/README.md`. Example: the golden image on the iPhone (1073 of 1048576 bytes differ, by at most 1).
- Stage times on the iPhone: 0.37 s, 2.54 s and 0.73 s.
- No testimonials, users, press or comparison claims exist. Do not invent any.

## Product Principles

- Show the model running, not a description of it.
- Every number on a page comes from a real run.
- List only what runs on the phone today.
- The mechanism is the story: limits, kernels and memory are content, not footnotes.
