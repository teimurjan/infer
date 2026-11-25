# Inference in Safari on iPhone

This repository is a collection of model engines that run in Safari on iPhone with WebGPU. SD-Turbo is the first model. Each model is one folder in `models/`.

## Layout

- `models/<model>/`: one model, with its engine, typed client (`client/`), test suite (`client/tests/`), Python scripts, data and log. SD-Turbo is `models/sd-turbo/`.
- `apps/site/`: the website "infer". The landing lists every inference that runs today. Each model has an entry page (the public demo) and developer pages (tests, benchmark). Built with Bun into `apps/site/dist/`.
- `packages/ui/`: the design system of infer, a pocket dictionary on bible paper: tokens, self-hosted fonts (Archivo, Source Serif 4), styles and small DOM components. Pages use it, and no UI framework. The product and design records are `apps/site/PRODUCT.md` and `apps/site/DESIGN.md`.
- `packages/engine-core/`: the Rust base that every model engine shares: the GPU connection, kernels ready to run, memory accounting, timed submits, the profiler, the benchmarks, the activation planner and the weight files. A model engine adds its own kernels and stages.
- `packages/browser/`: browser helpers that many pages share, the run events (`run-event.ts`, `report.ts`) and the test suite type.
- `packages/harness/`: drives Safari on the connected iPhone with safaridriver, and waits for the result of a page.
- `packages/server/`: the development web server, for the Mac and for the iPhone. It saves the run events in `runs/`.
- `packages/probes/`: pages that measure the limits of the phone.
- `crash-record/`: the evidence for the phone limits below.

A new model adds `models/<model>/` with a client, a test suite and a dictionary entry (`client/entry.ts`), lists the entry in `apps/site/src/shell/entries.ts`, and adds its pages in `apps/site/src/<model>/` and `apps/site/pages/<model>/`.

## Stack

- Write each engine in Rust with wgpu, for example `models/sd-turbo/engine/`. The same crate runs natively on the Mac and compiles to WASM for the browser. Code that more than one engine needs goes into `packages/engine-core/`.
- Write every GPU kernel by hand in WGSL.
- Do not put an ML framework inside an engine. The onnxruntime-web code in `models/sd-turbo/baseline/` is the baseline that we measured. It is not part of the engine.
- Run every script with `bun` or `uv`. TypeScript scripts and pages use Bun. Python scripts use `uv run`, with the packages in the script header.

## Commands

Run these in the repository folder:

- `bun install`: install the packages of every workspace.
- `bun run build`: build every workspace.
- `bun run typecheck`: check the types of every workspace.
- `bun run serve`: serve the repository at `http://127.0.0.1:8765/`.
- `bun run serve:lan`: serve it over HTTPS on port 8443, for the iPhone.
- `bun run cert`: make the HTTPS certificate for the Wi-Fi address of the Mac.
- `bun run phone <model>:<test>`: run one test page in Safari on the connected iPhone, for example `bun run phone sd-turbo:golden-image`. It prints the lines of the page and exits with 0 only if the test passed.

Use the harness to measure on the phone when the user has connected it. Still ask the user before a long run, and do not guess a number that a run did not print.

## How to work with the user

- The user is learning. Keep each file small.
- Comment every line that is not obvious, in plain words. In this project, this rule replaces the global "no comments" rule.
- Do one step per prompt. Then stop. Tell the user the exact command to run and what they must see.
- Do not say that something works unless a command proves it.
- Do not guess a number about the phone. Ask the user to measure it.
- Log every step and its outputs in a `PROGRESS.md`. The rules are in @AGENTS.md.

## Limits measured on iPhone 16 Pro Max

- iOS kills the tab at about 1920 MB in total. The JS heap, WASM memory, and GPU memory share this one budget.
- One GPU buffer can be at most about 1073 MB.
- If one GPU submit runs longer than 0.5 s, the device is lost. Keep every submit under 0.25 s.

The evidence for these limits is in `crash-record/2026-11-15-iphone16promax.md`.
