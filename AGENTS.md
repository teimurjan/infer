# Agent instructions

This repository builds model engines in Rust and WGSL that run in Safari on iPhone. The project rules are in `CLAUDE.md`. This file adds one rule: every step goes into a progress log, `PROGRESS.md`, with its outputs.

## Which log

- A step that changes one model goes into the log of that model, for example `models/sd-turbo/PROGRESS.md`.
- A step that changes the shared parts goes into `PROGRESS.md` at the repository root. The shared parts are `packages/`, `apps/`, the workspace files and the layout.
- A step that changes both goes into the root log, and the model log gets a short entry that names the root entry.

## Log every step in PROGRESS.md

A step is one change that the user asked for, or one measurement that the user sent. Add the entry before you tell the user that the step is done.

Each entry has these parts, in this order:

1. A heading with the date and a short name: `## YYYY-MM-DD: <step>`.
2. The files that the step added or changed.
3. The design choices and the reason for each one.
4. The exact command, or the page, that produced each output.
5. The device and runtime of each output: Mac native, Mac Chromium, or iPhone. The names are in the table at the top of `models/sd-turbo/PROGRESS.md`.
6. The output lines in a code block, copied unchanged.
7. Each calculation from the output: the formula, the numbers that go into it, and the result.
8. What failed or was rejected, with its output and the reason.
9. The conclusion and the next step.

## Rules for entries

- Copy output lines exactly. Do not round them, shorten them, or join lines from different runs.
- If the user did not send an iPhone number, do not write one. If the step has no iPhone run yet, write "not measured on the iPhone".
- Do not change or delete old entries. If a new result shows that an old entry is wrong, add a new entry that names the old one and gives the correction.
- Record failed runs, deliberate-bug (mutation) checks, and rejected designs, with their numbers.
- If the output format changes, write the old and the new format in the entry.
- Add new entries at the end of the file.
- Follow the writing rules of the project: short sentences, active voice, no contractions.
