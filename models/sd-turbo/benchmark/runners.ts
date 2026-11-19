// The two runtimes behind one shape: open loads everything that stays loaded, and gives a function that makes one image.

import { openEngine } from './engine-runner.ts';
import type { GoldenInputs } from './inputs.ts';
import { openOnnx } from './onnx-runner.ts';

// Make one 512 x 512 image and give its RGBA bytes.
export type MakeImage = (inputs: GoldenInputs) => Promise<Uint8Array>;

// The names of the runtimes, in run order. The engine goes first, so its results are saved before onnxruntime-web can kill the tab.
export const RUNTIMES = ['engine', 'onnx'] as const;
export type Runtime = typeof RUNTIMES[number];

// The open function of each runtime.
export const OPEN: Record<Runtime, () => Promise<MakeImage>> = { engine: openEngine, onnx: openOnnx };

// The name of each runtime in the table.
export const LABEL: Record<Runtime, string> = { engine: 'this engine (Rust + WGSL)', onnx: 'onnxruntime-web example' };
