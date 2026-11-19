// The starting noise of a seed, the same values as torch.randn on the CPU. The engine computes it in WASM.
// The WASM build must be loaded first: openSession() in engine.ts does that.

export { startingNoise } from '../engine/pkg/engine.js';
