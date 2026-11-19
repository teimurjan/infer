// The inputs that both runtimes get: the golden token ids, noise, sigma and timestep.

import { readGoldenInputs, type Inputs } from '../client/golden.ts';

// The 77 token ids of "a photo of a cat sitting on a wooden table" (PROMPT in python/dump_golden.py), the [1, 4, 64, 64] noise with
// unit variance, the noise level and the timestep of the one SD-Turbo step.
export type GoldenInputs = Inputs;

// Read the golden inputs with the same code as the developer page.
export const readInputs = (): Promise<GoldenInputs> => readGoldenInputs();
