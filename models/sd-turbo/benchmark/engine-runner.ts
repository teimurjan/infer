// Runs the engine of this project: the Rust and WGSL build in engine/pkg, with the three stages of client/pipeline.ts.

import { openCache } from '../client/cache.ts';
import { openSession } from '../client/engine.ts';
import { generate } from '../client/pipeline.ts';
import type { MakeImage } from './runners.ts';

// Load the WASM build and open the GPU. The weights load later, in each image, because each stage frees its file
// (the text encoder and the UNet together do not fit the iPhone tab budget). That is the engine's normal path.
export async function openEngine(): Promise<MakeImage> {
    const { engine } = await openSession();
    // The OPFS cache folder of the weights, or null.
    const folder = await openCache();
    // One image: the three stages, each with its weight file from the cache.
    return async inputs => (await generate(engine, inputs, folder, () => {})).rgba;
}
