// One SD-Turbo image in three stages. Each stage loads its weight file just before it runs and frees it after,
// because the text encoder and the UNet together do not fit in the memory of one iPhone tab.

import type { Engine } from '../engine/pkg/engine.js';
import { fetchBytes } from './fetch.ts';
import type { Inputs } from './golden.ts';
import { urls } from './urls.ts';
import { loadWeights, type LoadProgress, type LoadResult } from './weights.ts';

// The text encoder file: every matrix at 8 bits (quant/text_plan_8bit.json), 342 MB in place of 1362 MB at float32.
export const TEXT_FILE = 'text_encoder_q8';
// The UNet file: 281 of 282 layers at 8 bits (quant/quant_plan_8bit.json). "Make 20" passed on the iPhone with it.
export const DEFAULT_UNET_FILE = 'unet_q8';
// The small image decoder. It is read whole, not streamed.
export const DECODER_FILE = 'taesd';

// The three stages, in order.
export type Stage = 'text-encoder' | 'unet' | 'decoder';

// What the pipeline tells its page while it runs.
export type PipelineEvent =
    | { kind: 'load'; file: string; progress: LoadProgress }
    | { kind: 'loaded'; file: string; result: LoadResult }
    | { kind: 'stage-start'; stage: Stage }
    | { kind: 'stage-end'; stage: Stage; seconds: number; line: string };

export type Image = { rgba: Uint8Array; seconds: number };

// The decoder result from the engine: its report line and the 512 x 512 RGBA bytes.
type Decoded = { report: string; rgba: Uint8Array };

// Make one image from `inputs`. `folder` is the OPFS cache, or null. onEvent gets every event.
export async function generate(engine: Engine, inputs: Inputs, folder: FileSystemDirectoryHandle | null, onEvent: (event: PipelineEvent) => void, unetFile = DEFAULT_UNET_FILE): Promise<Image> {
    const started = performance.now();
    // Stream one weight file into the engine, with its progress.
    const load = async (file: string): Promise<void> => {
        const result = await loadWeights(engine, folder, file, progress => onEvent({ kind: 'load', file, progress }));
        onEvent({ kind: 'loaded', file, result });
    };
    // Run one stage with its clock. The engine returns the report line of the stage.
    const stage = async <Result>(name: Stage, run: () => Promise<Result>, line: (result: Result) => string): Promise<Result> => {
        onEvent({ kind: 'stage-start', stage: name });
        const stageStarted = performance.now();
        const result = await run();
        onEvent({ kind: 'stage-end', stage: name, seconds: (performance.now() - stageStarted) / 1000, line: line(result) });
        return result;
    };
    // Stage 1: the text encoder turns 77 token ids into the text embeddings. It frees its file.
    await load(TEXT_FILE);
    await stage('text-encoder', () => engine.runTextEncoder(TEXT_FILE, inputs.ids) as Promise<string>, line => line);
    // Stage 2: the UNet predicts the noise in one step, and the scheduler removes it. It frees its file.
    await load(unetFile);
    await stage('unet', () => engine.runUnet(unetFile, inputs.noise, inputs.sigma, inputs.timestep) as Promise<string>, line => line);
    // Stage 3: the decoder turns the 4 x 64 x 64 latent into the 512 x 512 image.
    const decoded = await stage('decoder', async () => {
        const [index, bytes] = await Promise.all([fetchBytes(`${urls.weight(DECODER_FILE)}.json`), fetchBytes(`${urls.weight(DECODER_FILE)}.bin`)]);
        return engine.runDecoder(new TextDecoder().decode(index), new Uint8Array(bytes)) as Promise<Decoded>;
    }, result => result.report);
    return { rgba: decoded.rgba, seconds: (performance.now() - started) / 1000 };
}
