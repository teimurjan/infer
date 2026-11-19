// Runs the onnxruntime-web SD-Turbo example: the same models, options and scheduler math as index.ts.
// Only the inputs differ: this page feeds the golden token ids and noise, so both runtimes get the same prompt and noise.

import * as ort from 'onnxruntime-web/webgpu';
import { MODELS, MODEL_NAMES, float32Data, outputTensor, scaleModelInputs, sessionOptions, step } from '../baseline/onnx-pipeline.ts';
import type { ModelName } from '../baseline/onnx-pipeline.ts';
import { toRgba } from '../client/image.ts';
import type { GoldenInputs } from './inputs.ts';
import { ONNX_MODELS, ORT_WASM } from './paths.ts';
import type { MakeImage } from './runners.ts';

// The latent shape: 1 image, 4 channels, 64 x 64.
const LATENT_SHAPE = [1, 4, 64, 64];

// One model file, from the browser cache "onnx" that index.ts also uses. A missing file is downloaded and cached first.
async function modelBytes(path: string): Promise<ArrayBuffer> {
    const url = `${ONNX_MODELS}/${path}`;
    const cache = await caches.open('onnx');
    const cached = await cache.match(url) ?? await cache.add(url).then(() => cache.match(url));
    if (!cached) {
        throw new Error(`${url} is not in the cache after cache.add`);
    }
    return cached.arrayBuffer();
}

// Create the three sessions, in the order of index.ts, with the settings of index.ts.
export async function openOnnx(): Promise<MakeImage> {
    // The same runtime settings as index.ts.
    ort.env.wasm.wasmPaths = ORT_WASM;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    const options = sessionOptions('webgpu', 'gpu');
    const sessions = new Map<ModelName, ort.InferenceSession>();
    for (const name of MODEL_NAMES) {
        sessions.set(name, await ort.InferenceSession.create(await modelBytes(MODELS[name].url), { ...options, ...MODELS[name].opt }));
    }
    return inputs => onnxImage(sessions, inputs);
}

// The session of one model. All three exist after openOnnx.
function session(sessions: Map<ModelName, ort.InferenceSession>, name: ModelName): ort.InferenceSession {
    const found = sessions.get(name);
    if (!found) {
        throw new Error(`${name} has no session`);
    }
    return found;
}

// One image: text encoder, UNet, the scheduler step on the CPU, VAE decoder. The steps are those of generateImage in index.ts.
async function onnxImage(sessions: Map<ModelName, ort.InferenceSession>, inputs: GoldenInputs): Promise<Uint8Array> {
    // The 77 golden ids. index.ts gets its ids from a tokenizer instead.
    const ids = new ort.Tensor('int32', Int32Array.from(inputs.ids), [1, inputs.ids.length]);
    const lastHiddenState = outputTensor(await session(sessions, 'text_encoder').run({ input_ids: ids }), 'last_hidden_state');
    try {
        // index.ts draws random noise times sigma. The golden noise has unit variance, so it is multiplied by sigma here.
        const latent = new ort.Tensor(Float32Array.from(inputs.noise, value => value * inputs.sigma), LATENT_SHAPE);
        const feed = {
            sample: scaleModelInputs(latent, inputs.sigma),
            timestep: new ort.Tensor('int64', BigInt64Array.of(BigInt(Math.round(inputs.timestep))), [1]),
            encoder_hidden_states: lastHiddenState,
        };
        const outSample = outputTensor(await session(sessions, 'unet').run(feed), 'out_sample');
        const decoded = outputTensor(await session(sessions, 'vae_decoder').run({ latent_sample: step(outSample, latent, inputs.sigma) }), 'sample');
        return toRgba(float32Data(decoded));
    } finally {
        // The text encoder output is a GPU buffer that this page owns, as in index.ts.
        lastHiddenState.dispose();
    }
}
