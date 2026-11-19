// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run sd-turbo with webgpu in onnxruntime-web.
//

import * as ort from 'onnxruntime-web/webgpu';
import { crumb, track, trackGlobalErrors } from '@infer/browser/breadcrumbs';
import { byId } from '@infer/browser/dom';
import { errorText } from '@infer/browser/errors';
import { MODEL_NAMES, MODELS as models, SIGMA as sigma, float32Data, outputTensor, scaleModelInputs, sessionOptions, step } from './onnx-pipeline.ts';
import type { DeviceType, ModelName } from './onnx-pipeline.ts';

trackGlobalErrors();
crumb('page', 'boot', 'info', { ua: navigator.userAgent, ort: ort.env.versions?.web });

function log(message: unknown): void {
    console.log(message);
    byId('status', HTMLElement).innerText += `\n${message}`;
}

// WebNN accepts only these device names.
const DEVICE_TYPES: readonly string[] = ['cpu', 'gpu', 'npu'];
const isDeviceType = (value: string): value is DeviceType => DEVICE_TYPES.includes(value);

type Config = {
    model: string;
    provider: string;
    device: DeviceType;
    threads: number;
    images: number;
};

// Every URL argument the page accepts, with its default value.
const DEFAULT_QUERY = {
    // model: "models/onnx-sd-turbo-fp16",
    model: 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main',
    provider: 'webgpu',
    device: 'gpu',
    threads: '1',
    images: '2',
};

/*
 * get configuration from url
 */
function getConfig(): Config {
    const query: Record<string, string> = { ...DEFAULT_QUERY };
    for (const [key, value] of new URLSearchParams(location.search)) {
        // hasOwn skips names that every object has, for example "toString".
        if (!Object.hasOwn(query, key)) {
            throw new Error(`unknown argument: ${key}`);
        }
        query[key] = value;
    }
    const { model, provider, device, threads, images } = query;
    if (!isDeviceType(device)) {
        throw new Error(`unknown device: ${device}`);
    }
    return { model, provider, device, threads: parseInt(threads), images: parseInt(images) };
}

/*
 * initialize latents with random noise
 */
function randnLatents(shape: readonly number[], noiseSigma: number): Float32Array {
    // One random number from a normal distribution (mean 0, spread 1).
    function randn(): number {
        // Use the Box-Muller transform
        const u = Math.random();
        const v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    // The element count is the product of all dimensions.
    const size = shape.reduce((product, dimension) => product * dimension, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = randn() * noiseSigma;
    }
    return data;
}

// Adds the file to the browser cache, then reads it back from there.
async function addAndMatch(cache: Cache, url: string): Promise<Response> {
    await cache.add(url);
    const response = await cache.match(url);
    if (!response) {
        throw new Error(`${url} is not in the cache after cache.add`);
    }
    return response;
}

/*
 * fetch and cache model
 */
async function fetchAndCache(baseUrl: string, modelPath: string): Promise<ArrayBuffer> {
    const url = `${baseUrl}/${modelPath}`;
    try {
        const cache = await caches.open('onnx');
        const cached = await cache.match(url);
        const response = cached ?? await addAndMatch(cache, url);
        const from = cached ? 'cache' : 'network';
        log(`${modelPath} (${from})`);
        crumb(modelPath, 'fetch', 'source', { from });
        return await response.arrayBuffer();
    } catch (error) {
        // The Cache API failed, so download the file without caching it.
        log(`${modelPath} (network)`);
        crumb(modelPath, 'fetch', 'cache-fallback', { error: errorText(error) });
        return await fetch(url).then(response => response.arrayBuffer());
    }
}

const config = getConfig();

// Sessions that loaded without an error.
const sessions = new Map<ModelName, ort.InferenceSession>();

const opt = sessionOptions(config.provider, config.device);

/*
 * load models used in the pipeline
 */
async function loadModels(): Promise<void> {
    const cache = await caches.open('onnx');
    let missingMb = 0;
    for (const name of MODEL_NAMES) {
        const cached = await cache.match(`${config.model}/${models[name].url}`);
        if (cached === undefined) {
            missingMb += models[name].size;
        }
    }
    log(missingMb > 0 ? `downloading ${missingMb} MB from network ... it might take a while` : 'loading...');
    for (const name of MODEL_NAMES) {
        const model = models[name];
        try {
            const start = performance.now();
            const modelBytes = await track(name, 'fetch', () => fetchAndCache(config.model, model.url),
                bytes => ({ mb: +(bytes.byteLength / 1e6).toFixed(1) }));
            const session = await track(name, 'session', () => ort.InferenceSession.create(modelBytes, { ...opt, ...model.opt }),
                created => ({ inputs: created.inputNames, outputs: created.outputNames }));
            sessions.set(name, session);
            log(`${model.url} in ${(performance.now() - start).toFixed(1)}ms`);
        } catch (error) {
            log(`${model.url} failed, ${error}`);
        }
    }
    log('ready.');
}

// Where onnxruntime finds its .wasm files: the dist folder one level up, which build.ts fills.
ort.env.wasm.wasmPaths = new URL('../dist/', location.href).href;
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

let tokenizer: Tokenizer | undefined;
const ranOnce = new Set<ModelName>();

/*
 * run a model, with breadcrumbs around its first run
 */
function runModel(name: ModelName, feed: ort.InferenceSession.FeedsType): Promise<ort.InferenceSession.ReturnType> {
    const session = sessions.get(name);
    if (!session) {
        return Promise.reject(new Error(`${name} did not load`));
    }
    const run = () => session.run(feed);
    if (ranOnce.has(name)) {
        return run();
    }
    ranOnce.add(name);
    return track(name, 'first-run', run, outputs => ({ outputs: Object.keys(outputs) }));
}

let loading: Promise<void> | undefined;
const text = byId('user-input', HTMLTextAreaElement);

text.value = 'Paris with the river in the background';

// Event listener for Ctrl + Enter
text.addEventListener('keydown', event => {
    if (event.ctrlKey && event.key === 'Enter') {
        generateImage();
    }
});
byId('send-button', HTMLButtonElement).addEventListener('click', () => generateImage());

/*
 * draw an image from tensor
 */
function drawImage(tensor: ort.Tensor, imageNr: number): void {
    const pixels = float32Data(tensor);
    // The VAE gives values from -1 to 1. Map them to 0..1 and cut off the rest, in place.
    for (let i = 0; i < pixels.length; i++) {
        pixels[i] = Math.min(1, Math.max(0, pixels[i] / 2 + 0.5));
    }
    // NCHW: batch, then color channel, then rows, then columns.
    const imageData = tensor.toImageData({ tensorLayout: 'NCHW', format: 'RGB' });
    const canvas = byId(`img_canvas_${imageNr}`, HTMLCanvasElement);
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const context = canvas.getContext('2d');
    if (!context) {
        throw new Error('canvas 2d context is not available');
    }
    context.putImageData(imageData, 0, 0);
    byId(`img_div_${imageNr}`, HTMLDivElement).style.opacity = '1';
}

async function generateImage(): Promise<void> {
    try {
        byId('status', HTMLElement).innerText = 'generating ...';

        if (tokenizer === undefined) {
            tokenizer = await track('tokenizer', 'fetch', () => AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16'));
            tokenizer.pad_token_id = 0;
        }
        await loading;

        for (let j = 0; j < config.images; j++) {
            byId(`img_div_${j}`, HTMLDivElement).style.opacity = '0.5';
        }

        const { input_ids: inputIds } = await tokenizer(text.value, { padding: true, max_length: 77, truncation: true, return_tensor: false });

        // text-encoder
        let start = performance.now();
        const textOutputs = await runModel('text_encoder', { input_ids: new ort.Tensor('int32', inputIds, [1, inputIds.length]) });
        const lastHiddenState = outputTensor(textOutputs, 'last_hidden_state');

        let perfInfo = [`text_encoder: ${(performance.now() - start).toFixed(1)}ms`];

        for (let j = 0; j < config.images; j++) {
            // 1 image, 4 latent channels, 64 x 64. The VAE scales this up to 512 x 512 pixels.
            const latentShape = [1, 4, 64, 64];
            const latent = new ort.Tensor(randnLatents(latentShape, sigma), latentShape);
            const latentModelInput = scaleModelInputs(latent, sigma);

            // unet
            start = performance.now();
            const feed = {
                sample: latentModelInput,
                // sd-turbo runs one step, at the last timestep (999).
                timestep: new ort.Tensor('int64', [999n], [1]),
                encoder_hidden_states: lastHiddenState,
            };
            const outSample = outputTensor(await runModel('unet', feed), 'out_sample');
            perfInfo.push(`unet: ${(performance.now() - start).toFixed(1)}ms`);

            // scheduler
            const newLatents = step(outSample, latent, sigma);

            // vae_decoder
            start = performance.now();
            const sample = outputTensor(await runModel('vae_decoder', { latent_sample: newLatents }), 'sample');
            perfInfo.push(`vae_decoder: ${(performance.now() - start).toFixed(1)}ms`);

            drawImage(sample, j);
            log(perfInfo.join(', '));
            perfInfo = [];
        }
        // this is a gpu-buffer we own, so we need to dispose it
        lastHiddenState.dispose();
        log('done');
    } catch (error) {
        log(error);
    }
}

async function hasFp16(): Promise<boolean> {
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            crumb('gpu', 'adapter', 'fail', { error: 'requestAdapter returned null' });
            return false;
        }
        const { vendor, architecture, description } = adapter.info;
        crumb('gpu', 'adapter', 'info', {
            vendor, architecture, description,
            f16: adapter.features.has('shader-f16'),
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        });
        return adapter.features.has('shader-f16');
    } catch (error) {
        crumb('gpu', 'adapter', 'fail', { error: errorText(error) });
        return false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    hasFp16().then(fp16 => {
        if (fp16) {
            loading = loadModels();
        } else {
            log("Your GPU or Browser doesn't support webgpu/f16");
        }
    });
});
