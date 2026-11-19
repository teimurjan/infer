// The parts of the onnxruntime-web SD-Turbo example that its page (index.ts) and the benchmark page share:
// the three models with their options, the session options, and the scheduler math.
// Moved here from index.ts without a change, so both pages run the baseline with the same settings.

import * as ort from 'onnxruntime-web/webgpu';

// WebNN accepts only these device names.
export type DeviceType = 'cpu' | 'gpu' | 'npu';

// The load order matters: it is the order in which memory is used.
export const MODEL_NAMES = ['unet', 'text_encoder', 'vae_decoder'] as const;
export type ModelName = typeof MODEL_NAMES[number];

export type ModelSpec = {
    url: string;
    // Download size in MB, used only for the "downloading ..." message.
    size: number;
    opt: ort.InferenceSession.SessionOptions;
};

export const MODELS: Record<ModelName, ModelSpec> = {
    unet: {
        url: 'unet/model.onnx', size: 640,
        // should have 'steps: 1' but will fail to create the session
        opt: { freeDimensionOverrides: { batch_size: 1, num_channels: 4, height: 64, width: 64, sequence_length: 77 } },
    },
    text_encoder: {
        url: 'text_encoder/model.onnx', size: 1700,
        // should have 'sequence_length: 77' but produces a bad image
        opt: { freeDimensionOverrides: { batch_size: 1 } },
    },
    vae_decoder: {
        url: 'vae_decoder/model.onnx', size: 95,
        opt: { freeDimensionOverrides: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } },
    },
};

// Session options for every model. Each model adds its own options on top.
export function sessionOptions(provider: string, device: DeviceType): ort.InferenceSession.SessionOptions {
    const base: ort.InferenceSession.SessionOptions = {
        enableMemPattern: false,
        enableCpuMemArena: false,
        extra: {
            session: {
                disable_prepacking: '1',
                use_device_allocator_for_initializers: '1',
                use_ort_model_bytes_directly: '1',
                use_ort_model_bytes_for_initializers: '1',
            },
        },
    };
    switch (provider) {
        case 'webgpu':
            if (!('gpu' in navigator)) {
                throw new Error('webgpu is NOT supported');
            }
            // Keep the text encoder output on the GPU, so the unet reads it without a copy.
            return { ...base, executionProviders: [{ name: provider }], preferredOutputLocation: { last_hidden_state: 'gpu-buffer' } };
        case 'webnn':
            if (!('ml' in navigator)) {
                throw new Error('webnn is NOT supported');
            }
            return { ...base, executionProviders: [{ name: 'webnn', deviceType: device, powerPreference: 'default' }] };
        default:
            return { ...base, executionProviders: [{ name: provider }] };
    }
}

// Noise level of the single scheduler step that sd-turbo uses.
export const SIGMA = 14.6146;
const GAMMA = 0;
// The VAE expects latents divided by this number.
const VAE_SCALING_FACTOR = 0.18215;

// Picks one named output of a model run. A missing output is a bug, so we stop.
export function outputTensor(outputs: ort.InferenceSession.ReturnType, name: string): ort.Tensor {
    const tensor = outputs[name];
    if (!tensor) {
        throw new Error(`model output ${name} is missing`);
    }
    return tensor;
}

// The math below works on 32-bit floats only, so any other kind of tensor is a bug.
export function float32Data(tensor: ort.Tensor): Float32Array {
    if (!(tensor.data instanceof Float32Array)) {
        throw new Error(`expected a float32 tensor, got ${tensor.type}`);
    }
    return tensor.data;
}

/*
 * scale the latents
 */
export function scaleModelInputs(tensor: ort.Tensor, sigma: number): ort.Tensor {
    const input = float32Data(tensor);
    const output = new Float32Array(input.length);
    const divisor = (sigma ** 2 + 1) ** 0.5;
    for (let i = 0; i < input.length; i++) {
        output[i] = input[i] / divisor;
    }
    return new ort.Tensor(output, tensor.dims);
}

/*
 * Poor mens EulerA step
 * Since this example is just sd-turbo, implement the absolute minimum needed to create an image
 * Maybe next step is to support all sd flavors and create a small helper model in onnx can deal
 * much more efficient with latents.
 */
export function step(modelOutput: ort.Tensor, sample: ort.Tensor, sigma: number): ort.Tensor {
    const noise = float32Data(modelOutput);
    const latents = float32Data(sample);
    const output = new Float32Array(noise.length);
    const sigmaHat = sigma * (GAMMA + 1);

    for (let i = 0; i < noise.length; i++) {
        // The model predicts the noise. Removing it gives the clean image latents.
        const predOriginalSample = latents[i] - sigmaHat * noise[i];
        const derivative = (latents[i] - predOriginalSample) / sigmaHat;
        const dt = 0 - sigmaHat;
        output[i] = (latents[i] + derivative * dt) / VAE_SCALING_FACTOR;
    }
    return new ort.Tensor(output, modelOutput.dims);
}
