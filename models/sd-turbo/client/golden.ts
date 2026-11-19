// Reads the golden tensors that the Python scripts saved, in the safetensors format.

import { fetchBytes } from './fetch.ts';
import { urls } from './urls.ts';

export type Tensor = { shape: number[]; values: Float32Array };

// Read the one float32 tensor of a golden file. `path` is relative to golden/.
// File layout: 8 bytes with the header length (lowest byte first), the JSON header, then the raw values.
export async function readGolden(path: string): Promise<Tensor> {
    const bytes = await fetchBytes(urls.golden(path));
    // The header length. Golden headers are far below 2^53 bytes, so a Number holds it exactly.
    const headerLength = Number(new DataView(bytes).getBigUint64(0, true));
    // The JSON header: tensor name -> {dtype, shape, data_offsets}. Each golden file has one tensor, plus maybe __metadata__.
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 8, headerLength))) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
    const entry = Object.entries(header).find(([name]) => name !== '__metadata__')?.[1];
    if (entry?.dtype !== 'F32') throw new Error(`${path}: expected an F32 tensor, got ${entry?.dtype}`);
    // data_offsets count from the end of the header. slice copies, so the Float32Array is aligned.
    const [start, end] = entry.data_offsets.map(offset => 8 + headerLength + offset);
    return { shape: entry.shape, values: new Float32Array(bytes.slice(start, end)) };
}

// The inputs of one image: 77 token ids, the [1, 4, 64, 64] starting noise, the noise level and the timestep.
export type Inputs = { ids: Uint32Array; noise: Float32Array; sigma: number; timestep: number };

// The inputs of the golden run. The prompt is PROMPT in python/dump_golden.py: "a photo of a cat sitting on a wooden table".
export async function readGoldenInputs(): Promise<Inputs> {
    const [ids, noise, sigma, timestep] = await Promise.all(
        ['float32/text_input_ids.safetensors', 'starting_noise.safetensors', 'float32/sigma.safetensors', 'float32/timestep.safetensors'].map(readGolden),
    );
    // The ids are float32 whole numbers in the golden file.
    return { ids: Uint32Array.from(ids.values), noise: noise.values, sigma: sigma.values[0], timestep: timestep.values[0] };
}

// The golden prompt and seed.
export const GOLDEN_PROMPT = 'a photo of a cat sitting on a wooden table';
export const GOLDEN_SEED = 42;
