// Kernel tests: each one runs GPU kernels on test data from engine/testdata/ and compares with PyTorch.
// The engine prints the largest error of each case. A case fails only if the engine throws.

import type { TestCase } from '@infer/browser/test-suite';
import { attentionCaseNames, caseNames, checkAttention, checkCase, decodeTaesd, runBenchmarks, siluMaxError } from '../../engine/pkg/engine.js';
import { fetchBytes, fetchFloats } from '../fetch.ts';
import { readGolden } from '../golden.ts';
import { urls } from '../urls.ts';
import { session } from './session.ts';

// The bound of the native SiLU test (engine/tests/silu.rs): WGSL allows exp() and division to be a few steps off.
const SILU_TOLERANCE = 1e-4;
// Query rows per submit of the self attention. The old attention page took ?chunk=, with 256 as the default.
const ATTENTION_CHUNK_ROWS = 256;

// Read the float32 files of one test case folder.
const readCase = (folder: string, files: readonly string[]): Promise<Float32Array[]> => Promise.all(files.map(file => fetchFloats(urls.testdata(`${folder}/${file}.bin`))));

export const silu: TestCase = {
    id: 'silu',
    title: 'SiLU kernel',
    description: 'x * sigmoid(x) on 100003 values against PyTorch, within 1e-4.',
    async run({ say, metric }) {
        await session();
        const [input, expected] = await readCase('silu', ['input', 'expected']);
        const maxError = await siluMaxError(input, expected);
        say(`SiLU: ${input.length} values, largest error ${maxError.toExponential(6)}`);
        metric('silu largest error', maxError, '');
        return { passed: maxError <= SILU_TOLERANCE, summary: `largest error ${maxError.toExponential(6)}, bound ${SILU_TOLERANCE}` };
    },
};

export const layers: TestCase = {
    id: 'layers',
    title: 'Layer kernels',
    description: 'Each convolution and linear case of engine/testdata against PyTorch, with its speed.',
    async run({ say }) {
        await session();
        const names = caseNames();
        for (const name of names) {
            const [input, weight, bias, expected] = await readCase(name, ['input', 'weight', 'bias', 'expected']);
            say(await checkCase(name, input, weight, bias, expected));
        }
        return { passed: true, summary: `${names.length} cases ran` };
    },
};

export const attention: TestCase = {
    id: 'attention',
    title: 'Attention kernels',
    description: `Self and cross attention against PyTorch, ${ATTENTION_CHUNK_ROWS} query rows per submit.`,
    async run({ say }) {
        await session();
        const names = attentionCaseNames();
        for (const name of names) {
            const [query, key, value, expected] = await readCase(name, ['query', 'key', 'value', 'expected']);
            say(await checkAttention(name, ATTENTION_CHUNK_ROWS, query, key, value, expected));
        }
        return { passed: true, summary: `${names.length} cases ran` };
    },
};

export const taesd: TestCase = {
    id: 'taesd',
    title: 'TAESD decoder',
    description: 'The golden final latent through the image decoder, against the diffusers image.',
    async run({ say }) {
        await session();
        const [index, weights, latent, expected] = await Promise.all([
            fetchBytes(`${urls.weight('taesd')}.json`).then(bytes => new TextDecoder().decode(bytes)),
            fetchBytes(`${urls.weight('taesd')}.bin`).then(bytes => new Uint8Array(bytes)),
            readGolden('float32/final_latent.safetensors'),
            readGolden('float32/taesd_image.safetensors'),
        ]);
        const decoded = await decodeTaesd(index, weights, latent.values, expected.values);
        say(decoded.report);
        // The last line has the total time, which is the summary.
        return { passed: true, summary: decoded.report.split('\n').at(-1) ?? '' };
    },
};

export const bench: TestCase = {
    id: 'bench',
    title: 'Copy and math speed',
    description: 'Memory copy in GB/s and fused multiply-add in G ops/s, 30 timed submits each.',
    async run({ say }) {
        await session();
        const report = await runBenchmarks();
        // One event for the whole block, because separate events can arrive out of order.
        say(report);
        return { passed: true, summary: 'benchmarks ran' };
    },
};
