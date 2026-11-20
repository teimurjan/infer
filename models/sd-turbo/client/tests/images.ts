// Image tests: whole SD-Turbo images through the three stages, with the golden prompt and seed.

import type { TestCase, TestContext } from '@infer/browser/test-suite';
import type { Engine, PromptTokenizer } from '../../engine/pkg/engine.js';
import { startingNoise } from '../../engine/pkg/engine.js';
import { openCache } from '../cache.ts';
import { loadTokenizer } from '../engine.ts';
import { GOLDEN_PROMPT, GOLDEN_SEED, readGolden, readGoldenInputs, type Inputs } from '../golden.ts';
import { compareBytes, toRgba, type ByteCompare } from '../image.ts';
import { DEFAULT_UNET_FILE, TEXT_FILE, generate, type PipelineEvent } from '../pipeline.ts';
import { SIGMA, TIMESTEP } from '../schedule.ts';
import { session } from './session.ts';

// The golden set of the two packed files: `uv run python/dump_packed_golden.py unet_q8 text_encoder_q8` writes golden/text_encoder_q8_unet_q8.
const GOLDEN_SET = `${TEXT_FILE}_${DEFAULT_UNET_FILE}`;
// The largest byte difference that passes. The Mac Chromium result before the restructure was "by at most 1".
const BYTE_BOUND = 1;
// The number of images of "Make 20".
const IMAGES_IN_A_ROW = 20;

// Everything an image test needs: the engine, the tokenizer, the golden inputs and the golden image bytes.
type Setup = { engine: Engine; wasmMemory: WebAssembly.Memory; tokenizer: PromptTokenizer; inputs: Inputs; expected: Uint8Array };

// Load the setup, and print the two checks that come before an image: the tokenizer ids and the seed noise.
async function setup(say: TestContext['say']): Promise<Setup> {
    const [{ engine, wasmMemory }, tokenizer, golden, expected] = await Promise.all([session(), loadTokenizer(), readGoldenInputs(), readGolden(`${GOLDEN_SET}/taesd_image.safetensors`)]);
    const ids = tokenizer.encode(GOLDEN_PROMPT);
    const sameIds = ids.every((id, index) => id === golden.ids[index]);
    say(`prompt ${JSON.stringify(GOLDEN_PROMPT)}: ${sameIds ? 'the same as' : 'not'} the golden ids`);
    const noise = startingNoise(GOLDEN_SEED);
    const noiseDifference = noise.reduce((largest, value, index) => Math.max(largest, Math.abs(value - golden.noise[index])), 0);
    say(`seed ${GOLDEN_SEED} noise: largest difference from golden/starting_noise ${noiseDifference}`);
    // The demo uses the constants of schedule.ts instead of the golden files, so they must match.
    say(`schedule: sigma ${golden.sigma}, timestep ${golden.timestep}; ${golden.sigma === SIGMA && golden.timestep === TIMESTEP ? 'the same as' : 'not'} client/schedule.ts`);
    if (!sameIds) throw new Error('the tokenizer ids differ from the golden ids');
    return { engine, wasmMemory, tokenizer, inputs: { ...golden, ids, noise }, expected: toRgba(expected.values) };
}

// The log line of each pipeline event, in the format of the old generate.html. Progress events have no line.
function eventLine(event: PipelineEvent): string | null {
    if (event.kind === 'loaded') return `${event.result.line}; from ${event.result.from} in ${event.result.seconds.toFixed(2)} s`;
    if (event.kind === 'stage-end') return event.line;
    return null;
}

// One golden image. `say` gets the stage lines, or nothing if it is null.
async function goldenImage(ready: Setup, say: TestContext['say'] | null): Promise<{ seconds: number; compare: ByteCompare }> {
    const image = await generate(ready.engine, ready.inputs, await openCache(), event => {
        const line = eventLine(event);
        if (line && say) say(line);
    });
    return { seconds: image.seconds, compare: compareBytes(image.rgba, ready.expected) };
}

// The compare line, the same as in the old generate.html.
const compareLine = ({ differing, total, largest }: ByteCompare): string => `image bytes: ${differing} of ${total} differ from golden/${GOLDEN_SET}/taesd_image, by at most ${largest}`;

export const goldenImageTest: TestCase = {
    id: 'golden-image',
    title: 'Golden image',
    description: `The golden prompt with seed ${GOLDEN_SEED}, byte by byte against the PyTorch image, at most ${BYTE_BOUND} apart.`,
    async run({ say, metric }) {
        const ready = await setup(say);
        const { seconds, compare } = await goldenImage(ready, say);
        say(`total: ${seconds.toFixed(2)} s`);
        say(compareLine(compare));
        metric('image seconds', seconds, 's');
        metric('differing bytes', compare.differing, 'bytes');
        return { passed: compare.largest <= BYTE_BOUND, summary: compareLine(compare) };
    },
};

export const makeTwentyTest: TestCase = {
    id: 'make-20',
    title: 'Make 20',
    description: `${IMAGES_IN_A_ROW} golden images in a row. Each must free all GPU memory and keep every submit under 250 ms.`,
    async run({ say, metric }) {
        const ready = await setup(say);
        const started = performance.now();
        // Forget the submits before the row.
        ready.engine.takeLongestSubmit();
        for (let image = 1; image <= IMAGES_IN_A_ROW; image++) {
            const { seconds, compare } = await goldenImage(ready, null);
            // The longest submit of this image only. The engine stops the run at 250 ms.
            const longestMs = ready.engine.takeLongestSubmit() * 1000;
            const live = ready.engine.liveBytes();
            say(`image ${image} of ${IMAGES_IN_A_ROW}: ${seconds.toFixed(2)} s; longest submit ${longestMs.toFixed(1)} ms; live ${live} bytes; WASM memory ${ready.wasmMemory.buffer.byteLength} bytes; ${compare.differing} image bytes differ, by at most ${compare.largest}`);
            metric(`image ${image} seconds`, seconds, 's');
            metric(`image ${image} longest submit`, longestMs, 'ms');
            // Live bytes after an image mean a leak, and the next images would run out of memory.
            if (live !== 0) return { passed: false, summary: `image ${image} left ${live} live bytes` };
        }
        const total = (performance.now() - started) / 1000;
        say(`${IMAGES_IN_A_ROW} images in ${total.toFixed(2)} s`);
        return { passed: true, summary: `${IMAGES_IN_A_ROW} images in ${total.toFixed(2)} s` };
    },
};

export const profileTest: TestCase = {
    id: 'profile',
    title: 'Kernel profile',
    description: 'One golden image with a timer on every dispatch, then the table of time per kernel.',
    async run({ say }) {
        const ready = await setup(say);
        // ?clock=wall on the page address forces the CPU clock, as on the old page.
        say(await ready.engine.startProfile(new URLSearchParams(location.search).get('clock') === 'wall') as string);
        try {
            await goldenImage(ready, say);
        } finally {
            // The table also ends the profile, so the next image runs without it.
            // One event for the whole table, because separate events can arrive out of order.
            say(ready.engine.profileTable());
        }
        return { passed: true, summary: 'profile table printed' };
    },
};
