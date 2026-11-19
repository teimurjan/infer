// The weight load test: the decoder and the UNet into the engine, with the speed of each file.

import type { TestCase } from '@infer/browser/test-suite';
import { openCache } from '../cache.ts';
import { DECODER_FILE, DEFAULT_UNET_FILE } from '../pipeline.ts';
import { loadWeights } from '../weights.ts';
import { session } from './session.ts';

// Bytes in one MB, as in PROGRESS.md.
const MB = 1e6;

export const weightsTest: TestCase = {
    id: 'weights',
    title: 'Weight loading',
    description: `Streams ${DECODER_FILE} and ${DEFAULT_UNET_FILE} into GPU buffers, from the cache or the network, then frees them.`,
    async run({ say, metric }) {
        const { engine } = await session();
        const folder = await openCache();
        say(folder ? 'cache: OPFS folder weights/' : 'cache: off, this browser cannot write OPFS files from a page');
        for (const name of [DECODER_FILE, DEFAULT_UNET_FILE]) {
            const result = await loadWeights(engine, folder, name, () => {});
            say(result.line);
            say(`${name}: from ${result.from}, ${result.received} bytes in ${result.seconds.toFixed(2)} s, ${(result.received / MB / result.seconds).toFixed(1)} MB/s`);
            if (result.cacheError) say(`${name}: the cache copy stopped: ${result.cacheError}`);
            metric(`${name} MB/s`, result.received / MB / result.seconds, 'MB/s');
        }
        engine.freeWeights();
        const { usage, quota } = await navigator.storage.estimate();
        say(`storage: ${usage} bytes used, quota ${quota} bytes`);
        say(`after free: live ${engine.liveBytes()} bytes`);
        return { passed: engine.liveBytes() === 0, summary: `live ${engine.liveBytes()} bytes after free` };
    },
};
