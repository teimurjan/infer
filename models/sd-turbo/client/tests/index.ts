// The test suite of SD-Turbo, in the order of the developer page: small kernels first, whole images last.

import type { TestSuite } from '@infer/browser/test-suite';
import { goldenImageTest, makeTwentyTest, profileTest } from './images.ts';
import { attention, bench, layers, silu, taesd } from './kernels.ts';
import { weightsTest } from './weights.ts';

export const suite: TestSuite = {
    model: 'sd-turbo',
    title: 'SD-Turbo',
    tests: [silu, layers, attention, taesd, bench, weightsTest, goldenImageTest, makeTwentyTest, profileTest],
};
