// The dictionary entry of SD-Turbo, for the site.
// The stage times are from the golden-image run on the iPhone (models/sd-turbo/PROGRESS.md, 2026-11-19).

import type { Entry } from '@infer/browser/entry';
import { DECODER_FILE, DEFAULT_UNET_FILE, TEXT_FILE } from './pipeline.ts';

export const entry: Entry = {
    id: 'sd-turbo',
    letter: 'S',
    pos: 'n. text → image',
    definition: 'a diffusion model distilled to draw a 512 × 512 image in one step.',
    etymology: 'from Stable Diffusion 2.1, distilled by Stability AI with adversarial diffusion distillation.',
    parts: [
        { name: 'tokenizer', does: 'splits the prompt into 77 token ids', file: null, note: '77 ids' },
        { name: 'text encoder', does: 'turns the ids into 77 vectors of 1024 numbers', file: TEXT_FILE },
        { name: 'UNet, 8-bit', does: 'removes the noise from a 4 × 64 × 64 latent in one step', file: DEFAULT_UNET_FILE },
        { name: 'TAESD', does: 'turns the latent into 512 × 512 pixels', file: DECODER_FILE },
    ],
    measured: {
        device: 'iPhone 16 Pro Max',
        browser: 'Safari 26.6.2',
        date: '2026-11-19',
        seconds: [
            { part: 'text encoder', seconds: 0.37 },
            { part: 'UNet, 8-bit', seconds: 2.54 },
            { part: 'TAESD', seconds: 0.73 },
        ],
        source: 'golden-image test through the harness, runs/20261119072352-sd-turbo-dev-test-golden-image-hi29.jsonl',
    },
    input: { label: 'Prompt', example: 'a lighthouse on a cliff at sunset, oil painting' },
};
