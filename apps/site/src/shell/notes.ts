// The usage notes of the book: the limits that every engine on this site is built around, with their sources.
// The numbers are from CLAUDE.md and crash-record/2026-11-15-iphone16promax.md.

import type { Source } from '@infer/ui/components/cite';

export const limitsSource: Source = { id: 'limits', text: 'iPhone 16 Pro Max, Safari, crash probes of 2026-11-15 (crash-record/2026-11-15-iphone16promax.md).' };

// One note: the word, its part of speech, the sentence, and whether its numbers cite the limits source.
export type Note = { word: string; pos: string; text: string; cited: boolean };

export const notes: readonly Note[] = [
    { word: 'tab', pos: 'n.', text: 'the memory of one Safari tab. iOS closes it at about 1920 MB, and the JS heap, the WASM memory and the GPU buffers share that budget.', cited: true },
    { word: 'buffer', pos: 'n.', text: 'one block of GPU memory. On this phone one buffer holds at most about 1073 MB.', cited: true },
    { word: 'submit', pos: 'n.', text: 'one batch of GPU work. If a submit runs past 0.5 s the GPU is lost, so every engine keeps each submit under 0.25 s.', cited: true },
    { word: 'kernel', pos: 'n.', text: 'a small GPU program. Here each one is written by hand in WGSL. No ML framework runs inside an engine.', cited: false },
    { word: 'weights', pos: 'n. pl.', text: 'what a model learned. They download once into OPFS, the private storage of the page, and stream to the GPU one stage at a time.', cited: false },
    { word: 'stage', pos: 'n.', text: 'one part of a pipeline that loads, runs and frees its memory before the next part loads, so that a model larger than the tab still fits.', cited: false },
];
