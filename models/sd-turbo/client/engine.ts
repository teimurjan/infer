// Loads the WASM build of the engine and opens the GPU.

import init, { Engine, PromptTokenizer, type InitOutput } from '../engine/pkg/engine.js';
import { fetchText } from './fetch.ts';
import { urls } from './urls.ts';

export type Session = { engine: Engine; wasmMemory: WebAssembly.Memory };

// The WASM module loads once per page. Every export of engine.js needs it, so each caller waits for this first.
// The URL is given, because the bundler moves engine.js, and the default path next to it would be wrong.
let loaded: Promise<InitOutput> | null = null;
export const loadWasm = (): Promise<InitOutput> => (loaded ??= init({ module_or_path: urls.wasm }));

// Load the WASM module, then open the GPU.
export async function openSession(): Promise<Session> {
    const wasm = await loadWasm();
    const engine = await Engine.open();
    return { engine, wasmMemory: wasm.memory };
}

// The engine's tokenizer, from the two files that python/export_tokenizer.py writes. It needs the WASM module, not the GPU.
export async function loadTokenizer(): Promise<PromptTokenizer> {
    const [vocab, merges] = await Promise.all([fetchText(urls.weight('tokenizer_vocab.json')), fetchText(urls.weight('tokenizer_merges.txt')), loadWasm()]);
    return new PromptTokenizer(vocab, merges);
}
