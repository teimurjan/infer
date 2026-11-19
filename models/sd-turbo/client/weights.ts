// Streams one weight file into the engine: from the OPFS cache if it is there, or from the network.
// A network load also writes a copy into the cache. If that copy fails, for example because the storage is full,
// the load goes on without it.

import type { Engine } from '../engine/pkg/engine.js';
import { cachedFile, ignoreNotFound, writeIndex } from './cache.ts';
import { fetchOk } from './fetch.ts';
import { urls } from './urls.ts';

// The engine takes the weights in pieces of at most 1 MiB, so one piece never needs a large WASM copy.
const PIECE_BYTES = 1 << 20;

export type Source = 'cache' | 'network';
// The progress of one file. length is null if the server did not send Content-Length.
export type LoadProgress = { from: Source; received: number; length: number | null };
export type LoadResult = { line: string; from: Source; received: number; seconds: number; cacheError: string | null };

type Opened = { from: Source; index: string; stream: ReadableStream<Uint8Array<ArrayBuffer>>; length: number | null; copy: FileSystemWritableFileStream | null };

// Open the bytes of `name`: the cached copy, or the network with a new copy. The old .json goes first,
// so a copy that breaks halfway is never taken as complete.
async function openSource(folder: FileSystemDirectoryHandle | null, name: string): Promise<Opened> {
    const [index, bin] = folder ? await Promise.all([cachedFile(folder, `${name}.json`), cachedFile(folder, `${name}.bin`)]) : [null, null];
    if (index && bin) return { from: 'cache', index: await index.text(), stream: bin.stream(), length: bin.size, copy: null };
    const indexText = await (await fetchOk(`${urls.weight(name)}.json`)).text();
    const response = await fetchOk(`${urls.weight(name)}.bin`);
    if (!response.body) throw new Error(`${name}.bin: the response has no body`);
    const length = Number(response.headers.get('Content-Length')) || null;
    if (!folder) return { from: 'network', index: indexText, stream: response.body, length, copy: null };
    await folder.removeEntry(`${name}.json`).catch(ignoreNotFound);
    // A copy that cannot start is not an error for the load.
    const copy = await folder.getFileHandle(`${name}.bin`, { create: true }).then(handle => handle.createWritable()).catch(() => null);
    return { from: 'network', index: indexText, stream: response.body, length, copy };
}

// Load `name` into the engine. onProgress gets each chunk. An error frees what the engine took, and throws.
export async function loadWeights(engine: Engine, folder: FileSystemDirectoryHandle | null, name: string, onProgress: (progress: LoadProgress) => void): Promise<LoadResult> {
    const started = performance.now();
    const source = await openSource(folder, name);
    const reader = source.stream.getReader();
    // The copy can stop on its own, so it is a variable.
    let copy = source.copy;
    let cacheError: string | null = null;
    try {
        engine.beginWeights(name, source.index);
        let received = 0;
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            if (copy) {
                // A failed write ends the copy, not the load.
                await copy.write(value).catch(async (error: unknown) => {
                    cacheError = error instanceof Error ? error.message : String(error);
                    await copy?.abort().catch(() => {});
                    copy = null;
                });
            }
            for (let start = 0; start < value.length; start += PIECE_BYTES) {
                // writeWeights returns true when the engine queued GPU uploads, and settle() waits for them.
                if (engine.writeWeights(value.subarray(start, start + PIECE_BYTES))) await engine.settle();
            }
            received += value.length;
            onProgress({ from: source.from, received, length: source.length });
        }
        const line = engine.finishWeights();
        await engine.settle();
        if (copy && folder) {
            await copy.close();
            await writeIndex(folder, name, source.index);
        }
        return { line, from: source.from, received, seconds: (performance.now() - started) / 1000, cacheError };
    } catch (error) {
        engine.abortWeights();
        await reader.cancel().catch(() => {});
        await copy?.abort().catch(() => {});
        // A broken cached file must not be used again, so its .json goes.
        if (source.from === 'cache' && folder) await folder.removeEntry(`${name}.json`).catch(ignoreNotFound);
        throw error;
    }
}

// The size of a weight file on the server, without its body, for the download plan of the demo.
export async function weightSize(name: string): Promise<number | null> {
    const response = await fetchOk(`${urls.weight(name)}.bin`, { method: 'HEAD' });
    return Number(response.headers.get('Content-Length')) || null;
}
