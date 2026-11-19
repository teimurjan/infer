// Downloads weight files into the OPFS cache, without the GPU. The demo does this once, before the first image,
// so the first image does not wait for 2.2 GB, and the download shows its own progress.

import { ignoreNotFound, isCached, writeIndex } from './cache.ts';
import { fetchOk, fetchText } from './fetch.ts';
import { urls } from './urls.ts';

// The progress of one file: bytes so far, and the total if the server sent it.
export type DownloadProgress = { received: number; length: number | null };

// Download `name` into `folder`, unless a complete copy is there. Returns true if it downloaded.
export async function downloadToCache(folder: FileSystemDirectoryHandle, name: string, onProgress: (progress: DownloadProgress) => void): Promise<boolean> {
    if (await isCached(folder, name)) return false;
    const index = await fetchText(`${urls.weight(name)}.json`);
    // The old .json goes first, so a copy that breaks halfway is never taken as complete.
    await folder.removeEntry(`${name}.json`).catch(ignoreNotFound);
    const response = await fetchOk(`${urls.weight(name)}.bin`);
    if (!response.body) throw new Error(`${name}.bin: the response has no body`);
    const length = Number(response.headers.get('Content-Length')) || null;
    const writable = await (await folder.getFileHandle(`${name}.bin`, { create: true })).createWritable();
    const reader = response.body.getReader();
    let received = 0;
    try {
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            await writable.write(value);
            received += value.length;
            onProgress({ received, length });
        }
        await writable.close();
    } catch (error) {
        await reader.cancel().catch(() => {});
        await writable.abort().catch(() => {});
        throw error;
    }
    await writeIndex(folder, name, index);
    return true;
}
