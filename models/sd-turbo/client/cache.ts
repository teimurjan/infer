// The weight cache in OPFS, the private file storage of the page. A second visit loads the weights from there, not the network.

// The OPFS folder of the weights.
const CACHE_FOLDER = 'weights';

// OPFS throws NotFoundError for a missing file. Other errors are real.
export const ignoreNotFound = (error: unknown): void => {
    if (!(error instanceof DOMException && error.name === 'NotFoundError')) throw error;
};

// The cache folder, or null if this browser cannot write OPFS files from a page.
export async function openCache(): Promise<FileSystemDirectoryHandle | null> {
    if (!navigator.storage?.getDirectory) return null;
    // Safari before 26 had OPFS, but no createWritable on the main thread.
    if (!('createWritable' in FileSystemFileHandle.prototype)) return null;
    const root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle(CACHE_FOLDER, { create: true });
}

// Delete the whole cache folder.
export async function clearCache(): Promise<void> {
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(CACHE_FOLDER, { recursive: true }).catch(ignoreNotFound);
}

// A file of the cache, or null if it is missing.
export async function cachedFile(folder: FileSystemDirectoryHandle, name: string): Promise<File | null> {
    try {
        return await (await folder.getFileHandle(name)).getFile();
    } catch (error) {
        ignoreNotFound(error);
        return null;
    }
}

// True if both files of a weight set are in the cache. The .json file is written last, so it marks a complete copy.
export async function isCached(folder: FileSystemDirectoryHandle | null, name: string): Promise<boolean> {
    if (!folder) return false;
    const [index, bin] = await Promise.all([cachedFile(folder, `${name}.json`), cachedFile(folder, `${name}.bin`)]);
    return index !== null && bin !== null;
}

// Write the index text as the last file of a complete copy. Its presence marks the copy as complete.
export async function writeIndex(folder: FileSystemDirectoryHandle, name: string, text: string): Promise<void> {
    const writable = await (await folder.getFileHandle(`${name}.json`, { create: true })).createWritable();
    await writable.write(text);
    await writable.close();
}
