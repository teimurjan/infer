// Serves files. Each URL path is looked up in the folders in order, and the first file found wins.
// The built site comes first, so "/" is its home page. The repository comes second, for weights, golden files and the engine.

import { stat } from 'node:fs/promises';
import { join, normalize } from 'node:path';

// A path is refused if one of its parts starts with a dot (.git, .env) or is "..", which climbs out of the folder.
const isHidden = (path: string): boolean => path.split('/').some(part => part.startsWith('.'));

// The file for a URL path in one folder: the path itself, or index.html if the path is a folder. Null if neither exists.
async function findFile(root: string, urlPath: string): Promise<string | null> {
    const path = join(root, normalize(urlPath));
    // stat throws if the path does not exist.
    const info = await stat(path).catch(() => null);
    if (info?.isFile()) return path;
    if (info?.isDirectory()) return findFile(root, join(urlPath, 'index.html'));
    return null;
}

// Make the handler that serves files from `roots`.
export const fileServer = (roots: readonly string[]) => async (request: Request): Promise<Response> => {
    // decodeURIComponent turns %20 back into a space.
    const urlPath = decodeURIComponent(new URL(request.url).pathname);
    if (isHidden(urlPath)) return new Response('not found', { status: 404 });
    for (const root of roots) {
        const path = await findFile(root, urlPath);
        // Bun.file streams the file, so a 1 GB weight file does not go into memory. Bun sets the content type from the name.
        // no-cache: the browser asks again each time, so a new build is never hidden by an old copy.
        if (path) return new Response(Bun.file(path), { headers: { 'cache-control': 'no-cache' } });
    }
    return new Response('not found', { status: 404 });
};
