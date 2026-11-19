// The development server of the repository, as a function. main.ts starts it from the command line, and the harness starts it too.

import type { Server } from 'bun';
import { join } from 'node:path';
import { crumbSink } from './crumbs.ts';
import { fileServer } from './files.ts';
import { runSink, type RunListener } from './runs.ts';
import { loadTls, type Tls } from './tls.ts';

// The repository folder, two levels above this file (packages/server/src).
export const ROOT = join(import.meta.dir, '..', '..', '..');
// The folder of the run events. .gitignore leaves it out.
export const RUNS = join(ROOT, 'runs');

export type Running = {
    // The address to open, for example https://10.120.1.125:8443.
    origin: string;
    // Add a listener for run events. The return value removes it.
    listen(listener: RunListener): () => void;
    stop(): void;
};

// Start Bun.serve, or give an error. Bun.serve throws if the port is taken, for example by `bun run serve` in another terminal.
function serve(port: number, tls: Tls | undefined, listeners: ReadonlySet<RunListener>): Server<undefined> | Error {
    try {
        return Bun.serve({
            hostname: tls ? '0.0.0.0' : '127.0.0.1',
            port,
            tls: tls && { cert: tls.cert, key: tls.key },
            // No time limit on a request, because a weight file of 1 GB takes longer than the default of 10 s on Wi-Fi.
            idleTimeout: 0,
            routes: {
                '/crumb': { POST: crumbSink(join(ROOT, 'packages', 'probes', 'crumbs.jsonl')) },
                '/report': { POST: runSink(RUNS, listeners) },
            },
            // The built site first, then the whole repository.
            fetch: fileServer([join(ROOT, 'apps', 'site', 'dist'), ROOT]),
        });
    } catch (error) {
        return new Error(`cannot listen on port ${port}: ${error instanceof Error ? error.message : String(error)}. Stop the other server first.`);
    }
}

// Start the server. lan: HTTPS on port 8443 for the iPhone. Otherwise HTTP on 127.0.0.1:8765 for the Mac.
export async function startServer({ lan }: { lan: boolean }): Promise<Running | Error> {
    // The certificate and the address it names, only in the iPhone mode.
    const tls = lan ? await loadTls() : undefined;
    if (tls instanceof Error) return tls;
    const listeners = new Set<RunListener>();
    const server = serve(lan ? 8443 : 8765, tls, listeners);
    if (server instanceof Error) return server;
    const listen = (listener: RunListener): (() => void) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
    };
    return { origin: `${server.protocol}://${tls ? tls.host : server.hostname}:${server.port}`, listen, stop: () => server.stop(true) };
}
