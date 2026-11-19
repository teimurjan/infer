// Starts safaridriver, Apple's WebDriver server, on a free port, and stops it.

import type { Subprocess } from 'bun';

export type Driver = { base: string; stop(): void };

// How long safaridriver gets to start.
const START_LIMIT_MS = 10_000;

// Ask the system for a free port: a server on port 0 gets one, and stops at once.
function freePort(): number {
    const probe = Bun.serve({ port: 0, fetch: () => new Response() });
    const port = probe.port;
    probe.stop(true);
    return port as number;
}

// Start safaridriver and wait until its /status says ready.
export async function startDriver(): Promise<Driver | Error> {
    const port = freePort();
    const base = `http://127.0.0.1:${port}`;
    const child: Subprocess = Bun.spawn(['safaridriver', '-p', String(port)], { stdout: 'ignore', stderr: 'pipe' });
    const stop = (): void => child.kill();
    const started = Date.now();
    while (Date.now() - started < START_LIMIT_MS) {
        const ready = await fetch(`${base}/status`).then(response => response.ok).catch(() => false);
        if (ready) return { base, stop };
        // The process ended: safaridriver is not enabled, or the port was taken.
        if (child.exitCode !== null) return new Error(`safaridriver stopped: ${await new Response(child.stderr as ReadableStream).text()} (run: sudo safaridriver --enable)`);
        await Bun.sleep(200);
    }
    stop();
    return new Error(`safaridriver did not start in ${START_LIMIT_MS / 1000} s`);
}
