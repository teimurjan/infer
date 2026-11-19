// The events that a test page sends to the Mac while it runs. The server saves them, and the harness waits for them.
// This file has no browser code, so the server and the harness import it too.

// start: the page loaded. line: one log line. metric: one number with a unit. done: the end, passed or not.
// at: milliseconds since the page loaded. seq: the number of the event in this page load, from 0. Beacons can arrive
// out of order, so seq gives the real order in the saved file.
export type RunEvent = { seq: number } & (
    | { run: string; page: string; at: number; kind: 'start'; userAgent: string }
    | { run: string; page: string; at: number; kind: 'line'; text: string }
    | { run: string; page: string; at: number; kind: 'metric'; name: string; value: number; unit: string }
    | { run: string; page: string; at: number; kind: 'done'; passed: boolean; summary: string }
);

// A run id is safe as a file name: letters, digits, dash and underscore.
const RUN_ID = /^[A-Za-z0-9_-]{1,80}$/;

// Read one event from untrusted JSON, or give the reason it is not one. The server calls this before it saves anything.
export function parseRunEvent(text: string): RunEvent | Error {
    let value: unknown;
    try {
        value = JSON.parse(text);
    } catch {
        return new Error('the body is not JSON');
    }
    if (typeof value !== 'object' || value === null) return new Error('the body is not an object');
    const event = value as Record<string, unknown>;
    if (typeof event.run !== 'string' || !RUN_ID.test(event.run)) return new Error('run must match [A-Za-z0-9_-]{1,80}');
    if (typeof event.page !== 'string' || typeof event.at !== 'number' || typeof event.seq !== 'number') return new Error('page, at and seq are required');
    // Each kind needs its own fields with the right types.
    const fields: Record<RunEvent['kind'], Record<string, string>> = {
        start: { userAgent: 'string' },
        line: { text: 'string' },
        metric: { name: 'string', value: 'number', unit: 'string' },
        done: { passed: 'boolean', summary: 'string' },
    };
    const needed = fields[event.kind as RunEvent['kind']];
    if (!needed) return new Error(`unknown kind ${String(event.kind)}`);
    for (const [name, type] of Object.entries(needed)) {
        if (typeof event[name] !== type) return new Error(`${String(event.kind)} needs ${name} as a ${type}`);
    }
    return event as RunEvent;
}
