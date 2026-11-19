// The POST /report route. Test pages send their run events here. Each event goes to runs/<run>.jsonl and to the listeners.

import { appendFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { parseRunEvent, type RunEvent } from '@infer/browser/run-event';

// A listener gets every event that the server saved, for example the harness that waits for "done".
export type RunListener = (event: RunEvent) => void;

// Make the route handler. It saves into `folder` and calls `listeners`.
export const runSink = (folder: string, listeners: ReadonlySet<RunListener>) => async (request: Request): Promise<Response> => {
    const event = parseRunEvent(await request.text());
    // A bad event is the page's bug. 400 says so, and nothing is saved. sendBeacon never shows the answer to the page,
    // so the terminal shows it, for example when a page was built before a change of RunEvent.
    if (event instanceof Error) {
        console.warn(`[report] refused: ${event.message}`);
        return new Response(event.message, { status: 400 });
    }
    await mkdir(folder, { recursive: true });
    // The server clock, so lines from a page that reloaded keep their real order.
    await appendFile(join(folder, `${event.run}.jsonl`), `${JSON.stringify({ received: new Date().toISOString(), ...event })}\n`);
    for (const listener of listeners) listener(event);
    return new Response(null, { status: 204 });
};
