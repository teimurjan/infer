// Sends the events of one test run to the Mac, where the server saves them in runs/<run>.jsonl.
// The page address can name the run: ?run=<id>. The harness does that. A run by hand gets its own id.

import type { RunEvent } from './run-event.ts';

// The fields that every event of this page shares.
type Base = { run: string; page: string };
// The fields that the page gives for one event.
type EventBody = RunEvent extends infer Event ? (Event extends RunEvent ? Omit<Event, keyof Base | 'at' | 'seq'> : never) : never;

export type Reporter = {
    // The run id, for example to show it on the page.
    run: string;
    // True if the harness started this page. The page then runs its test without a tap.
    automated: boolean;
    send(event: EventBody): void;
};

// A new id for a run by hand: the time, plus random letters so two phones never collide.
const manualRun = (): string => `manual-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

// Make the reporter of this page. `page` names the test, for example "sd-turbo/golden-image".
export function reporter(page: string): Reporter {
    const fromAddress = new URLSearchParams(location.search).get('run');
    const run = fromAddress ?? manualRun();
    // The number of the next event of this page load.
    let seq = 0;
    const send = (event: EventBody): void => {
        const body = JSON.stringify({ ...event, run, page, seq: seq++, at: Math.round(performance.now()) });
        // sendBeacon still delivers if iOS kills the tab right after this call. It returns false if the browser refuses.
        if (!navigator.sendBeacon('/report', body)) console.warn(`report not sent: ${body}`);
    };
    return { run, automated: fromAddress !== null, send };
}
