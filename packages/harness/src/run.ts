// One harness run: the server, safaridriver and a WebDriver session, then the page, until it reports "done".

import type { RunEvent } from '@infer/browser/run-event';
import { RUNS, startServer, type Running } from '@infer/server/server';
import { join } from 'node:path';
import type { Args } from './args.ts';
import { runId } from './args.ts';
import { findIphone } from './device.ts';
import { eventText } from './print.ts';
import { startDriver, type Driver } from './safaridriver.ts';
import { createSession, currentUrl, deleteSession, navigate } from './webdriver.ts';

// How often the harness checks that the Safari window is still there.
const SESSION_CHECK_MS = 15_000;

export type Outcome = { passed: boolean; summary: string; file: string | null };

// Things to undo at the end, newest first.
type Cleanup = (() => Promise<void> | void)[];

// The capabilities of the session: Safari on the connected iPhone, or Safari on this Mac with --mac.
async function capabilities(mac: boolean): Promise<Record<string, unknown> | Error> {
    if (mac) return { browserName: 'safari' };
    const phone = await findIphone();
    if (phone instanceof Error) return phone;
    console.log(`device: ${phone.name}, iOS ${phone.osVersion}, ${phone.udid}`);
    return { browserName: 'safari', platformName: 'iOS', 'safari:deviceUDID': phone.udid };
}

// Wait for the "done" event of `run`, printing every event. Fail on the time limit, or when the Safari window is gone.
function waitForDone(server: Running, driver: Driver, session: string, run: string, timeoutMs: number): Promise<Outcome> {
    const file = join(RUNS, `${run}.jsonl`);
    return new Promise(resolve => {
        // How many times the page loaded. More than one means a reload, for example after iOS killed the tab.
        let loads = 0;
        const finish = (outcome: Outcome): void => {
            stopListening();
            clearTimeout(limit);
            clearInterval(check);
            resolve(outcome);
        };
        const stopListening = server.listen((event: RunEvent) => {
            if (event.run !== run) return;
            if (event.kind === 'start') loads++;
            const text = eventText(event, loads);
            if (text) console.log(text);
            if (event.kind === 'done') finish({ passed: event.passed, summary: event.summary, file });
        });
        const limit = setTimeout(() => finish({ passed: false, summary: `no "done" after ${timeoutMs / 60_000} minutes`, file }), timeoutMs);
        const check = setInterval(async () => {
            const url = await currentUrl(driver.base, session);
            if (url instanceof Error) finish({ passed: false, summary: `the Safari session ended: ${url.message}`, file });
        }, SESSION_CHECK_MS);
    });
}

// Run the page of `args` and give its outcome. Everything that started is stopped before this returns.
export async function runPage(args: Args): Promise<Outcome> {
    const cleanup: Cleanup = [];
    const fail = (error: Error): Outcome => ({ passed: false, summary: error.message, file: null });
    try {
        const caps = await capabilities(args.mac);
        if (caps instanceof Error) return fail(caps);
        // The phone needs HTTPS on the Wi-Fi address. Safari on the Mac can use plain HTTP on 127.0.0.1, which counts as secure.
        const server = await startServer({ lan: !args.mac });
        if (server instanceof Error) return fail(server);
        cleanup.push(() => server.stop());
        const driver = await startDriver();
        if (driver instanceof Error) return fail(driver);
        cleanup.push(() => driver.stop());
        const session = await createSession(driver.base, caps);
        if (session instanceof Error) return fail(session);
        cleanup.push(() => deleteSession(driver.base, session));
        const run = runId(args.path);
        const url = `${server.origin}${args.path}${args.path.includes('?') ? '&' : '?'}run=${run}`;
        console.log(`open: ${url}`);
        // Listen before the page opens, so the first event is not missed.
        const done = waitForDone(server, driver, session, run, args.timeoutMs);
        const opened = await navigate(driver.base, session, url);
        if (opened instanceof Error) return fail(opened);
        return await done;
    } finally {
        for (const step of cleanup.reverse()) await step();
    }
}
