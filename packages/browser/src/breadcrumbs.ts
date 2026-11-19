// Stage breadcrumbs: shown on the page, logged to the console, and beaconed to the dev server
// so they survive a tab crash on the phone.

import { errorText } from './errors.ts';

// Extra fields for one breadcrumb, for example a time in ms or an error text.
export type CrumbDetail = Record<string, unknown>;

// A short random id, so lines from different page loads can be told apart.
const pageId = crypto.randomUUID().slice(0, 8);
const pageStart = performance.now();
const elapsed = (since: number): number => Math.round(performance.now() - since);

export function crumb(subject: string, stage: string, event: string, detail: CrumbDetail = {}): void {
    const entry = { page: pageId, t: elapsed(pageStart), subject, stage, event, ...detail };
    const line = `+${entry.t}ms ${subject} ${stage} ${event} ${JSON.stringify(detail)}`;
    console.log(line);
    // Not every page has a #crumbs element, so a missing one is fine here.
    document.getElementById('crumbs')?.append(`${line}\n`);
    // sendBeacon still delivers if the page dies right after this call.
    navigator.sendBeacon('/crumb', JSON.stringify(entry));
}

// Runs one async action and writes a start crumb, then an ok or fail crumb with the time.
// `describe` picks the fields of the result that go into the ok crumb.
export async function track<Result>(
    subject: string,
    stage: string,
    action: () => Promise<Result>,
    describe: (result: Result) => CrumbDetail = () => ({}),
): Promise<Result> {
    crumb(subject, stage, 'start');
    const start = performance.now();
    try {
        const result = await action();
        crumb(subject, stage, 'ok', { ms: elapsed(start), ...describe(result) });
        return result;
    } catch (error) {
        crumb(subject, stage, 'fail', { ms: elapsed(start), error: errorText(error) });
        throw error;
    }
}

export function trackGlobalErrors(): void {
    window.addEventListener('error', event => crumb('page', 'global', 'error', { error: String(event.message) }));
    window.addEventListener('unhandledrejection', event => crumb('page', 'global', 'rejection', { error: errorText(event.reason) }));
}
