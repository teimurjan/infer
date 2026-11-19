import { crumb } from '@infer/browser/breadcrumbs';
import { byId } from '@infer/browser/dom';
import { createStore, markActiveAsLeftOnPageHide } from '@infer/browser/probe-store';
import { RAMPS, isRampName } from './allocators.ts';
import type { AllocatorKind, FailureOutcome, RampName } from './allocators.ts';

// Each ramp runs in a fresh page load. Progress lives in localStorage, so a page that
// loads with a ramp still active knows the previous tab died at that size.
const STORAGE_KEY = 'memory-probe';
const MB = 1024 * 1024;
const params = new URLSearchParams(location.search);
const stepMb = Number(params.get('step') ?? 64);
const maxMb = Number(params.get('max') ?? 8192);

// How a ramp ended. 'cap' means it reached maxMb without a failure.
type Outcome = FailureOutcome | 'cap' | 'page-left' | 'dead-tab';
type RampEnd = { outcome: Outcome; mb: number; error?: string; kind?: AllocatorKind };
type RampResult = RampEnd & { ramp: RampName };
// The ramp that runs now, and how far it got. `left` is set when the user left the page.
type ActiveRamp = { ramp: RampName; mb: number; left?: boolean };
type ProbeState = { queue: RampName[]; active: ActiveRamp | null; results: RampResult[] };

const EMPTY_STATE: ProbeState = { queue: [], active: null, results: [] };
const store = createStore(STORAGE_KEY, EMPTY_STATE);

function withResult(state: ProbeState, ramp: RampName, end: RampEnd): ProbeState {
    crumb(ramp, 'ramp', 'end', end);
    return { ...state, active: null, results: [...state.results, { ramp, ...end }] };
}

// Lets the browser draw the page between steps.
const yieldToBrowser = () => new Promise(resolve => setTimeout(resolve, 0));

async function runRamp(ramp: RampName): Promise<RampEnd> {
    const { create, kindsPerStep } = RAMPS[ramp];
    crumb(ramp, 'ramp', 'start', { stepMb, maxMb });
    const allocator = await create();
    if (!allocator.ok) {
        return { outcome: allocator.outcome, error: allocator.error, mb: 0 };
    }
    const mbPerStep = stepMb * kindsPerStep;
    let mb = 0;
    while (mb + mbPerStep <= maxMb) {
        const result = await allocator.allocate(stepMb * MB);
        if (!result.ok) {
            return { outcome: result.outcome, error: result.error, kind: result.kind, mb };
        }
        mb += mbPerStep;
        // Save before the next step, so a dead tab leaves the last good size behind.
        store.save({ ...store.load(), active: { ramp, mb } });
        crumb(ramp, 'step', 'ok', { mb });
        byId('progress', HTMLElement).textContent = `${ramp}: ${mb} MB`;
        await yieldToBrowser();
    }
    return { outcome: 'cap', mb };
}

function formatResult({ ramp, outcome, mb, kind, error }: RampResult): string {
    const failedKind = kind ? ` (${kind})` : '';
    const detail = error ? `  ${error}` : '';
    return `${ramp.padEnd(6)} ${String(mb).padStart(6)} MB  ${outcome}${failedKind}${detail}`;
}

function render(state: ProbeState, running: boolean): void {
    byId('results', HTMLElement).textContent = state.results.map(formatResult).join('\n') || 'no results yet';
    byId('settings', HTMLElement).textContent = `step ${stepMb} MB, cap ${maxMb} MB (set with ?step=&max=)`;
    byId('controls', HTMLElement).hidden = running;
}

// Saves the ramps to run, then reloads, so the first ramp starts in a fresh page.
function start(queue: RampName[]): void {
    store.save({ ...EMPTY_STATE, queue });
    location.reload();
}

function bindControls(): void {
    for (const button of document.querySelectorAll<HTMLElement>('[data-ramps]')) {
        // data-ramps holds names like "wasm,js". Unknown names are dropped.
        const ramps = (button.dataset.ramps ?? '').split(',').filter(isRampName);
        button.addEventListener('click', () => start(ramps));
    }
    markActiveAsLeftOnPageHide(store);
}

async function main(): Promise<void> {
    bindControls();
    let state = store.load();
    // A ramp that is still active means the last page load did not finish it.
    if (state.active) {
        const { ramp, mb, left } = state.active;
        state = withResult(state, ramp, { outcome: left ? 'page-left' : 'dead-tab', mb });
        store.save(state);
    }
    const [next, ...rest] = state.queue;
    render(state, Boolean(next));
    if (!next) {
        return;
    }
    store.save({ ...state, queue: rest, active: { ramp: next, mb: 0 } });
    const end = await runRamp(next);
    store.save(withResult(store.load(), next, end));
    location.reload();
}

main();
