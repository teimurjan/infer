import { crumb } from '@infer/browser/breadcrumbs';
import { byId } from '@infer/browser/dom';
import { createStore, markActiveAsLeftOnPageHide } from '@infer/browser/probe-store';
import { openGpu } from './gpu.ts';
import type { Gpu, OpenResult, RunOutcome } from './gpu.ts';
import { nextIterations, withBounds } from './search.ts';
import type { SearchState } from './search.ts';

// Every run is saved before it is submitted. A page that loads with a run still
// active knows the previous tab died during that run.
const STORAGE_KEY = 'watchdog-probe';
// Calibration doubles the work until one submit takes at least this long.
const CALIBRATION_TARGET_MS = 50;
const query = new URLSearchParams(location.search);
const settings = {
    startMs: Number(query.get('start') ?? 100),
    maxMs: Number(query.get('max') ?? 20000),
    chunkMs: Number(query.get('chunk') ?? 50),
};

// 'single' looks for the limit, 'calibrate' measures speed, 'split' runs the same work in small submits.
type Mode = 'single' | 'calibrate' | 'split';
// 'blocked' means no GPU device. 'page-left' and 'dead-tab' come from a run that a reload cut short.
type Outcome = RunOutcome | 'blocked' | 'page-left' | 'dead-tab';

type Run = {
    mode: Mode;
    iterations: number;
    estimatedMs?: number;
    outcome: Outcome;
    wallMs?: number;
    gpuMs?: number;
    error?: string;
    // Only split runs have the fields below.
    chunkIterations?: number;
    submits?: number;
    failedAtSubmit?: number;
    maxSubmitMs?: number;
};

// The run that is in progress. `left` is set when the user left the page.
type ActiveRun = Omit<Run, 'outcome'> & { left?: boolean };
type ProbeState = SearchState & { runs: Run[]; active: ActiveRun | null };

const EMPTY_STATE: ProbeState = { ratePerMs: null, lower: null, upper: null, runs: [], active: null };
const store = createStore(STORAGE_KEY, EMPTY_STATE);

// Our own timeout is generous, so the browser's watchdog fires before ours does.
const timeoutFor = (estimatedMs: number): number => Math.max(5000, estimatedMs * 3 + 5000);
const estimateMs = (iterations: number): number | undefined => {
    const { ratePerMs } = store.load();
    return ratePerMs ? Math.round(iterations / ratePerMs) : undefined;
};

// One device is reused until a run fails. Then the next run opens a fresh one.
let gpu: Gpu | null = null;

async function currentGpu(): Promise<OpenResult> {
    if (gpu) {
        return { ok: true, gpu };
    }
    const opened = await openGpu({ onUncapturedError: error => crumb('watchdog', 'gpu', 'uncaptured-error', { error }) });
    if (!opened.ok) {
        return opened;
    }
    gpu = opened.gpu;
    crumb('watchdog', 'gpu', 'device', { timestamps: gpu.hasTimestamps });
    return opened;
}

function dropGpu(): void {
    gpu?.destroy();
    gpu = null;
}

function record(run: Run): void {
    crumb('watchdog', run.mode, run.outcome, run);
    const state = store.load();
    store.save(withBounds({ ...state, active: null, runs: [...state.runs, run] }, run));
    render();
}

async function runSingle(iterations: number, mode: Mode = 'single'): Promise<Run> {
    const estimatedMs = estimateMs(iterations);
    // Save first: if the tab dies during the submit, the next page load finds this.
    store.save({ ...store.load(), active: { mode, iterations, estimatedMs } });
    const opened = await currentGpu();
    if (!opened.ok) {
        const run: Run = { mode, iterations, estimatedMs, outcome: 'blocked', error: opened.error };
        record(run);
        return run;
    }
    setProgress(`${mode}: ${iterations} iterations, about ${estimatedMs ?? '?'} ms`);
    const result = await opened.gpu.run(iterations, timeoutFor(estimatedMs ?? 0));
    if (result.outcome !== 'ok') {
        dropGpu();
    }
    const run: Run = { mode, iterations, estimatedMs, ...result };
    record(run);
    return run;
}

async function calibrate(): Promise<boolean> {
    for (let iterations = 1; ; iterations *= 2) {
        const run = await runSingle(iterations, 'calibrate');
        if (run.outcome !== 'ok') {
            return false;
        }
        // An ok run always has wallMs. gpuMs is there only with timestamp queries.
        const ms = run.gpuMs ?? run.wallMs ?? 0;
        if (ms >= CALIBRATION_TARGET_MS) {
            store.save({ ...store.load(), ratePerMs: iterations / ms });
            return true;
        }
    }
}

const ensureCalibrated = async (): Promise<boolean> => Boolean(store.load().ratePerMs) || calibrate();

async function findLimit(): Promise<void> {
    if (!await ensureCalibrated()) {
        return;
    }
    for (let next = nextIterations(store.load(), settings); next; next = nextIterations(store.load(), settings)) {
        const run = await runSingle(next);
        if (run.outcome === 'blocked') {
            return;
        }
    }
    await splitRun();
}

// Same total work as the smallest single submit that died, in submits of about chunkMs each.
async function splitRun(): Promise<void> {
    const { ratePerMs, lower, upper } = store.load();
    const total = upper?.iterations ?? lower?.iterations;
    if (!total || !ratePerMs) {
        return;
    }
    const chunk = Math.max(1, Math.round(ratePerMs * settings.chunkMs));
    const submits = Math.ceil(total / chunk);
    const base = { mode: 'split', iterations: total, estimatedMs: estimateMs(total), chunkIterations: chunk, submits } as const;
    // Start from a fresh device, so earlier runs do not affect this one.
    dropGpu();
    const opened = await currentGpu();
    if (!opened.ok) {
        record({ ...base, outcome: 'blocked', error: opened.error });
        return;
    }
    let gpuMs = 0;
    let maxSubmitMs = 0;
    const start = performance.now();
    for (let submit = 0, done = 0; done < total; submit++, done += chunk) {
        store.save({ ...store.load(), active: { ...base, failedAtSubmit: submit } });
        setProgress(`split: submit ${submit + 1} of ${submits}`);
        // The last submit can be smaller than one chunk.
        const result = await opened.gpu.run(Math.min(chunk, total - done), timeoutFor(settings.chunkMs));
        if (result.outcome !== 'ok') {
            dropGpu();
            record({ ...base, outcome: result.outcome, error: result.error, failedAtSubmit: submit });
            return;
        }
        const submitMs = result.gpuMs ?? result.wallMs;
        gpuMs += submitMs;
        maxSubmitMs = Math.max(maxSubmitMs, submitMs);
    }
    const wallMs = Math.round(performance.now() - start);
    record({ ...base, outcome: 'ok', gpuMs: Math.round(gpuMs), maxSubmitMs, wallMs });
}

function formatRun(run: Run): string {
    const timing = run.outcome === 'ok' ? ` gpu ${run.gpuMs ?? '-'} ms, wall ${run.wallMs} ms` : '';
    const split = run.mode === 'split' ? ` in ${run.submits} submits` : '';
    const failedAt = run.failedAtSubmit !== undefined && run.outcome !== 'ok' ? ` at submit ${run.failedAtSubmit + 1}` : '';
    const error = run.error ? `  ${run.error}` : '';
    return `${run.mode.padEnd(9)} ~${run.estimatedMs ?? '?'} ms${split}: ${run.outcome}${failedAt}${timing}${error}`;
}

function summary({ lower, upper }: SearchState): string {
    const survived = lower ? `longest single submit that finished: ${lower.ms} ms` : 'no single submit finished yet';
    const died = upper ? `shortest that died: about ${upper.ms} ms` : 'no single submit died yet';
    return `${survived}\n${died}`;
}

const setProgress = (text: string): void => { byId('progress', HTMLElement).textContent = text; };

function render(): void {
    const state = store.load();
    byId('summary', HTMLElement).textContent = summary(state);
    // Calibration runs are noise for the reader, so we hide them.
    const shown = state.runs.filter(run => run.mode !== 'calibrate');
    byId('runs', HTMLElement).textContent = shown.map(formatRun).join('\n') || 'no runs yet';
}

function setBusy(busy: boolean): void {
    for (const control of document.querySelectorAll<HTMLButtonElement | HTMLInputElement>('button, input')) {
        control.disabled = busy;
    }
}

// Wraps a button action: the controls are off while it runs, so two runs cannot overlap.
const exclusive = (action: () => Promise<void>) => async (): Promise<void> => {
    setBusy(true);
    try {
        await action();
    } finally {
        setBusy(false);
        setProgress('');
    }
};

function bindControls(): void {
    byId('settings', HTMLElement).textContent =
        `start ${settings.startMs} ms, cap ${settings.maxMs} ms, split chunk ${settings.chunkMs} ms (set with ?start=&max=&chunk=)`;
    byId('find', HTMLButtonElement).addEventListener('click', exclusive(findLimit));
    byId('dial-run', HTMLButtonElement).addEventListener('click', exclusive(async () => {
        if (await ensureCalibrated()) {
            const ms = Number(byId('dial-ms', HTMLInputElement).value);
            // ensureCalibrated returned true, so ratePerMs is set.
            await runSingle(Math.max(1, Math.round((store.load().ratePerMs ?? 0) * ms)));
        }
    }));
    byId('reset', HTMLButtonElement).addEventListener('click', () => {
        store.save(EMPTY_STATE);
        render();
    });
    markActiveAsLeftOnPageHide(store);
}

// A run that is still active means the last page load did not finish it.
function recoverInterruptedRun(): void {
    const { active } = store.load();
    if (!active) {
        return;
    }
    const { left, ...run } = active;
    record({ ...run, outcome: left ? 'page-left' : 'dead-tab' });
}

bindControls();
recoverInterruptedRun();
render();
