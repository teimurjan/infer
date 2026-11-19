// The benchmark page: the engine of this project and the onnxruntime-web example, with the same prompt, noise and steps.
// Each run is one fresh page: the page runs one job, saves its result, and reloads itself for the next one.
// So every cold time starts from a page with nothing loaded, and a tab that iOS kills counts as a failed run.
// The harness opens it with ?run=<id>&runs=<n>&warm=<n>&runtimes=engine,onnx, and the plan starts by itself.

import { reporter } from '@infer/browser/report';
import { installBufferCounter, peakBytes } from '@infer/sd-turbo/benchmark/gpu-memory';
import { SIDE, checkImage } from '@infer/sd-turbo/benchmark/image';
import { readInputs, type GoldenInputs } from '@infer/sd-turbo/benchmark/inputs';
import { clearState, diedResult, loadState, nextJob, saveState, type Job, type Plan, type RunResult } from '@infer/sd-turbo/benchmark/record';
import { LABEL, OPEN, RUNTIMES, type Runtime } from '@infer/sd-turbo/benchmark/runners';
import { reportCsv, reportMarkdown } from '@infer/sd-turbo/benchmark/table';
import { mountPage } from '../../shell/page.ts';
import { benchmarkView } from './view.ts';

// Count GPU buffers from the start, before any runtime opens a device.
installBufferCounter();

// A run that takes longer than this counts as failed. The page then reloads, which ends the GPU work too.
const RUN_LIMIT_MS = 15 * 60 * 1000;
// The pause before the next run starts by itself, so the user can press Stop.
const NEXT_RUN_DELAY_MS = 3000;
// The run id of the harness that owns the saved state. A new harness run starts from an empty state.
const HARNESS_KEY = 'sd-turbo-benchmark-run';

const view = benchmarkView();
mountPage({ title: 'sd-turbo · benchmark', guide: 'benchmark', entry: 'sd-turbo' }, view.element);
const report = reporter('sd-turbo/benchmark');
report.send({ kind: 'start', userAgent: navigator.userAgent });

// The text of a thrown value.
const errorText = (error: unknown): string => (error instanceof Error ? error.message : String(error));

// Add one line to the log, and send it to the Mac.
function say(line: string): void {
    view.output.add(line);
    report.send({ kind: 'line', text: line });
}

// Show an image on the canvas.
function draw(rgba: Uint8Array): void {
    view.canvas.getContext('2d')?.putImageData(new ImageData(new Uint8ClampedArray(rgba), SIDE, SIDE), 0, 0);
}

// Fail after `ms` milliseconds, so a run that hangs does not stop the whole benchmark.
function timeLimit(ms: number): Promise<never> {
    return new Promise((_, reject) => setTimeout(() => reject(new Error(`the run took more than ${ms / 1000} s`)), ms));
}

// One run: open the runtime and make the first image (cold), then `warm` more images on the same page (warm).
// The clock starts after the golden inputs are read, because both runtimes read the same inputs.
async function measure(job: Job, warm: number, inputs: GoldenInputs): Promise<RunResult> {
    const warmMs: number[] = [];
    let coldMs: number | null = null;
    try {
        const started = performance.now();
        const makeImage = await OPEN[job.runtime]();
        draw(checkImage(await makeImage(inputs)));
        coldMs = performance.now() - started;
        say(`${job.runtime} run ${job.run}: cold ${(coldMs / 1000).toFixed(2)} s`);
        for (let image = 1; image <= warm; image++) {
            const imageStarted = performance.now();
            draw(checkImage(await makeImage(inputs)));
            warmMs.push(performance.now() - imageStarted);
            say(`${job.runtime} run ${job.run}: warm ${image} ${(warmMs[warmMs.length - 1]! / 1000).toFixed(2)} s`);
        }
        return { ...job, ok: true, error: null, coldMs, warmMs, peakBytes: peakBytes() };
    } catch (error) {
        say(`${job.runtime} run ${job.run}: FAILED: ${errorText(error)}`);
        return { ...job, ok: false, error: errorText(error), coldMs, warmMs, peakBytes: peakBytes() };
    }
}

// Run one job, save its result, and load a fresh page for the next one.
async function runJob(job: Job, plan: Plan): Promise<void> {
    // Mark the run as going on. If the tab dies, the next page finds this mark.
    saveState({ ...loadState(), pending: job });
    say(`${job.runtime} run ${job.run}${job.run === 0 ? ' (warm-up, fills the caches, not counted)' : ''}: started`);
    const inputs = await readInputs();
    const result = await Promise.race([measure(job, plan.warm, inputs), timeLimit(RUN_LIMIT_MS)])
        .catch((error: unknown): RunResult => ({ ...job, ok: false, error: errorText(error), coldMs: null, warmMs: [], peakBytes: peakBytes() }));
    // Read the state again: the user can have pressed Stop during the run.
    const state = loadState();
    saveState({ ...state, results: [...state.results, result], pending: null });
    location.reload();
}

// Show the report and turn on the download buttons when there is something to download.
function render(): void {
    const state = loadState();
    // With no result yet, a sentence says what fills the table, instead of an empty table.
    view.report.textContent = state.results.length === 0 ? 'No runs yet. Press start: each run reloads this page, and its row lands here.' : reportMarkdown(state);
    view.controls.csv.disabled = view.controls.markdown.disabled = state.results.length === 0;
}

// Give the user a file with `text`.
function download(name: string, type: string, text: string): void {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([text], { type }));
    link.download = name;
    link.click();
    URL.revokeObjectURL(link.href);
}

// The run that starts by itself, so Stop can cancel it.
let scheduled: ReturnType<typeof setTimeout> | undefined;

// Start the next job of the plan after a short pause, or say that the plan is done.
function continuePlan(): void {
    const state = loadState();
    const job = nextJob(state);
    if (!state.plan || !job) {
        if (!state.plan) {
            say('ready. Press Start.');
            return;
        }
        say('done. Download the table below.');
        // The harness gets the whole table and the end of the run.
        // One event for the whole table, because separate events can arrive out of order.
        report.send({ kind: 'line', text: reportMarkdown(state) });
        const failed = state.results.filter(result => !result.ok).length;
        report.send({ kind: 'done', passed: failed === 0, summary: `${state.results.length - failed} of ${state.results.length} runs succeeded` });
        return;
    }
    const plan = state.plan;
    say(`next: ${LABEL[job.runtime]}, run ${job.run} of ${plan.runs}, in ${NEXT_RUN_DELAY_MS / 1000} s. Press Stop to pause.`);
    scheduled = setTimeout(() => void runJob(job, plan), NEXT_RUN_DELAY_MS);
}

// The plan from the page address of the harness: runs, warm images, and the runtimes (engine by default).
function planFromAddress(): Plan {
    const query = new URLSearchParams(location.search);
    const runtimes = (query.get('runtimes') ?? 'engine').split(',').filter((name): name is Runtime => (RUNTIMES as readonly string[]).includes(name));
    return { runtimes, runs: Number(query.get('runs') ?? 10), warm: Number(query.get('warm') ?? 3) };
}

// A harness run that is new to this page starts from an empty state with its own plan.
if (report.automated && localStorage.getItem(HARNESS_KEY) !== report.run) {
    clearState();
    saveState({ ...loadState(), plan: planFromAddress() });
    localStorage.setItem(HARNESS_KEY, report.run);
}
// A run that was going on when this page started never ended: the tab was killed or reloaded.
const start = loadState();
if (start.pending) {
    saveState({ ...start, results: [...start.results, diedResult(start.pending)], pending: null });
    say(`${start.pending.runtime} run ${start.pending.run}: FAILED: the page ended during the run`);
}
// Show the plan of the saved state in the fields.
const saved = loadState().plan;
if (saved) {
    view.runsField.value = String(saved.runs);
    view.warmField.value = String(saved.warm);
    view.runtimeBoxes.forEach(({ runtime, box }) => { box.checked = saved.runtimes.includes(runtime); });
}

// Start (or continue) the plan in the fields. Results that exist are kept, so Start after Stop goes on where it stopped.
// During a run, Start does nothing: a second job in the same page would open a second GPU device next to the first.
view.controls.start.addEventListener('click', () => {
    if (loadState().pending) {
        say('a run is going on. Wait for the page to reload.');
        return;
    }
    clearTimeout(scheduled);
    const runtimes = view.runtimeBoxes.filter(({ box }) => box.checked).map(({ runtime }) => runtime);
    saveState({ ...loadState(), plan: { runtimes, runs: Number(view.runsField.value), warm: Number(view.warmField.value) } });
    render();
    continuePlan();
});
// Stop after the run that goes on now. The results stay.
view.controls.stop.addEventListener('click', () => {
    clearTimeout(scheduled);
    saveState({ ...loadState(), plan: null });
    say('stopped. Press Start to go on.');
});
// Forget every result.
view.controls.reset.addEventListener('click', () => {
    clearTimeout(scheduled);
    clearState();
    render();
    say('all results deleted.');
});
view.controls.csv.addEventListener('click', () => download('sd-turbo-benchmark.csv', 'text/csv', reportCsv(loadState())));
view.controls.markdown.addEventListener('click', () => download('sd-turbo-benchmark.md', 'text/markdown', reportMarkdown(loadState())));

render();
continuePlan();
