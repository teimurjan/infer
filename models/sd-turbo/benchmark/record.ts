// The benchmark state, kept in localStorage, so it lives through the page reload after each run and through a killed tab.

import { RUNTIMES } from './runners.ts';
import type { Runtime } from './runners.ts';

// What the user started: the runtimes, the counted runs per runtime, and the warm images per run.
export type Plan = { runtimes: Runtime[]; runs: number; warm: number };

// One run of one runtime. Run 0 is the warm-up that fills the caches. It is saved, but the table leaves it out.
export type Job = { runtime: Runtime; run: number };

// The result of one run. coldMs is null if the first image failed. peakBytes is null if the tab died.
export type RunResult = Job & { ok: boolean; error: string | null; coldMs: number | null; warmMs: number[]; peakBytes: number | null };

// Everything the page keeps. `pending` is the run that is going on now. If the page starts and finds it set, that run never ended.
export type State = { plan: Plan | null; results: RunResult[]; pending: Job | null; userAgent: string };

// The localStorage key.
const KEY = 'sd-turbo-benchmark';

// The state at the start: nothing planned, nothing measured.
const empty = (): State => ({ plan: null, results: [], pending: null, userAgent: navigator.userAgent });

// Read the state, or the empty state if there is none or it cannot be read.
export function loadState(): State {
    try {
        const text = localStorage.getItem(KEY);
        return text ? JSON.parse(text) as State : empty();
    } catch {
        return empty();
    }
}

// Save the state. A failed save stops the run, because the next page would not know what happened.
export function saveState(state: State): void {
    localStorage.setItem(KEY, JSON.stringify(state));
}

// Forget everything.
export function clearState(): State {
    localStorage.removeItem(KEY);
    return empty();
}

// The run that never ended, as a failed result. The tab was killed (for example by iOS for memory) or reloaded by hand.
export function diedResult(job: Job): RunResult {
    return { ...job, ok: false, error: 'the page ended during this run: the tab was killed or reloaded', coldMs: null, warmMs: [], peakBytes: null };
}

// The next run of the plan that has no result yet, or null when the plan is done.
export function nextJob(state: State): Job | null {
    const { plan } = state;
    if (!plan) {
        return null;
    }
    // Every runtime of the plan in run order, runs 0 (warm-up) to plan.runs.
    const jobs = RUNTIMES.filter(runtime => plan.runtimes.includes(runtime))
        .flatMap(runtime => Array.from({ length: plan.runs + 1 }, (_, run) => ({ runtime, run })));
    return jobs.find(job => !state.results.some(result => result.runtime === job.runtime && result.run === job.run)) ?? null;
}
