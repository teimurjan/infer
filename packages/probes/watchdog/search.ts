// Double the single-submit dial until the device dies, then bisect between the bounds.

// Stop when the upper bound is at most 10% above the lower bound.
const PRECISION = 1.1;
// Outcomes that count as "the submit was too long".
const DEATHS = new Set(['lost', 'hang', 'dead-tab']);

// One measured point: a number of iterations and how long it took (or was expected to take).
export type Bound = { iterations: number; ms?: number };

export type SearchState = {
    // Iterations per millisecond, measured by calibration. null until calibration ran.
    ratePerMs: number | null;
    // The longest single submit that finished.
    lower: Bound | null;
    // The shortest single submit that died.
    upper: Bound | null;
};

export type SearchSettings = { startMs: number; maxMs: number };

// Only these fields of a run matter for the search.
type SearchRun = { mode: string; outcome: string; iterations: number; gpuMs?: number; wallMs?: number; estimatedMs?: number };

// The next iteration count to try, or null when the search is done.
export function nextIterations({ ratePerMs, lower, upper }: SearchState, { startMs, maxMs }: SearchSettings): number | null {
    if (!ratePerMs) {
        return null;
    }
    if (!upper) {
        const next = lower ? lower.iterations * 2 : Math.max(1, Math.round(ratePerMs * startMs));
        return next / ratePerMs > maxMs ? null : next;
    }
    if (!lower) {
        const next = Math.floor(upper.iterations / 2);
        return next >= 1 ? next : null;
    }
    if (upper.iterations <= lower.iterations * PRECISION) {
        return null;
    }
    // The geometric mean splits the range evenly on a log scale.
    const next = Math.round(Math.sqrt(lower.iterations * upper.iterations));
    return next > lower.iterations && next < upper.iterations ? next : null;
}

// Moves the lower or upper bound if this single run is a better bound than the current one.
export function withBounds<State extends SearchState>(state: State, run: SearchRun): State {
    if (run.mode !== 'single') {
        return state;
    }
    if (run.outcome === 'ok' && (!state.lower || run.iterations > state.lower.iterations)) {
        return { ...state, lower: { iterations: run.iterations, ms: run.gpuMs ?? run.wallMs } };
    }
    if (DEATHS.has(run.outcome) && (!state.upper || run.iterations < state.upper.iterations)) {
        return { ...state, upper: { iterations: run.iterations, ms: run.estimatedMs } };
    }
    return state;
}
