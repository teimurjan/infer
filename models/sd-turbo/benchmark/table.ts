// The summary table and the exports. Pure functions of the saved state.

import type { RunResult, State } from './record.ts';
import { LABEL, RUNTIMES } from './runners.ts';
import type { Runtime } from './runners.ts';

// Bytes in one MB, as in PROGRESS.md.
const MB = 1e6;

// One line of the summary.
type Summary = { runtime: Runtime; runs: number; successes: number; coldS: number | null; warmS: number | null; peakMb: number | null };

// The middle value, or null for an empty list. An even count gives the mean of the two middle values.
function median(values: number[]): number | null {
    if (values.length === 0) {
        return null;
    }
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

// The counted runs: run 0 is the warm-up, so it is left out.
const counted = (state: State): RunResult[] => state.results.filter(result => result.run > 0);

// One summary line per runtime that has counted runs.
// Cold: median over the successful runs. Warm: median over every warm image of the successful runs. Peak: largest over all runs with a number.
export function summarize(state: State): Summary[] {
    return RUNTIMES.map(runtime => {
        const runs = counted(state).filter(result => result.runtime === runtime);
        const good = runs.filter(result => result.ok);
        const cold = median(good.flatMap(result => result.coldMs === null ? [] : [result.coldMs]));
        const warm = median(good.flatMap(result => result.warmMs));
        const peaks = runs.flatMap(result => result.peakBytes === null ? [] : [result.peakBytes]);
        return {
            runtime,
            runs: runs.length,
            successes: good.length,
            coldS: cold === null ? null : cold / 1000,
            warmS: warm === null ? null : warm / 1000,
            peakMb: peaks.length === 0 ? null : Math.max(...peaks) / MB,
        };
    }).filter(summary => summary.runs > 0);
}

// A number with fixed decimals, or "-" when there is none.
const fixed = (value: number | null, digits: number): string => value === null ? '-' : value.toFixed(digits);

// A Markdown table from a header row and body rows.
function markdownTable(header: string[], rows: string[][]): string {
    const line = (cells: string[]) => `| ${cells.join(' | ')} |`;
    return [line(header), line(header.map(() => '---')), ...rows.map(line)].join('\n');
}

// The summary as a Markdown table.
export function summaryMarkdown(state: State): string {
    const rows = summarize(state).map(summary => [
        LABEL[summary.runtime], `${summary.successes} of ${summary.runs}`, fixed(summary.coldS, 2), fixed(summary.warmS, 2), fixed(summary.peakMb, 1),
    ]);
    return markdownTable(['runtime', 'successes', 'cold s (median)', 'warm s (median)', 'peak GPU buffers MB (max)'], rows);
}

// Every run as a Markdown table, with its error.
export function runsMarkdown(state: State): string {
    const rows = state.results.map(result => [
        result.runtime, result.run === 0 ? '0 (warm-up)' : String(result.run), result.ok ? 'yes' : 'no',
        fixed(result.coldMs === null ? null : result.coldMs / 1000, 2), result.warmMs.map(ms => (ms / 1000).toFixed(2)).join(' '),
        fixed(result.peakBytes === null ? null : result.peakBytes / MB, 1), result.error ?? '',
    ]);
    return markdownTable(['runtime', 'run', 'ok', 'cold s', 'warm s', 'peak MB', 'error'], rows);
}

// The whole report as Markdown: the device, the plan, the summary and every run.
export function reportMarkdown(state: State): string {
    const plan = state.plan ? `${state.plan.runs} runs per runtime, ${state.plan.warm} warm images per run` : 'no plan';
    return [`Browser: ${state.userAgent}`, `Plan: ${plan}`, '', summaryMarkdown(state), '', runsMarkdown(state), ''].join('\n');
}

// Every run as CSV, one line per run. Warm times are separated by spaces inside their cell.
export function reportCsv(state: State): string {
    const quote = (cell: string) => `"${cell.replaceAll('"', '""')}"`;
    const header = ['runtime', 'run', 'ok', 'cold_ms', 'warm_ms', 'peak_bytes', 'error'];
    const rows = state.results.map(result => [
        result.runtime, String(result.run), String(result.ok), result.coldMs === null ? '' : result.coldMs.toFixed(1),
        result.warmMs.map(ms => ms.toFixed(1)).join(' '), result.peakBytes === null ? '' : String(result.peakBytes), result.error ?? '',
    ]);
    return [header, ...rows].map(cells => cells.map(quote).join(',')).join('\n') + '\n';
}
