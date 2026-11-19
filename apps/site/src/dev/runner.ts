// The developer page of one test suite. A person taps Run, or the harness opens ?test=<id>&run=<id> and the test starts by itself.
// Every line, number and result goes to the Mac (runs/<run>.jsonl), so each run on the phone is recorded.

import { reporter } from '@infer/browser/report';
import type { TestCase, TestOutcome, TestSuite } from '@infer/browser/test-suite';
import { badge, type Badge } from '@infer/ui/components/badge';
import { button } from '@infer/ui/components/button';
import { log as logView } from '@infer/ui/components/log';
import { h } from '@infer/ui/dom';
import { formatSeconds } from '@infer/ui/format';
import { mountPage } from '../shell/page.ts';
import { checkSupport } from '../shell/support.ts';

// The id of "run every test" in the page address.
const ALL = 'all';

// One row of the list: the test, its status badge, its time and its Run button.
type Row = { test: TestCase; status: Badge; time: HTMLElement; run: HTMLButtonElement };

// The text of a thrown value. The engine throws JavaScript Errors with the Rust message.
const errorText = (error: unknown): string => (error instanceof Error ? error.message : String(error));

// Put the suite on the page. `extras` are more controls of the model, for example "Clear the cache".
export async function mountSuite(suite: TestSuite, extras: readonly HTMLElement[] = []): Promise<void> {
    const report = reporter(`${suite.model}/dev`);
    const output = logView();
    const rows: Row[] = suite.tests.map(test => ({ test, status: badge('not run'), time: h('span', { class: 'ui-num ui-small ui-muted' }, ''), run: button('run') }));
    // The GPU line fills in after the check, so the page shows at once.
    const gpuLine = h('p', { class: 'ui-small ui-muted ui-num' }, 'GPU: checking…');
    const runAll = button('run all', { kind: 'primary' });
    const copy = button('copy log', { kind: 'ghost', onClick: () => void navigator.clipboard?.writeText(output.text()) });

    mountPage({ title: `${suite.model} · tests`, guide: 'tests', entry: suite.model },
        h('section', { class: 'site-open', style: 'padding-bottom: var(--s-5)' },
            h('h1', { class: 'ui-headword ui-headword-entry' }, 'tests'),
            h('p', { class: 'ui-pron' }, h('span', { class: 'ui-pos' }, `n. pl., of ${suite.model}`)),
            h('p', { class: 'ui-small', style: 'max-width: var(--measure)' }, `Each test runs on the GPU of this device and sends every line to the Mac, into runs/${report.run}.jsonl.`),
            gpuLine,
            h('div', { class: 'ui-row', style: 'gap: var(--s-3)' }, runAll, ...extras, copy)),
        h('section', { class: 'ui-section', 'aria-labelledby': 'list-label' },
            h('h2', { class: 'ui-section-label', id: 'list-label' }, `${rows.length} tests`),
            h('div', {}, ...rows.map((row, index) =>
                h('div', { class: 'site-test' },
                    h('div', { class: 'ui-row', style: 'gap: var(--s-3); align-items: baseline' }, h('span', { class: 'ui-sense-n' }, String(index + 1)), h('h3', {}, row.test.title), row.status.element, row.time),
                    row.run,
                    h('p', {}, row.test.description))))),
        h('section', { class: 'ui-section', style: 'margin-top: var(--s-6)', 'aria-labelledby': 'log-label' }, h('h2', { class: 'ui-section-label', id: 'log-label' }, 'log'), output.element),
    );

    // Print a line on the page and send it to the Mac.
    const say = (line: string, tone: 'normal' | 'danger' | 'success' | 'muted' = 'normal'): void => {
        output.add(line, tone);
        report.send({ kind: 'line', text: line });
    };

    // Run one test with its clock. A thrown error is a failed test, with the error as the reason.
    const runOne = async (row: Row): Promise<TestOutcome> => {
        row.status.set('running', 'accent');
        say(`${row.test.id}: started`, 'muted');
        const started = performance.now();
        const outcome = await row.test.run({
            say: line => say(`${row.test.id}: ${line}`),
            metric: (name, value, unit) => report.send({ kind: 'metric', name: `${row.test.id}: ${name}`, value, unit }),
        }).catch((error: unknown): TestOutcome => ({ passed: false, summary: errorText(error) }));
        row.time.textContent = formatSeconds((performance.now() - started) / 1000);
        row.status.set(outcome.passed ? 'passed' : 'failed', outcome.passed ? 'success' : 'danger');
        say(`${row.test.id}: ${outcome.passed ? 'PASSED' : 'FAILED'}: ${outcome.summary}`, outcome.passed ? 'success' : 'danger');
        return outcome;
    };

    // Run tests one after the other, with every button off, and send "done" at the end.
    const runMany = async (chosen: readonly Row[]): Promise<void> => {
        const buttons = [runAll, ...rows.map(row => row.run)];
        buttons.forEach(control => { control.disabled = true; });
        const outcomes: TestOutcome[] = [];
        for (const row of chosen) outcomes.push(await runOne(row));
        const passed = outcomes.filter(outcome => outcome.passed).length;
        const summary = chosen.length === 1 ? outcomes[0]!.summary : `${passed} of ${chosen.length} tests passed`;
        report.send({ kind: 'done', passed: passed === chosen.length, summary });
        buttons.forEach(control => { control.disabled = false; });
    };

    const support = await checkSupport();
    gpuLine.textContent = `GPU: ${support.ok ? support.adapter : support.reason}`;
    rows.forEach(row => row.run.addEventListener('click', () => void runMany([row])));
    runAll.addEventListener('click', () => void runMany(rows));
    report.send({ kind: 'start', userAgent: navigator.userAgent });
    say(`${navigator.userAgent}`, 'muted');
    if (!support.ok) {
        say(support.reason, 'danger');
        report.send({ kind: 'done', passed: false, summary: support.reason });
        return;
    }

    // The harness names the test in the address. Without it, the page waits for a tap.
    const wanted = new URLSearchParams(location.search).get('test');
    if (!report.automated || !wanted) return;
    // A page that loads a second time with the same run id was reloaded during the run: iOS killed the tab, most likely.
    // localStorage keeps the mark across that reload, so the test does not start again in a loop.
    const mark = `harness-run:${report.run}`;
    if (localStorage.getItem(mark)) {
        say('the page loaded again during this run: the tab was killed or reloaded', 'danger');
        report.send({ kind: 'done', passed: false, summary: 'the page loaded again during the run' });
        return;
    }
    localStorage.setItem(mark, String(Date.now()));
    const chosen = wanted === ALL ? rows : rows.filter(row => row.test.id === wanted);
    if (chosen.length === 0) {
        say(`no test with the id ${wanted}`, 'danger');
        report.send({ kind: 'done', passed: false, summary: `no test with the id ${wanted}` });
        return;
    }
    await runMany(chosen);
}
