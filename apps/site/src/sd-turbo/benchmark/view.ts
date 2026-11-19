// The parts of the benchmark page, in the dictionary world: the plan, the controls, the log, the last image and the table.

import { LABEL, RUNTIMES, type Runtime } from '@infer/sd-turbo/benchmark/runners';
import { SIDE } from '@infer/sd-turbo/benchmark/image';
import { button } from '@infer/ui/components/button';
import { log as logView } from '@infer/ui/components/log';
import { h } from '@infer/ui/dom';

export function benchmarkView() {
    const runtimeBoxes = RUNTIMES.map(runtime => ({ runtime, box: h('input', { type: 'checkbox', checked: true, style: 'accent-color: var(--ink); width: 20px; height: 20px' }) }));
    const runsField = h('input', { class: 'ui-input ui-num', type: 'number', min: 1, value: 10, inputmode: 'numeric', style: 'max-width: 7rem' });
    const warmField = h('input', { class: 'ui-input ui-num', type: 'number', min: 0, value: 3, inputmode: 'numeric', style: 'max-width: 7rem' });
    const controls = {
        start: button('start', { kind: 'primary' }),
        stop: button('stop', { kind: 'secondary' }),
        reset: button('reset', { kind: 'secondary' }),
        csv: button('csv', { kind: 'secondary', disabled: true }),
        markdown: button('markdown', { kind: 'secondary', disabled: true }),
    };
    const output = logView();
    const canvas = h('canvas', { width: SIDE, height: SIDE, role: 'img', 'aria-label': 'The last image of the run' });
    const report = h('pre', { class: 'ui-log' });
    const element = h('div', { class: 'ui-stack', style: 'gap: var(--s-6)' },
        h('section', { class: 'site-open', style: 'padding-bottom: 0' },
            h('h1', { class: 'ui-headword ui-headword-entry' }, 'benchmark'),
            h('p', { class: 'ui-pron' }, h('span', { class: 'ui-pos' }, 'n., of sd-turbo')),
            h('p', { class: 'ui-small', style: 'max-width: var(--measure)' }, 'This engine against the onnxruntime-web example, with the same prompt and noise. Each run is a fresh page: it runs one job, saves the result, and reloads. A tab that iOS kills counts as a failed run.')),
        h('section', { class: 'ui-section', 'aria-labelledby': 'plan-label' },
            h('h2', { class: 'ui-section-label', id: 'plan-label' }, 'plan'),
            h('div', { class: 'ui-row', style: 'gap: var(--s-5)' }, ...runtimeBoxes.map(({ runtime, box }) => h('label', { class: 'ui-row ui-field', style: 'flex-direction: row' }, box, LABEL[runtime as Runtime]))),
            h('div', { class: 'ui-row', style: 'gap: var(--s-4)' }, h('label', { class: 'ui-field' }, 'counted runs', runsField), h('label', { class: 'ui-field' }, 'warm images per run', warmField)),
            // Two lines: the run controls, then the downloads of the table.
            h('div', { class: 'ui-row', style: 'gap: var(--s-2)' }, controls.start, controls.stop, controls.reset),
            h('div', { class: 'ui-row', style: 'gap: var(--s-2)' }, h('span', { class: 'ui-field' }, 'download'), controls.csv, controls.markdown)),
        h('div', { class: 'site-entry-page' },
            h('section', { class: 'ui-section', 'aria-labelledby': 'blog-label' }, h('h2', { class: 'ui-section-label', id: 'blog-label' }, 'log'), output.element),
            h('figure', { class: 'site-figure' }, h('div', { class: 'site-plate' }, canvas), h('figcaption', { class: 'site-caption' }, h('b', {}, 'fig. 1'), ' the last image of the run.'))),
        h('section', { class: 'ui-section', 'aria-labelledby': 'table-label' }, h('h2', { class: 'ui-section-label', id: 'table-label' }, 'results'), report),
    );
    return { element, runtimeBoxes, runsField, warmField, controls, output, canvas, report };
}
