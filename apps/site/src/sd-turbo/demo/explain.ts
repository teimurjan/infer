// The usage notes of the entry: the two limits that shape how one image runs. The parts themselves are listed once, under "this run".

import { citeMark, type Source } from '@infer/ui/components/cite';
import { h } from '@infer/ui/dom';

export function explain(limits: Source, limitsNumber: number): HTMLElement {
    return h('section', { class: 'ui-section', 'aria-labelledby': 'how-label' },
        h('h2', { class: 'ui-section-label', id: 'how-label' }, 'usage'),
        h('p', { class: 'ui-small', style: 'max-width: var(--measure)' }, h('b', { class: 'ui-see' }, 'one part at a time.'), ' The text encoder and the UNet do not fit in one Safari tab together, so each part loads from storage, runs, and frees its memory before the next. ', h('a', { class: 'ui-see', href: '#run-label' }, '→ this run'), citeMark(limits, limitsNumber)),
        h('p', { class: 'ui-small', style: 'max-width: var(--measure)' }, h('b', { class: 'ui-see' }, 'short GPU jobs.'), ' iOS resets the GPU when one job runs half a second, so the engine cuts the work into jobs under 0.25 s.', citeMark(limits, limitsNumber)),
    );
}
