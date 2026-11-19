// The numbered lines of a run. A line that changes state snaps into place by one line, with a small overshoot.

import { h } from '../dom.ts';

export type StepState = 'pending' | 'active' | 'done' | 'failed';

export type Step = { set(state: StepState, detail?: string, time?: string): void };
export type Steps = { element: HTMLOListElement; steps: Step[] };

// Make the list from the step titles and their first detail lines. Steps are numbered from 1.
export function steps(items: readonly { title: string; detail: string }[]): Steps {
    const rows = items.map((item, index) => {
        const detail = h('div', { class: 'ui-step-detail' }, item.detail);
        const time = h('div', { class: 'ui-step-time' }, '');
        const body = h('div', {}, h('div', { class: 'ui-step-title' }, item.title), detail);
        const row = h('li', { class: 'ui-step', 'data-state': 'pending' }, h('span', { class: 'ui-step-dot', 'data-n': index + 1, 'aria-hidden': 'true' }), body, time);
        const set = (state: StepState, nextDetail?: string, nextTime?: string): void => {
            const changed = row.dataset.state !== state;
            row.dataset.state = state;
            if (nextDetail !== undefined) detail.textContent = nextDetail;
            if (nextTime !== undefined) time.textContent = nextTime;
            if (!changed) return;
            // Restart the snap: remove the class, force a layout, and add it again.
            row.classList.remove('ui-snap');
            void row.offsetWidth;
            row.classList.add('ui-snap');
        };
        return { row, step: { set } };
    });
    return { element: h('ol', { class: 'ui-steps' }, ...rows.map(row => row.row)), steps: rows.map(row => row.step) };
}
