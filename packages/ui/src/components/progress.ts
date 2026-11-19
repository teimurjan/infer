// A progress bar with a label on the left and a value on the right.

import { h } from '../dom.ts';

// running: the fill shows the fraction. unknown: the total is not known, so the fill moves. done and failed: the end colors.
export type ProgressState = 'running' | 'unknown' | 'done' | 'failed';

export type Progress = { element: HTMLDivElement; set(fraction: number | null, value: string, state?: ProgressState): void; setLabel(label: string): void };

// Make a progress bar. set(0.5, "433 of 866 MB") fills half of it. set(null, ...) shows that the total is not known.
export function progress(label: string): Progress {
    const fill = h('div', { class: 'ui-progress-fill' });
    const track = h('div', { class: 'ui-progress-track', role: 'progressbar', 'aria-label': label, 'aria-valuemin': 0, 'aria-valuemax': 100 }, fill);
    const labelText = h('span', {}, label);
    const valueText = h('span', { class: 'ui-mono' }, '');
    const element = h('div', { class: 'ui-progress', 'data-state': 'running' }, h('div', { class: 'ui-progress-label' }, labelText, valueText), track);
    const set = (fraction: number | null, value: string, state: ProgressState = fraction === null ? 'unknown' : 'running'): void => {
        // Keep the fraction between 0 and 1, so a wrong length never overflows the track.
        const percent = fraction === null ? 0 : Math.round(Math.min(Math.max(fraction, 0), 1) * 100);
        fill.style.width = `${percent}%`;
        track.setAttribute('aria-valuenow', String(percent));
        valueText.textContent = value;
        element.dataset.state = state;
    };
    const setLabel = (next: string): void => {
        labelText.textContent = next;
    };
    return { element, set, setLabel };
}
