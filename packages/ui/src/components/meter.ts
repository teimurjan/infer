// A meter: how much of a budget is in use, for example GPU memory against the tab limit of the phone.

import { h } from '../dom.ts';

export type Meter = { element: HTMLDivElement; set(used: number, label: string): void };

// Make a meter for a budget of `limit`. Over 80% it turns amber, over 95% red.
export function meter(title: string, limit: number): Meter {
    const fill = h('div', { class: 'ui-meter-fill' });
    const valueText = h('span', { class: 'ui-mono' }, '');
    const element = h('div', { class: 'ui-meter ui-stack-tight', role: 'meter', 'aria-label': title, 'aria-valuemin': 0, 'aria-valuemax': limit },
        h('div', { class: 'ui-progress-label' }, h('span', {}, title), valueText), h('div', { class: 'ui-meter-track' }, fill));
    const set = (used: number, label: string): void => {
        const fraction = Math.min(used / limit, 1);
        fill.style.width = `${(fraction * 100).toFixed(1)}%`;
        element.dataset.tone = fraction > 0.95 ? 'danger' : fraction > 0.8 ? 'warning' : 'normal';
        element.setAttribute('aria-valuenow', String(used));
        valueText.textContent = label;
    };
    return { element, set };
}
