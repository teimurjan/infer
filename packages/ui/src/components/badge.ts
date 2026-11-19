// A small status label, for example "passed" in green.

import { h } from '../dom.ts';

// The tones match the colors of the tokens. neutral is gray.
export type Tone = 'neutral' | 'success' | 'danger' | 'warning' | 'accent';

export type Badge = { element: HTMLSpanElement; set(label: string, tone: Tone): void };

// Make a badge. set() changes its text and color later.
export function badge(label: string, tone: Tone = 'neutral'): Badge {
    const element = h('span', { class: 'ui-badge', 'data-tone': tone }, label);
    const set = (next: string, nextTone: Tone): void => {
        element.textContent = next;
        element.dataset.tone = nextTone;
    };
    return { element, set };
}
