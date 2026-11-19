// The reverse page: the real text of the book, mirrored and faint behind the page, as thin dictionary paper shows it.
// Screen readers skip it, because it repeats text that the page already has.

import { h } from '../dom.ts';

export type Term = { word: string; text: string };

// The text of one column: the terms from `start` on, around and around, enough to fill a screen of 1600 px in a narrow column.
function column(terms: readonly Term[], start: number): HTMLElement {
    const nodes = Array.from({ length: terms.length * 6 }, (_, index) => terms[(start + index) % terms.length]!)
        .flatMap(term => [h('b', {}, term.word), ` ${term.text} `]);
    return h('div', { class: 'ui-show-through-col' }, ...nodes);
}

// Make the layer: three columns, each starting at a different term, so the columns do not repeat side by side.
export function showThrough(terms: readonly Term[]): HTMLElement {
    const third = Math.max(1, Math.floor(terms.length / 3));
    return h('div', { class: 'ui-show-through', 'aria-hidden': 'true' }, column(terms, 0), column(terms, third), column(terms, third * 2));
}
