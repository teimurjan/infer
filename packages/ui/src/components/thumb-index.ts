// The thumb index: cut tabs on the right edge of the page, one per entry, as on a dictionary's fore-edge.
// One empty cut after the last tab shows where the next entry will go.

import { h } from '../dom.ts';

export type Thumb = { letter: string; label: string; href: string };

// Make the tabs. Each entry runs, so each tab is blue. The tab of the entry this page belongs to sticks out further.
export function thumbIndex(thumbs: readonly Thumb[], current: string | null): HTMLElement {
    const tabs = thumbs.map(thumb => h('a', { class: 'ui-thumb', href: thumb.href, 'aria-label': thumb.label, 'aria-current': thumb.href === current ? 'page' : undefined }, thumb.letter));
    return h('nav', { class: 'ui-thumbs', 'aria-label': 'Entries' }, ...tabs, h('span', { class: 'ui-thumb ui-thumb-empty', 'aria-hidden': 'true' }));
}
