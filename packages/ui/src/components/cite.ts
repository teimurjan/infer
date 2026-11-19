// Citations: every measured number carries a small mark that points to its source run at the foot of the page.

import { h } from '../dom.ts';

export type Source = { id: string; text: string };

// The small raised mark after a number. It links to the source in the list.
export const citeMark = (source: Source, number: number): HTMLAnchorElement =>
    h('a', { class: 'ui-cite-mark', href: `#cite-${source.id}`, 'aria-label': `source ${number}` }, String(number));

// The numbered list of sources, in the order given.
export const citeList = (sources: readonly Source[]): HTMLOListElement =>
    h('ol', { class: 'ui-cites' }, ...sources.map((source, index) => h('li', { id: `cite-${source.id}` }, `${index + 1}. ${source.text}`)));
