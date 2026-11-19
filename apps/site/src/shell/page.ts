// The frame of every site page: the styles, the running head, the thumb index, the reverse page, and the main column.

import '@infer/ui/styles.css';
import './site.css';
import { runningHead, type HeadLink, type RunningHead } from '@infer/ui/components/running-head';
import { showThrough } from '@infer/ui/components/show-through';
import { thumbIndex } from '@infer/ui/components/thumb-index';
import { h } from '@infer/ui/dom';
import { entries } from './entries.ts';
import { links } from './links.ts';
import { notes } from './notes.ts';

export type PageOptions = {
    // The tab title after "infer / ", or null for the home page, whose title is "infer".
    title: string | null;
    // The guide word on the right of the running head: the last headword on the page.
    guide: string;
    // The entry this page belongs to, for its blue thumb tab and its page links. Null on the home page.
    entry: string | null;
};

export type Page = { main: HTMLElement; head: RunningHead };

// Put the frame on the page, with `children` in the main column.
export function mountPage({ title, guide, entry }: PageOptions, ...children: Node[]): Page {
    document.title = title ? `infer / ${title}` : 'infer';
    // An entry page links to its tests and benchmark from the running head.
    const pageLinks: HeadLink[] = entry ? [{ label: 'entry', href: links.entry(entry) }, { label: 'tests', href: links.tests(entry) }, { label: 'bench', href: links.benchmark(entry) }] : [];
    const head = runningHead(links.home, guide, pageLinks);
    const main = h('main', { class: 'ui-page' }, ...children);
    const thumbs = thumbIndex(entries.map(item => ({ letter: item.letter, label: item.id, href: links.entry(item.id) })), entry ? links.entry(entry) : null);
    const reverse = showThrough([...entries.map(item => ({ word: item.id, text: `${item.pos} ${item.definition}` })), ...notes.map(note => ({ word: note.word, text: `${note.pos} ${note.text}` }))]);
    document.body.replaceChildren(reverse, head.element, thumbs, main);
    return { main, head };
}
