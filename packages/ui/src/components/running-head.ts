// The running head of a dictionary page: "infer" on the left links home, the guide word on the right names the page.
// Links to the other pages of the same entry sit beside the guide word.

import { h } from '../dom.ts';

export type HeadLink = { label: string; href: string };

export type RunningHead = { element: HTMLElement; setGuide(word: string): void };

// Make the head. The link whose address matches this page gets aria-current.
export function runningHead(home: string, guide: string, links: readonly HeadLink[] = []): RunningHead {
    const guideWord = h('span', { class: 'ui-guide' }, guide);
    const here = location.pathname;
    const nav = links.length === 0 ? null : h('nav', { 'aria-label': 'Pages of this entry' }, ...links.map(link =>
        h('a', { href: link.href, 'aria-current': here === new URL(link.href, location.href).pathname ? 'page' : undefined }, link.label)));
    const element = h('header', { class: 'ui-running-head' },
        h('div', { class: 'ui-running-head-row' }, h('a', { href: home, 'aria-label': 'infer, home' }, 'infer'), h('div', { class: 'ui-row' }, nav, guideWord)));
    // A new guide word fades in, as the head changes when you turn to the next page.
    const setGuide = (word: string): void => {
        if (guideWord.textContent === word) return;
        guideWord.style.opacity = '0';
        requestAnimationFrame(() => {
            guideWord.textContent = word;
            guideWord.style.opacity = '1';
        });
    };
    return { element, setGuide };
}
