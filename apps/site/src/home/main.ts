// The home page: the headword "infer", what it means here, every inference that runs today, and the usage notes.

import type { Entry } from '@infer/browser/entry';
import { isCached, openCache } from '@infer/sd-turbo/client/cache';
import { weightSize } from '@infer/sd-turbo/client/weights';
import { button } from '@infer/ui/components/button';
import { citeList, citeMark, type Source } from '@infer/ui/components/cite';
import { h } from '@infer/ui/dom';
import { formatBytes } from '@infer/ui/format';
import { entries } from '../shell/entries.ts';
import { links } from '../shell/links.ts';
import { limitsSource, notes } from '../shell/notes.ts';
import { mountPage } from '../shell/page.ts';

// The source of each entry's measured times, numbered after the limits source.
const runSources: Source[] = entries.map(entry => ({ id: `run-${entry.id}`, text: `${entry.id}: ${entry.measured.device}, ${entry.measured.browser}, ${entry.measured.date}, ${entry.measured.source}.` }));
const sources = [limitsSource, ...runSources];

// The contents map of an entry: numbered parts with their sizes. The sizes arrive later, from the server.
function contents(entry: Entry): { element: HTMLElement; total: Promise<number> } {
    const sizes = entry.parts.map(() => h('span', { class: 'ui-num' }, entry.parts.length > 0 ? '…' : ''));
    const element = h('div', { class: 'ui-contents', role: 'list', 'aria-label': `Parts of ${entry.id}` },
        ...entry.parts.flatMap((part, index) => [h('span', {}, String(index + 1)), h('span', { role: 'listitem' }, part.name), sizes[index]!]));
    // The size of each weight file, from a HEAD request. A part without a file has no size.
    const total = Promise.all(entry.parts.map(async (part, index) => {
        const bytes = part.file ? await weightSize(part.file).catch(() => null) : null;
        sizes[index]!.textContent = part.file ? (bytes ? formatBytes(bytes) : '?') : (part.note ?? '');
        return bytes ?? 0;
    })).then(all => all.reduce((sum, bytes) => sum + bytes, 0));
    return { element, total };
}

// One entry: headword, part of speech, definition, contents, the prompt line that runs it, and its storage band.
function entryBlock(entry: Entry, sourceNumber: number): HTMLElement {
    const map = contents(entry);
    const band = h('p', { class: 'site-band' }, 'checking this device…');
    // The total time of the parts that run, with its source.
    const seconds = entry.measured.seconds.map(item => item.seconds.toFixed(2)).join(' + ');
    const measured = h('p', { class: 'site-band' }, `on ${entry.measured.device}: `, h('strong', { class: 'ui-num' }, `${seconds} s`), citeMark(runSources[sourceNumber - 2]!, sourceNumber));
    // The prompt goes to the entry page after "#", so the browser never sends it to the server.
    const prompt = h('input', { class: 'ui-input', name: 'prompt', 'aria-label': entry.input.label, placeholder: entry.input.example, autocomplete: 'off', autocapitalize: 'off', enterkeyhint: 'go', required: true });
    const run = button('run', { kind: 'primary' });
    run.type = 'submit';
    const form = h('form', { class: 'ui-run-line', action: links.entry(entry.id) }, prompt, run);
    form.addEventListener('submit', event => {
        event.preventDefault();
        location.href = `${links.entry(entry.id)}#prompt=${encodeURIComponent(prompt.value.trim())}`;
    });
    void Promise.all([map.total, openCache().then(folder => Promise.all(entry.parts.filter(part => part.file).map(part => isCached(folder, part.file!))))])
        .then(([total, stored]) => {
            band.replaceChildren(stored.every(Boolean) ? h('strong', {}, 'stored on this device') : h('strong', {}, 'not stored yet'), stored.every(Boolean) ? ' · runs at once' : ` · ${formatBytes(total)}, downloaded once`);
        })
        .catch(() => band.replaceChildren('storage unknown in this browser'));
    return h('article', { class: 'site-entry', 'data-guide': entry.id, id: entry.id },
        h('h3', { class: 'ui-headword ui-headword-entry' }, h('a', { href: links.entry(entry.id), style: 'text-decoration:none' }, entry.id)),
        h('p', { class: 'ui-pron' }, h('span', { class: 'ui-pos' }, entry.pos)),
        h('p', {}, entry.definition),
        map.element, form, band, measured,
        h('p', { class: 'ui-small' }, h('a', { class: 'ui-see', href: links.tests(entry.id) }, '→ tests'), '  ', h('a', { class: 'ui-see', href: links.benchmark(entry.id) }, '→ benchmark')));
}

// The usage notes, each with its source mark when its numbers are measured.
const noteList = h('div', { class: 'site-notes' }, ...notes.map(note =>
    h('p', {}, h('b', {}, note.word), ' ', h('span', { class: 'ui-pos' }, note.pos), ' ', note.text, note.cited ? citeMark(limitsSource, 1) : null)));

// The guide word of a dictionary page is its last headword.
const lastHeadword = entries[entries.length - 1]?.id ?? 'infer';
const { main, head } = mountPage({ title: null, guide: lastHeadword, entry: null },
    h('section', { class: 'site-open site-open-home' },
        h('h1', { class: 'ui-headword' }, 'infer'),
        h('p', { class: 'ui-pron' }, h('span', {}, '/ɪnˈfɜː/'), h('span', { class: 'ui-pos' }, 'v.'), h('span', { class: 'ui-muted' }, 'infers, inferring, inferred')),
        h('ol', { class: 'ui-senses' },
            h('li', { class: 'ui-sense' }, h('span', { class: 'ui-sense-n' }, '1'), h('span', {}, 'to run a trained model forward on new input.')),
            h('li', { class: 'ui-sense' }, h('span', { class: 'ui-sense-n' }, '2'), h('span', {}, 'here: on the GPU of the phone in your hand, inside one Safari tab, with every kernel written by hand. ', h('a', { class: 'ui-see', href: '#entries' }, `→ ${entries.length} ${entries.length === 1 ? 'entry' : 'entries'}`), ', only what runs on a phone today.')))),
    h('div', { class: 'site-spread' },
        h('section', { class: 'ui-section', id: 'entries', 'aria-label': 'Entries' },
            ...entries.map((entry, index) => entryBlock(entry, index + 2))),
        h('section', { class: 'ui-section', 'data-guide': 'usage notes', 'aria-labelledby': 'notes-label' },
            h('h2', { class: 'ui-section-label', id: 'notes-label' }, 'usage notes'),
            noteList)),
    h('footer', { class: 'site-foot' },
        h('div', { class: 'ui-stack' }, h('h2', { class: 'ui-section-label' }, 'sources'), citeList(sources)),
        h('p', { class: 'site-colophon' }, 'infer is set in Archivo and Source Serif 4. Each engine is Rust compiled to WASM, with GPU kernels written by hand in WGSL, and no ML framework. The math runs on your device. Nothing you type leaves it.')),
);

// The guide word follows the part of the page you read, as a dictionary's running head does.
const observer = new IntersectionObserver(seen => {
    const visible = seen.filter(item => item.isIntersecting).map(item => (item.target as HTMLElement).dataset.guide);
    if (visible[0]) head.setGuide(visible[0]);
}, { rootMargin: '-20% 0px -70% 0px' });
main.querySelectorAll<HTMLElement>('[data-guide]').forEach(element => observer.observe(element));
