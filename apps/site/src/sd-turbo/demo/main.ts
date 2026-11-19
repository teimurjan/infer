// The entry page of SD-Turbo: what it is, its contents, a prompt to run, the run itself, and figure 1.

import { openCache } from '@infer/sd-turbo/client/cache';
import { loadTokenizer, openSession, type Session } from '@infer/sd-turbo/client/engine';
import { entry } from '@infer/sd-turbo/client/entry';
import { startingNoise } from '@infer/sd-turbo/client/noise';
import { generate } from '@infer/sd-turbo/client/pipeline';
import { SIGMA, TIMESTEP } from '@infer/sd-turbo/client/schedule';
import { citeList, citeMark, type Source } from '@infer/ui/components/cite';
import { h } from '@infer/ui/dom';
import { limitsSource } from '../../shell/notes.ts';
import { mountPage } from '../../shell/page.ts';
import { checkSupport } from '../../shell/support.ts';
import { createPanel, type Request } from './create-panel.ts';
import { explain } from './explain.ts';
import { imagePanel } from './image-panel.ts';
import { modelPanel } from './model-panel.ts';
import { runPanel } from './run-panel.ts';

// The sources of the numbers on this page.
const runSource: Source = { id: 'run', text: `${entry.measured.device}, ${entry.measured.browser}, ${entry.measured.date}, ${entry.measured.source}.` };
const sources = [runSource, limitsSource];

const notice = h('p', { class: 'ui-notice', hidden: true, role: 'status' });
const measured = entry.measured.seconds.map(item => `${item.part} ${item.seconds.toFixed(2)} s`).join(', ');
const { main } = mountPage({ title: entry.id, guide: entry.id, entry: entry.id },
    h('section', { class: 'site-open' },
        h('h1', { class: 'ui-headword ui-headword-entry' }, entry.id),
        h('p', { class: 'ui-pron' }, h('span', { class: 'ui-pos' }, entry.pos)),
        h('ol', { class: 'ui-senses' },
            h('li', { class: 'ui-sense' }, h('span', { class: 'ui-sense-n' }, '1'), h('span', {}, entry.definition)),
            h('li', { class: 'ui-sense' }, h('span', { class: 'ui-sense-n' }, '2'), h('span', {}, `on ${entry.measured.device}: ${measured}.`, citeMark(runSource, 1)))),
        h('p', { class: 'ui-small ui-muted' }, `[${entry.etymology}]`)),
    notice,
);

// The page shows its parts at once. The GPU check only decides whether "run" can work.
const folder = await openCache();
const model = modelPanel(entry, folder);
const create = createPanel();
const run = runPanel(entry);
const image = imagePanel();
main.append(
    h('div', { class: 'site-entry-page' }, h('div', { class: 'ui-stack', style: 'gap: var(--s-6)' }, model.element, create.element, run.element), h('div', { class: 'ui-stack', style: 'gap: var(--s-6)' }, image.element, explain(limitsSource, 2))),
    h('footer', { class: 'site-foot' }, h('div', { class: 'ui-stack' }, h('h2', { class: 'ui-section-label' }, 'sources'), citeList(sources))),
);

const support = await checkSupport();
if (!support.ok) {
    notice.textContent = support.reason;
    notice.hidden = false;
    create.setEnabled(false, support.reason);
} else {

    // The engine opens at the first image, so a visit that only reads the page never takes the GPU.
    let session: Promise<Session> | null = null;
    const tokenizer = loadTokenizer();

    // Make one image from the request, with every event on the run section.
    const makeImage = async (request: Request): Promise<void> => {
        create.setEnabled(false, 'Running.');
        session ??= openSession();
        const { engine } = await session;
        const stopWatch = run.watch(engine);
        try {
            const ids = (await tokenizer).encode(request.prompt);
            // The used slots: the ids before the trailing pad ids (0).
            run.begin(ids.findLastIndex(id => id !== 0) + 1);
            const inputs = { ids, noise: startingNoise(request.seed), sigma: SIGMA, timestep: TIMESTEP };
            const result = await generate(engine, inputs, folder, run.onEvent);
            run.finish(result.seconds);
            image.show(result.rgba, request, result.seconds);
            create.setEnabled(true, 'Done. Change the prompt or the seed for a new image.');
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            run.fail(message);
            // Free what the failed image left, so the next try starts empty.
            engine.freeWeights();
            create.setEnabled(true, `The run failed: ${message}. Try again.`);
        } finally {
            stopWatch();
        }
    };
    create.onGenerate(request => void makeImage(request));

    // A prompt from the home page arrives after "#prompt=". It runs as soon as the weights are stored.
    const sent = new URLSearchParams(location.hash.slice(1)).get('prompt');
    if (sent) create.fill(sent);
    model.ready.then(() => {
        create.setEnabled(true, 'Ready.');
        if (sent) create.run();
    }, () => create.setEnabled(false, 'The weights are not on this device.'));
}
