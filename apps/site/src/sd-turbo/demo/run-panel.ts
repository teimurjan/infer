// This run: the four numbered parts of the pipeline, stepping one line at a time, the load band of the current file,
// and the budget rule of the GPU memory against the tab limit.

import type { Engine } from '@infer/sd-turbo/client/engine-types';
import type { PipelineEvent, Stage } from '@infer/sd-turbo/client/pipeline';
import type { Entry } from '@infer/browser/entry';
import { meter } from '@infer/ui/components/meter';
import { progress } from '@infer/ui/components/progress';
import { steps } from '@infer/ui/components/steps';
import { h } from '@infer/ui/dom';
import { formatBytes, formatSeconds } from '@infer/ui/format';

// The tab limit of Safari on the iPhone 16 Pro Max (CLAUDE.md): JS, WASM and GPU memory share it.
const TAB_LIMIT_BYTES = 1920e6;

// The line of each pipeline stage. Line 0 is the tokenizer, which runs before the pipeline.
const LINE_OF: Record<Stage, number> = { 'text-encoder': 1, unet: 2, decoder: 3 };

export type RunPanel = {
    element: HTMLElement;
    // A new image starts. tokens: how many of the 77 token slots the prompt uses.
    begin(tokens: number): void;
    onEvent(event: PipelineEvent): void;
    finish(seconds: number): void;
    fail(message: string): void;
    // Read the live GPU bytes of the engine about 4 times per second, until the returned function is called.
    watch(engine: Engine): () => void;
};

export function runPanel(entry: Entry): RunPanel {
    const view = steps(entry.parts.map(part => ({ title: part.name, detail: part.does })));
    const load = progress('weights');
    load.element.hidden = true;
    const memory = meter('GPU memory against the tab limit', TAB_LIMIT_BYTES);
    memory.set(0, `0 MB of ${formatBytes(TAB_LIMIT_BYTES)}`);
    const element = h('section', { class: 'ui-section', 'aria-labelledby': 'run-label' },
        h('h2', { class: 'ui-section-label', id: 'run-label' }, 'this run, on this device'),
        view.element, load.element, memory.element);
    // The line that runs now, so a failure marks the right one.
    let current = 0;

    const begin = (tokens: number): void => {
        view.steps.forEach((step, index) => step.set('pending', entry.parts[index]!.does, ''));
        view.steps[0]!.set('done', `${tokens} of 77 token slots used`, '');
        current = 1;
        view.steps[1]!.set('active', 'loading its weights');
    };
    const onEvent = (event: PipelineEvent): void => {
        if (event.kind === 'load') {
            const { from, received, length } = event.progress;
            load.element.hidden = false;
            load.setLabel(`${event.file} from ${from === 'cache' ? 'storage' : 'the network'}`);
            load.set(length ? received / length : null, length ? `${formatBytes(received)} of ${formatBytes(length)}` : formatBytes(received));
        }
        if (event.kind === 'loaded') load.element.hidden = true;
        if (event.kind === 'stage-start') {
            current = LINE_OF[event.stage];
            view.steps[current]!.set('active', 'running on the GPU');
        }
        if (event.kind === 'stage-end') view.steps[LINE_OF[event.stage]]!.set('done', entry.parts[LINE_OF[event.stage]]!.does, formatSeconds(event.seconds));
        // The next stage loads its weights before it starts, and that line shows it.
        if (event.kind === 'stage-end' && LINE_OF[event.stage] < view.steps.length - 1) view.steps[LINE_OF[event.stage] + 1]!.set('active', 'loading its weights');
    };
    const finish = (seconds: number): void => {
        load.element.hidden = true;
        memory.set(0, `all freed · ${formatSeconds(seconds)} in total`);
    };
    const fail = (message: string): void => {
        load.element.hidden = true;
        view.steps[current]!.set('failed', message);
    };
    const watch = (engine: Engine): (() => void) => {
        let peak = 0;
        const timer = setInterval(() => {
            const live = engine.liveBytes();
            peak = Math.max(peak, live);
            memory.set(live, `${formatBytes(live)} now · ${formatBytes(peak)} peak · of ${formatBytes(TAB_LIMIT_BYTES)}`);
        }, 250);
        return () => clearInterval(timer);
    };
    return { element, begin, onEvent, finish, fail, watch };
}
