// The use of the entry: the prompt, example prompts, the seed and the run button.

import { button } from '@infer/ui/components/button';
import { h } from '@infer/ui/dom';

// The largest seed: the seed is a 32-bit number.
const MAX_SEED = 4294967295;

// Prompts that SD-Turbo draws well in one step. The first one is the golden prompt of the tests.
const EXAMPLES = [
    'a photo of a cat sitting on a wooden table',
    'a lighthouse on a cliff at sunset, oil painting',
    'a bowl of ramen, studio photo, soft light',
    'a red fox in fresh snow, wildlife photo',
];

export type Request = { prompt: string; seed: number };

export type CreatePanel = {
    element: HTMLElement;
    setEnabled(enabled: boolean, label?: string): void;
    onGenerate(handler: (request: Request) => void): void;
    // Put a prompt in the box, for example the one the home page sent.
    fill(prompt: string): void;
    // Press run from code, with the same checks as a tap.
    run(): void;
};

// A random seed from the browser's random source.
const randomSeed = (): number => crypto.getRandomValues(new Uint32Array(1))[0]!;

export function createPanel(): CreatePanel {
    const prompt = h('textarea', { class: 'ui-input', id: 'prompt', rows: 3, maxlength: 400, autocapitalize: 'off', spellcheck: 'false' }, EXAMPLES[1]!);
    const seed = h('input', { class: 'ui-input ui-num', id: 'seed', type: 'number', inputmode: 'numeric', min: 0, max: MAX_SEED, value: 42, style: 'max-width: 11rem' });
    const examples = h('div', { class: 'site-examples' }, ...EXAMPLES.map(text => h('button', { type: 'button', onclick: () => { prompt.value = text; } }, `→ ${text}`)));
    const generate = button('run', { kind: 'primary', large: true, disabled: true });
    const hint = h('p', { class: 'ui-small', 'aria-live': 'polite' }, 'Store the weights first.');
    const element = h('section', { class: 'ui-section', 'aria-labelledby': 'use-label' },
        h('h2', { class: 'ui-section-label', id: 'use-label' }, 'use'),
        h('label', { class: 'ui-field', for: 'prompt' }, 'prompt', prompt),
        examples,
        h('div', { class: 'ui-row', style: 'align-items: end; gap: var(--s-3)' },
            h('label', { class: 'ui-field', for: 'seed' }, 'seed', seed),
            button('new seed', { kind: 'ghost', onClick: () => { seed.value = String(randomSeed()); } }),
            h('span', { style: 'flex: 1' }),
            generate),
        h('p', { class: 'ui-field-hint' }, 'The same prompt and seed always give the same image.'),
        hint);

    const setEnabled = (enabled: boolean, label?: string): void => {
        generate.disabled = !enabled;
        if (label !== undefined) hint.textContent = label;
    };
    // The handler of a checked request. A bad seed or an empty prompt stops here, with the reason next to the button.
    let handle: ((request: Request) => void) | null = null;
    const run = (): void => {
        const value = Number(seed.value);
        if (!Number.isInteger(value) || value < 0 || value > MAX_SEED) {
            hint.textContent = `The seed must be a whole number from 0 to ${MAX_SEED}.`;
            return;
        }
        if (!prompt.value.trim()) {
            hint.textContent = 'Write a prompt first.';
            return;
        }
        handle?.({ prompt: prompt.value.trim(), seed: value });
    };
    generate.addEventListener('click', run);
    const onGenerate = (handler: (request: Request) => void): void => {
        handle = handler;
    };
    const fill = (text: string): void => {
        prompt.value = text;
    };
    return { element, setEnabled, onGenerate, fill, run };
}
