// Figure 1: the image of the last run, with its caption and a save link. Black until the first image lands.

import { button } from '@infer/ui/components/button';
import { h } from '@infer/ui/dom';
import { formatSeconds } from '@infer/ui/format';
import type { Request } from './create-panel.ts';

// The side of the image in pixels.
const SIDE = 512;

export type ImagePanel = { element: HTMLElement; show(rgba: Uint8Array, request: Request, seconds: number): void };

export function imagePanel(): ImagePanel {
    const canvas = h('canvas', { width: SIDE, height: SIDE, role: 'img', 'aria-label': 'No image yet' });
    const empty = h('div', { class: 'site-plate-empty' }, 'fig. 1 is drawn here, on this device, when you run the entry.');
    const caption = h('p', { class: 'site-caption' }, h('b', {}, 'fig. 1'), ' not drawn yet.');
    // The file name of the last image.
    let fileName = 'sd-turbo.png';
    const save = button('save image', { kind: 'ghost', disabled: true, onClick: () => {
        canvas.toBlob(blob => {
            if (!blob) return;
            const link = h('a', { href: URL.createObjectURL(blob), download: fileName });
            link.click();
            URL.revokeObjectURL(link.href);
        }, 'image/png');
    } });
    const element = h('figure', { class: 'site-figure' }, h('div', { class: 'site-plate' }, canvas, empty), h('figcaption', { class: 'ui-row', style: 'justify-content: space-between; align-items: baseline' }, caption, save));

    const show = (rgba: Uint8Array, request: Request, seconds: number): void => {
        canvas.getContext('2d')?.putImageData(new ImageData(new Uint8ClampedArray(rgba), SIDE, SIDE), 0, 0);
        canvas.setAttribute('aria-label', request.prompt);
        empty.hidden = true;
        caption.replaceChildren(h('b', {}, 'fig. 1'), ` “${request.prompt}”, seed ${request.seed}. ${formatSeconds(seconds)} on this device.`);
        fileName = `sd-turbo-${request.seed}.png`;
        save.disabled = false;
    };
    return { element, show };
}
