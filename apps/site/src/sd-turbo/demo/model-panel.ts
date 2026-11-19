// The contents of the entry: each weight file with its size and whether it is stored on this device,
// and the one tap that stores them. A download of 2 GB never starts by itself.

import type { Entry } from '@infer/browser/entry';
import { isCached } from '@infer/sd-turbo/client/cache';
import { downloadToCache } from '@infer/sd-turbo/client/download';
import { weightSize } from '@infer/sd-turbo/client/weights';
import { badge } from '@infer/ui/components/badge';
import { button } from '@infer/ui/components/button';
import { progress, type Progress } from '@infer/ui/components/progress';
import { h } from '@infer/ui/dom';
import { formatBytes, formatRate } from '@infer/ui/format';

export type ModelPanel = { element: HTMLElement; ready: Promise<void> };

// One stored part: its weight file, its size and its band.
type Row = { file: string; size: number | null; bar: Progress };

// Make the section. `ready` settles when every file is stored, or at once if the browser has no storage.
export function modelPanel(entry: Entry, folder: FileSystemDirectoryHandle | null): ModelPanel {
    const status = badge('checking', 'neutral');
    const summary = h('p', { class: 'ui-small' }, 'Looking for the weights on this device.');
    const list = h('div', { class: 'ui-stack', style: 'gap: var(--s-3)' });
    const action = h('div', { class: 'ui-row' });
    const element = h('section', { class: 'ui-section', 'aria-labelledby': 'contents-label' },
        h('div', { class: 'ui-row', style: 'justify-content: space-between' }, h('h2', { class: 'ui-section-label', id: 'contents-label' }, 'contents'), status.element),
        list, summary, action);

    const ready = (async () => {
        // Only parts with a weight file are stored. Their numbers keep the order of the pipeline.
        const stored = entry.parts.map((part, index) => ({ part, n: index + 1 })).filter(({ part }) => part.file !== null);
        const rows: Row[] = await Promise.all(stored.map(async ({ part, n }) => ({ file: part.file!, size: await weightSize(part.file!), bar: progress(`${n}  ${part.name}`) })));
        const cached = await Promise.all(rows.map(row => isCached(folder, row.file)));
        const total = rows.reduce((sum, row) => sum + (row.size ?? 0), 0);
        rows.forEach((row, index) => {
            row.bar.set(cached[index] ? 1 : 0, cached[index] ? `${formatBytes(row.size ?? 0)}, stored` : formatBytes(row.size ?? 0), cached[index] ? 'done' : 'running');
            list.append(row.bar.element);
        });
        // Without storage, every image downloads the files again. The page says so, and the entry still runs.
        if (!folder) {
            status.set('streams each time', 'warning');
            summary.textContent = `This browser cannot store files for a page, so each image downloads all ${formatBytes(total)} again.`;
            return;
        }
        if (cached.every(Boolean)) {
            status.set('stored', 'success');
            summary.textContent = `All ${formatBytes(total)} are stored on this device. An image starts at once.`;
            return;
        }
        const missing = rows.filter((_, index) => !cached[index]);
        const missingBytes = missing.reduce((sum, row) => sum + (row.size ?? 0), 0);
        status.set('not stored', 'neutral');
        summary.textContent = `The weights download once and stay in this browser. ${formatBytes(missingBytes)} to go. Use Wi-Fi.`;
        // Wait for the tap.
        await new Promise<void>(resolve => action.append(button(`store ${formatBytes(missingBytes)} on this device`, { kind: 'ink', large: true, onClick: resolve })));
        action.replaceChildren();
        status.set('storing', 'accent');
        // Ask the browser to keep the files when storage runs low. The browser can say no, and the download still works.
        await navigator.storage.persist?.().catch(() => false);
        for (const row of missing) {
            const started = performance.now();
            await downloadToCache(folder, row.file, ({ received, length }) => {
                const seconds = (performance.now() - started) / 1000;
                row.bar.set(length ? received / length : null, `${formatBytes(received)} of ${formatBytes(length ?? row.size ?? 0)} · ${formatRate(received / Math.max(seconds, 0.001))}`);
            });
            row.bar.set(1, `${formatBytes(row.size ?? 0)}, stored`, 'done');
        }
        status.set('stored', 'success');
        summary.textContent = `All ${formatBytes(total)} are stored on this device. An image starts at once.`;
    })();
    // A failed download names its reason, and a reload tries again.
    ready.catch((error: unknown) => {
        status.set('failed', 'danger');
        summary.textContent = `${error instanceof Error ? error.message : String(error)}. Reload the page to try again.`;
    });
    return { element, ready };
}
