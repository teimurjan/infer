// A log view: monospace lines, newest at the bottom. Screen readers announce new lines.

import { h } from '../dom.ts';

// danger: an error. success: a pass. muted: a detail.
export type LogTone = 'normal' | 'danger' | 'success' | 'muted';

export type Log = { element: HTMLPreElement; add(line: string, tone?: LogTone): void; clear(): void; text(): string };

// Make an empty log.
export function log(): Log {
    const element = h('pre', { class: 'ui-log', 'aria-live': 'polite' });
    const add = (line: string, tone: LogTone = 'normal'): void => {
        element.append(h('div', { class: 'ui-log-line', 'data-tone': tone }, line));
        // Scroll to the new line.
        element.scrollTop = element.scrollHeight;
    };
    const clear = (): void => element.replaceChildren();
    // All lines as one text, for a copy button.
    const text = (): string => [...element.children].map(line => line.textContent).join('\n');
    return { element, add, clear, text };
}
