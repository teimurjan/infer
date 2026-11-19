// Prints the events of a run in the terminal, one line each.

import type { RunEvent } from '@infer/browser/run-event';

// The text of one event, or null for an event that prints nothing.
export function eventText(event: RunEvent, loads: number): string | null {
    // seconds since the page loaded, so slow steps stand out.
    const time = `${(event.at / 1000).toFixed(1).padStart(6)} s`;
    if (event.kind === 'start') return `${time}  page loaded${loads > 1 ? ` again (load ${loads}): the page reloaded` : ''}`;
    if (event.kind === 'line') return `${time}  ${event.text}`;
    if (event.kind === 'metric') return `${time}  metric ${event.name} = ${event.value} ${event.unit}`.trimEnd();
    return `${time}  ${event.passed ? 'PASSED' : 'FAILED'}: ${event.summary}`;
}
