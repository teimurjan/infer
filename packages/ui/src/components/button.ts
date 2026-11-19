// A button of the design system.

import { h } from '../dom.ts';

// primary: an action that starts a run, in the live blue. ink: a main action that runs nothing, in black.
// secondary: other actions, outlined. ghost: small actions, as an underlined word.
export type ButtonKind = 'primary' | 'ink' | 'secondary' | 'ghost';

export type ButtonOptions = { kind?: ButtonKind; large?: boolean; disabled?: boolean; onClick?: () => void };

// Make a button with a label.
export function button(label: string, { kind = 'secondary', large = false, disabled = false, onClick }: ButtonOptions = {}): HTMLButtonElement {
    const className = ['ui-button', `ui-button-${kind}`, large ? 'ui-button-large' : ''].join(' ').trim();
    return h('button', { type: 'button', class: className, disabled, onclick: onClick && (() => onClick()) }, label);
}
