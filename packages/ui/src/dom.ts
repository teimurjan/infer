// A small typed builder for DOM elements, so pages do not write long createElement code.

// The attributes of an element: strings become attributes, functions on "on..." become event listeners.
type Props = Record<string, string | number | boolean | undefined | ((event: Event) => void)>;
// A child is an element, a text, or nothing.
type Child = Node | string | null | undefined | false;

// Make an element of `tag` with its props and children. For example h('p', { class: 'ui-muted' }, 'Hello').
export function h<Tag extends keyof HTMLElementTagNameMap>(tag: Tag, props: Props = {}, ...children: Child[]): HTMLElementTagNameMap[Tag] {
    const element = document.createElement(tag);
    for (const [name, value] of Object.entries(props)) {
        // undefined and false leave the attribute out.
        if (value === undefined || value === false) continue;
        if (typeof value === 'function') element.addEventListener(name.slice(2), value);
        // true makes an attribute with no value, for example disabled.
        else element.setAttribute(name, value === true ? '' : String(value));
    }
    element.append(...children.filter((child): child is Node | string => Boolean(child)));
    return element;
}

// Find an element by id with its type, or throw: a missing element is a bug in the page, found at load time.
export function byId<Type extends HTMLElement>(id: string, type: new () => Type): Type {
    const element = document.getElementById(id);
    if (!(element instanceof type)) throw new Error(`#${id} is not a ${type.name}`);
    return element;
}
