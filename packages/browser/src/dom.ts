// Finds an element by id and checks its kind, for example HTMLCanvasElement.
// A missing or wrong element is a bug in the page, so we stop at once with a clear message.
export function byId<T extends HTMLElement>(id: string, kind: new () => T): T {
    const found = document.getElementById(id);
    if (!(found instanceof kind)) {
        throw new Error(`#${id} is missing or is not a ${kind.name}`);
    }
    return found;
}
