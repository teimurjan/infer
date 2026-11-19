// Probe state that survives a reload or a dead tab, kept in localStorage under one key.
export type Store<State> = {
    load: () => State;
    save: (state: State) => void;
};

export function createStore<State extends object>(key: string, empty: State): Store<State> {
    const load = (): State => {
        try {
            // Only this probe writes the key, so we trust the saved shape.
            // Saved fields win over the empty defaults. A missing key parses to null and keeps the defaults.
            return { ...empty, ...JSON.parse(localStorage.getItem(key) ?? 'null') };
        } catch {
            return empty;
        }
    };
    const save = (state: State): void => localStorage.setItem(key, JSON.stringify(state));
    return { load, save };
}

// A crash never fires pagehide, so this separates "the user left" from "the tab died".
export function markActiveAsLeftOnPageHide<State extends { active: object | null }>(store: Store<State>): void {
    window.addEventListener('pagehide', () => {
        const state = store.load();
        if (state.active) {
            store.save({ ...state, active: { ...state.active, left: true } });
        }
    });
}
