// The dictionary entry of one inference. Each model exports one, and the site lists every entry it imports.
// Every number here comes from a recorded run, named in `measured.source`.

// One part of the pipeline: its name, what it does, and the weight file it streams.
// A part with no weight file has a note instead, shown where the size would be.
export type Part = { name: string; does: string; file: string | null; note?: string };

export type Entry = {
    // The id in addresses, for example "sd-turbo" in /sd-turbo/.
    id: string;
    // The letter of its thumb tab.
    letter: string;
    // The part of speech, for example "n. text → image".
    pos: string;
    // The one-line definition.
    definition: string;
    // Where the model comes from, in one sentence.
    etymology: string;
    parts: readonly Part[];
    // A measured run on a phone: the seconds of each part that runs, and where the numbers come from.
    measured: { device: string; browser: string; date: string; seconds: readonly { part: string; seconds: number }[]; source: string };
    // The input a visitor types, with an example.
    input: { label: string; example: string };
};
