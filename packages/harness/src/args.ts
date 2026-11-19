// Reads the command line of the harness.
//
//   bun run phone <page> [--mac] [--timeout <minutes>]
//
// <page> is a path of the site, for example "/sd-turbo/dev/?test=golden-image",
// or a short name "<model>:<test>", for example "sd-turbo:golden-image" or "sd-turbo:all".
// "<model>:benchmark" opens the benchmark page of the model, and takes the extra query after it, for example
// "sd-turbo:benchmark?runs=10&warm=3&runtimes=engine".

export type Args = { path: string; mac: boolean; timeoutMs: number };

// The time limit of a run if the command line names none: 30 minutes, enough for "Make 20" on the phone.
const DEFAULT_TIMEOUT_MINUTES = 30;

// The page path of a short name, or the argument itself if it is already a path.
function pagePath(page: string): string {
    if (page.startsWith('/')) return page;
    const [model, rest = ''] = page.split(':');
    const [name, query = ''] = rest.split('?');
    if (name === 'benchmark') return `/${model}/benchmark/?${query}`;
    return `/${model}/dev/?test=${name}`;
}

// The arguments, or an error with the usage line.
export function parseArgs(argv: readonly string[]): Args | Error {
    const usage = 'usage: bun run phone <page> [--mac] [--timeout <minutes>], for example: bun run phone sd-turbo:golden-image';
    const page = argv.find(arg => !arg.startsWith('--') && argv[argv.indexOf(arg) - 1] !== '--timeout');
    if (!page) return new Error(usage);
    const timeoutIndex = argv.indexOf('--timeout');
    const minutes = timeoutIndex === -1 ? DEFAULT_TIMEOUT_MINUTES : Number(argv[timeoutIndex + 1]);
    if (!Number.isFinite(minutes) || minutes <= 0) return new Error(`--timeout needs a number of minutes. ${usage}`);
    return { path: pagePath(page), mac: argv.includes('--mac'), timeoutMs: minutes * 60_000 };
}

// A run id that is safe as a file name: the time, the page, and 4 random letters.
export function runId(path: string): string {
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '');
    const page = path.replace(/[^A-Za-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40);
    return `${stamp}-${page}-${Math.random().toString(36).slice(2, 6)}`;
}
