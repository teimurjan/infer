// Deploys the site to Vercel, the production site of infer. The SD-Turbo weights stay on Hugging Face.
// Run it with `bun run deploy` in the repository folder. It needs `bunx vercel login` once.

import { copyFile, cp, mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';

// The weight files of the demo, at the commit of the upload of python/upload_weights.py.
// The commit, not "main", so a later upload cannot change the weights under a deployed engine.
const WEIGHTS_URL = 'https://huggingface.co/teimurjan/infer-sd-turbo/resolve/ad104bd1d8a5dea36cdec628db65e0deb6cf13b0';
// The Vercel project. The first run creates it.
const PROJECT = 'infer';
// The Vercel team of the project. The account has more than one team, and teimurjan.dev is in this one.
const SCOPE = 'teimurjan';
// The folder that Vercel uploads. It is made again on each run, and git ignores it.
const OUT = join(import.meta.dir, 'deploy');
// The WASM build of the engine. The pages fetch it at /models/sd-turbo/engine/pkg/engine_bg.wasm, from client/urls.ts.
const ENGINE_PKG = join(import.meta.dir, '..', '..', 'models', 'sd-turbo', 'engine', 'pkg');
const ENGINE_URL_FOLDER = join(OUT, 'models', 'sd-turbo', 'engine', 'pkg');

// Run one command in this folder, with the output on the screen. Stop the script if the command fails.
async function run(command: string[], env: Record<string, string> = {}): Promise<void> {
    const child = Bun.spawn(command, { cwd: import.meta.dir, env: { ...process.env, ...env }, stdout: 'inherit', stderr: 'inherit' });
    const code = await child.exited;
    if (code !== 0) {
        console.error(`failed with exit code ${code}: ${command.join(' ')}`);
        process.exit(code);
    }
}

// Build the engine again, so the deployed WASM matches the Rust code of this commit.
await run(['bun', 'run', '--filter', '@infer/sd-turbo', 'build:wasm']);
// Build the pages with the Hugging Face address of the weights.
await run(['bun', 'build.ts'], { SD_TURBO_WEIGHTS_URL: WEIGHTS_URL });

// Copy the pages, then put the engine at the same path as on the dev server.
await rm(OUT, { recursive: true, force: true });
await cp(join(import.meta.dir, 'dist'), OUT, { recursive: true });
await mkdir(ENGINE_URL_FOLDER, { recursive: true });
await copyFile(join(ENGINE_PKG, 'engine_bg.wasm'), join(ENGINE_URL_FOLDER, 'engine_bg.wasm'));

// Build dist/ again without the variable, so the dev server reads the local weights again.
await run(['bun', 'build.ts']);

// Upload the folder as a production deployment. --yes answers the questions of the CLI, and --create-if-missing makes the project on the first run.
await run(['bunx', 'vercel', 'deploy', OUT, '--prod', '--yes', '--project', PROJECT, '--create-if-missing', '--scope', SCOPE]);
