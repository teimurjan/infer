// Builds the onnxruntime-web baseline page into dist/baseline.js, and copies the onnxruntime-web .wasm files next to it.
// Run it with `bun build.ts` in this folder.

import { cp } from 'node:fs/promises';
import { dirname, join } from 'node:path';

// The page script lands in dist/ under its folder name: dist/baseline.js.
const result = await Bun.build({
    entrypoints: ['baseline/index.ts'],
    outdir: 'dist',
    root: '.',
    naming: '[dir].[ext]',
    sourcemap: 'linked',
});
// Bun.build does not throw on a bad import, so stop here with its messages.
if (!result.success) {
    console.error(result.logs.join('\n'));
    process.exit(1);
}

// onnxruntime-web loads its .wasm and .mjs files at run time from ort.env.wasm.wasmPaths, which is dist/.
// The resolved entry file sits in the dist folder of the package, next to those files.
const ortDist = dirname(Bun.resolveSync('onnxruntime-web', import.meta.dir));
await cp(ortDist, 'dist', { recursive: true, filter: source => source === ortDist || /ort-wasm-.*\.(wasm|mjs)$/.test(source) });

console.log(`built ${result.outputs.map(output => output.path.replace(`${process.cwd()}/`, '')).join(', ')}`);
