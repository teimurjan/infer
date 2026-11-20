// Builds the site into dist/. Each HTML file in pages/ becomes one page, with its scripts and styles bundled.
// The server serves dist/ first, so pages/index.html is "/", and pages/sd-turbo/index.html is "/sd-turbo/".
// Run it with `bun build.ts` in this folder.

import { copyFile, mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';

// The font sheet of the design system, and the packages that hold its font files.
const UI = join(import.meta.dir, '..', '..', 'packages', 'ui');
const FONT_PACKAGES = join(UI, 'node_modules', '@fontsource-variable');

// Every page of the site.
const pages = [...new Bun.Glob('pages/**/*.html').scanSync('.')];
// The deployed site reads the SD-Turbo weights from Hugging Face, for example
// SD_TURBO_WEIGHTS_URL=https://huggingface.co/<user>/<repo>/resolve/<commit> bun run build.
const weightsUrl = process.env.SD_TURBO_WEIGHTS_URL;
// define writes the value into the bundle in place of the name. Without the variable, the pages read the weights from the dev server.
const define: Record<string, string> = weightsUrl ? { SD_TURBO_WEIGHTS_URL: JSON.stringify(weightsUrl.replace(/\/$/, '')) } : {};
// An old build can hold files of a deleted page, so it goes first.
await rm('dist', { recursive: true, force: true });
// The font files stay outside the bundle: the copy below puts them in dist/fonts/, so the stylesheet stays small.
const result = await Bun.build({ entrypoints: pages, outdir: 'dist', root: 'pages', minify: true, sourcemap: 'linked', external: ['/fonts/*'], define });
// Bun.build does not throw on a bad import, so stop here with its messages.
if (!result.success) {
    console.error(result.logs.join('\n'));
    process.exit(1);
}

// The font sheet names its files as /fonts/<file>. Copy exactly those files, from whichever package holds each one.
const sheet = await Bun.file(join(UI, 'src', 'fonts.css')).text();
const names = [...sheet.matchAll(/url\("\/fonts\/([^"]+)"\)/g)].map(match => match[1]!);
await mkdir('dist/fonts', { recursive: true });
for (const name of names) {
    // The packages are links into the Bun store, so the scan follows links.
    const [found] = [...new Bun.Glob(`*/files/${name}`).scanSync({ cwd: FONT_PACKAGES, followSymlinks: true })];
    if (!found) {
        console.error(`font file ${name} is named in fonts.css but is in no @fontsource-variable package`);
        process.exit(1);
    }
    await copyFile(join(FONT_PACKAGES, found), join('dist/fonts', name));
}
// Print where the pages will read the weights, so a deploy build shows the URL it holds.
console.log(`weights: ${weightsUrl ?? 'from the dev server'}`);
console.log(`built ${pages.length} pages, ${result.outputs.length} files, ${names.length} fonts`);
