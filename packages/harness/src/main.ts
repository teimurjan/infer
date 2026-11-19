// The harness: runs one page in Safari on the connected iPhone, prints what the page reports, and saves it on the Mac.
// `bun run phone sd-turbo:golden-image`. The exit code is 0 if the page passed, and 1 if not.

import { parseArgs } from './args.ts';
import { runPage } from './run.ts';

const args = parseArgs(process.argv.slice(2));
if (args instanceof Error) {
    console.error(args.message);
    process.exit(2);
}
const outcome = await runPage(args);
console.log(`${outcome.passed ? 'PASSED' : 'FAILED'}: ${outcome.summary}`);
if (outcome.file) console.log(`events: ${outcome.file}`);
process.exit(outcome.passed ? 0 : 1);
