// Start the development server from the command line.
// `bun run serve`: http://127.0.0.1:8765, for the Mac only.
// `bun run serve:lan`: https on port 8443, for the iPhone on the same Wi-Fi.

import { RUNS, ROOT, startServer } from './server.ts';

const running = await startServer({ lan: process.argv.includes('--lan') });
// Stop at once with the reason, for example a missing certificate.
if (running instanceof Error) {
    console.error(running.message);
    process.exit(1);
}
console.log(`serving ${ROOT} at ${running.origin}/`);
console.log(`run events go to ${RUNS}/`);
// Print each run line in the terminal too, so a run by hand on the phone shows up on the Mac.
running.listen(event => {
    if (event.kind === 'line') console.log(`[${event.page}] ${event.text}`);
    if (event.kind === 'done') console.log(`[${event.page}] ${event.passed ? 'PASSED' : 'FAILED'}: ${event.summary}`);
});
