// The POST /crumb route. Probe pages send one breadcrumb per request with navigator.sendBeacon.
// Each one goes to a JSONL file (one JSON object per line) and to the terminal.

import { appendFile } from 'node:fs/promises';

// Add the server clock time, so a kill can be matched to the phone's log and crash reports.
function stampReceipt(body: string): string {
    try {
        return JSON.stringify({ at: new Date().toISOString(), ...JSON.parse(body) });
    } catch {
        // The body is not JSON, so we store it as it came.
        return body;
    }
}

// Make the route handler that appends to the file at `logPath`.
export const crumbSink = (logPath: string) => async (request: Request): Promise<Response> => {
    const line = stampReceipt(await request.text());
    await appendFile(logPath, `${line}\n`);
    console.log(`[crumb] ${line}`);
    // 204 means "received, nothing to send back".
    return new Response(null, { status: 204 });
};
