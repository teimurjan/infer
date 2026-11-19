// A small WebDriver client: only the commands that the harness needs. safaridriver speaks this protocol over HTTP.
// Each function returns its value or an Error, and never throws.

// A WebDriver answer is { value: ... }. An error answer has value.error and value.message.
type Answer = { value: unknown };

// Send one command to the driver at `base`, for example http://127.0.0.1:4444.
async function command(base: string, method: 'GET' | 'POST' | 'DELETE', path: string, body?: unknown): Promise<unknown> {
    const response = await fetch(`${base}${path}`, { method, headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
        .catch((error: unknown) => (error instanceof Error ? error : new Error(String(error))));
    if (response instanceof Error) return new Error(`safaridriver did not answer: ${response.message}`);
    const answer = (await response.json().catch(() => ({ value: null }))) as Answer;
    const value = answer.value as { error?: string; message?: string } | null;
    if (!response.ok || (value && typeof value === 'object' && 'error' in value)) return new Error(value?.message || value?.error || `HTTP ${response.status}`);
    return answer.value;
}

// Open a session with these capabilities. Returns the session id.
export async function createSession(base: string, capabilities: Record<string, unknown>): Promise<string | Error> {
    const value = await command(base, 'POST', '/session', { capabilities: { alwaysMatch: capabilities } });
    if (value instanceof Error) return value;
    const id = (value as { sessionId?: string }).sessionId;
    return id ?? new Error('safaridriver gave no session id');
}

// Open a page in the session's window.
export async function navigate(base: string, session: string, url: string): Promise<null | Error> {
    const value = await command(base, 'POST', `/session/${session}/url`, { url });
    return value instanceof Error ? value : null;
}

// The address of the page now open. An error means the session or its window is gone.
export async function currentUrl(base: string, session: string): Promise<string | Error> {
    const value = await command(base, 'GET', `/session/${session}/url`);
    return value instanceof Error ? value : String(value);
}

// Close the session and its window.
export async function deleteSession(base: string, session: string): Promise<void> {
    await command(base, 'DELETE', `/session/${session}`);
}
