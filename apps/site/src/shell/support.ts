// Checks that this browser can run the engines: a secure page, WebGPU, and a GPU adapter.

// ok, or the reason in plain words, for the page to show.
export type Support = { ok: true; adapter: string } | { ok: false; reason: string };

export async function checkSupport(): Promise<Support> {
    // Safari gives WebGPU only to HTTPS pages (and to localhost).
    if (!window.isSecureContext) return { ok: false, reason: 'This page must open over HTTPS. WebGPU works only on secure pages.' };
    if (!('gpu' in navigator)) return { ok: false, reason: 'This browser has no WebGPU. Use Safari 26 or later on iPhone, or a current Chrome.' };
    // Some browsers never answer when they have no GPU, so the check gives up after 5 s.
    const timeout = new Promise<null>(resolve => setTimeout(() => resolve(null), 5000));
    const adapter = await Promise.race([navigator.gpu.requestAdapter().catch(() => null), timeout]);
    if (!adapter) return { ok: false, reason: 'The browser has WebGPU, but it gave no GPU adapter.' };
    // The vendor and architecture, for example "apple" and "common-3". Some browsers leave them empty.
    const { vendor, architecture } = adapter.info;
    return { ok: true, adapter: [vendor, architecture].filter(Boolean).join(' ') || 'unknown GPU' };
}
