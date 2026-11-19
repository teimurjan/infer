// Fetch helpers that throw with the URL and the HTTP status, so a missing file names itself.

// Fetch a URL, or throw if the status is not 2xx.
export async function fetchOk(url: string, init?: RequestInit): Promise<Response> {
    const response = await fetch(url, init);
    if (!response.ok) throw new Error(`cannot fetch ${url}: HTTP ${response.status}`);
    return response;
}

// The whole body as bytes.
export const fetchBytes = async (url: string): Promise<ArrayBuffer> => (await fetchOk(url)).arrayBuffer();

// The whole body as text.
export const fetchText = async (url: string): Promise<string> => (await fetchOk(url)).text();

// The whole body as float32 values. The test files are raw little-endian float32.
export const fetchFloats = async (url: string): Promise<Float32Array> => new Float32Array(await fetchBytes(url));
