// Counts the bytes of every WebGPU buffer that this page makes, and the peak of the live bytes.
// It wraps GPUDevice.createBuffer and GPUBuffer.destroy, so it counts the engine and onnxruntime-web the same way.
// A buffer stops counting when the page calls destroy() on it, or when the garbage collector frees it.
// The browser frees the GPU memory of a buffer at one of these two moments, so this is what the page really holds.

// One counted buffer. The size goes to 0 once it no longer counts, so it is never taken off twice.
type Entry = { size: number };

// The entry of each live buffer. A WeakMap does not keep a buffer alive.
const entries = new WeakMap<GPUBuffer, Entry>();
// Bytes of the buffers that count now.
let live = 0;
// The most bytes at one time since the page loaded.
let peak = 0;

// Stop counting one buffer.
function release(entry: Entry): void {
    live -= entry.size;
    entry.size = 0;
}

// Calls release when the garbage collector frees a buffer that was never destroyed.
const collected = new FinalizationRegistry<Entry>(release);

// Wrap the two WebGPU calls. Call this before any GPU device exists, so no buffer is missed.
export function installBufferCounter(): void {
    // A browser without WebGPU has nothing to count.
    if (!('GPUDevice' in globalThis)) {
        return;
    }
    // The real functions.
    const create = GPUDevice.prototype.createBuffer;
    const destroy = GPUBuffer.prototype.destroy;
    // Count each new buffer.
    GPUDevice.prototype.createBuffer = function (descriptor: GPUBufferDescriptor): GPUBuffer {
        const buffer = create.call(this, descriptor);
        const entry = { size: descriptor.size };
        entries.set(buffer, entry);
        // The entry is also the token that unregisters it.
        collected.register(buffer, entry, entry);
        live += entry.size;
        peak = Math.max(peak, live);
        return buffer;
    };
    // Stop counting a destroyed buffer.
    GPUBuffer.prototype.destroy = function (): undefined {
        const entry = entries.get(this);
        if (entry) {
            collected.unregister(entry);
            release(entry);
            entries.delete(this);
        }
        return destroy.call(this);
    };
}

// The peak since the page loaded, in bytes.
export const peakBytes = (): number => peak;
