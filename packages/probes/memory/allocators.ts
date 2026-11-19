import { errorText } from '@infer/browser/errors';

// WebAssembly memory grows in pages of 64 KB.
const WASM_PAGE_BYTES = 64 * 1024;
// We upload GPU data in 4 MB pieces, so one upload never needs a huge temporary copy.
const GPU_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024;

export type AllocatorKind = 'wasm' | 'js' | 'gpu';
export type RampName = AllocatorKind | 'mixed';

// Why an allocation step failed.
export type FailureOutcome = 'refused' | 'no-webgpu' | 'device-lost' | 'gpu-oom' | 'gpu-validation' | 'gpu-error';

export type Failure = { ok: false; outcome: FailureOutcome; error: string; kind?: AllocatorKind };
export type AllocationResult = { ok: true } | Failure;

// Something that can take `bytes` more memory of one kind and hold it until the page dies.
export type Allocator = { ok: true; allocate: (bytes: number) => Promise<AllocationResult> };

type Ramp = {
    create: () => Promise<Allocator | Failure>;
    // How many kinds of memory one step takes. The mixed ramp takes all three.
    kindsPerStep: number;
};

const OK = { ok: true } as const;

const failure = (outcome: FailureOutcome, error: unknown): Failure => ({ ok: false, outcome, error: errorText(error) });

// Random words, so OS memory compression cannot hide the pages we touch.
function fillRandom(words: Uint32Array): void {
    // xorshift32: a fast pseudo-random generator. The start value must not be 0.
    let state = (Math.random() * 2 ** 32) >>> 0 || 1;
    for (let i = 0; i < words.length; i++) {
        state ^= state << 13;
        state ^= state >>> 17;
        state ^= state << 5;
        words[i] = state;
    }
}

async function wasmAllocator(): Promise<Allocator> {
    const memory = new WebAssembly.Memory({ initial: 0 });
    const allocate = async (bytes: number): Promise<AllocationResult> => {
        // The new pages start where the old memory ended.
        const offset = memory.buffer.byteLength;
        try {
            memory.grow(bytes / WASM_PAGE_BYTES);
        } catch (error) {
            return failure('refused', error);
        }
        fillRandom(new Uint32Array(memory.buffer, offset, bytes / 4));
        return OK;
    };
    return { ok: true, allocate };
}

async function jsAllocator(): Promise<Allocator> {
    // Keeping a reference stops the garbage collector from freeing the buffers.
    const held: ArrayBuffer[] = [];
    const allocate = async (bytes: number): Promise<AllocationResult> => {
        let buffer: ArrayBuffer;
        try {
            buffer = new ArrayBuffer(bytes);
        } catch (error) {
            return failure('refused', error);
        }
        fillRandom(new Uint32Array(buffer));
        held.push(buffer);
        return OK;
    };
    return { ok: true, allocate };
}

async function requestDevice(): Promise<GPUDevice | null> {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
        return null;
    }
    // Ask for the largest buffer the adapter allows. The default limit is smaller.
    return adapter.requestDevice({ requiredLimits: { maxBufferSize: adapter.limits.maxBufferSize } });
}

async function gpuAllocator(): Promise<Allocator | Failure> {
    const device = await requestDevice().catch(() => null);
    if (!device) {
        return failure('no-webgpu', 'no adapter or device');
    }
    let lostInfo: GPUDeviceLostInfo | null = null;
    const lost = device.lost.then(info => { lostInfo = info; });
    // A failure if the device is lost by now, else null.
    const lostFailure = (): Failure | null => lostInfo && failure('device-lost', `${lostInfo.reason}: ${lostInfo.message}`);

    const uploadChunk = new Uint32Array(GPU_UPLOAD_CHUNK_BYTES / 4);
    fillRandom(uploadChunk);
    const held: GPUBuffer[] = [];

    const allocate = async (bytes: number): Promise<AllocationResult> => {
        const lostBefore = lostFailure();
        if (lostBefore) {
            return lostBefore;
        }
        try {
            // Error scopes catch errors from the calls below. They pop in reverse order.
            device.pushErrorScope('out-of-memory');
            device.pushErrorScope('validation');
            const buffer = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            // Writing real data makes the driver back the buffer with real memory.
            for (let offset = 0; offset < bytes; offset += GPU_UPLOAD_CHUNK_BYTES) {
                device.queue.writeBuffer(buffer, offset, uploadChunk, 0, Math.min(GPU_UPLOAD_CHUNK_BYTES, bytes - offset) / 4);
            }
            // Wait until the GPU finished the uploads, or the device is lost.
            await Promise.race([device.queue.onSubmittedWorkDone(), lost]);
            const validationError = await device.popErrorScope();
            const outOfMemoryError = await device.popErrorScope();
            const lostAfter = lostFailure();
            if (lostAfter) {
                return lostAfter;
            }
            if (outOfMemoryError) {
                return failure('gpu-oom', outOfMemoryError.message);
            }
            if (validationError) {
                return failure('gpu-validation', validationError.message);
            }
            held.push(buffer);
            return OK;
        } catch (error) {
            return lostFailure() ?? failure('gpu-error', error);
        }
    };
    return { ok: true, allocate };
}

// One step of each kind per mixed step; the first failure ends the ramp.
async function mixedAllocator(): Promise<Allocator | Failure> {
    const parts = { wasm: await wasmAllocator(), js: await jsAllocator(), gpu: await gpuAllocator() };
    if (!parts.gpu.ok) {
        return parts.gpu;
    }
    const ready: [AllocatorKind, Allocator][] = [['wasm', parts.wasm], ['js', parts.js], ['gpu', parts.gpu]];
    const allocate = async (bytes: number): Promise<AllocationResult> => {
        for (const [kind, part] of ready) {
            const result = await part.allocate(bytes);
            if (!result.ok) {
                return { ...result, kind };
            }
        }
        return OK;
    };
    return { ok: true, allocate };
}

export const RAMPS: Record<RampName, Ramp> = {
    wasm: { create: wasmAllocator, kindsPerStep: 1 },
    js: { create: jsAllocator, kindsPerStep: 1 },
    gpu: { create: gpuAllocator, kindsPerStep: 1 },
    mixed: { create: mixedAllocator, kindsPerStep: 3 },
};

export const isRampName = (name: string): name is RampName => Object.hasOwn(RAMPS, name);
