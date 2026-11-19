import { errorText } from '@infer/browser/errors';

const WORKGROUP_SIZE = 64;
const WORKGROUPS = 256;
// Two 8-byte timestamps: one at the start of the pass, one at the end.
const TIMESTAMP_BYTES = 16;

// Each outer iteration runs a fixed inner loop, so a u32 dial reaches minutes on fast GPUs.
// The result is stored, so the compiler cannot drop the loop.
const SHADER = /* wgsl */ `
struct Params { iterations: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
    var state = id.x + 1u;
    for (var i = 0u; i < params.iterations; i++) {
        for (var j = 0u; j < 256u; j++) {
            state = state * 1664525u + 1013904223u;
            state ^= state >> 13u;
        }
    }
    output[id.x] = state;
}`;

// 'hang' means our own timeout ran out before the GPU answered.
export type RunOutcome = 'ok' | 'lost' | 'hang' | 'error';
export type RunResult = { outcome: RunOutcome; wallMs: number; gpuMs?: number; error?: string };

export type Gpu = {
    hasTimestamps: boolean;
    run: (iterations: number, timeoutMs: number) => Promise<RunResult>;
    destroy: () => void;
};

export type OpenResult = { ok: true; gpu: Gpu } | { ok: false; error: string };

type Timing = {
    passDescriptor: GPUComputePassDescriptor;
    resolve: (encoder: GPUCommandEncoder) => void;
    read: () => Promise<number | undefined>;
};

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
// Rounds to one decimal place.
const roundMs = (ms: number): number => Math.round(ms * 10) / 10;

// GPU-side timing of one pass, with timestamp queries.
function createTiming(device: GPUDevice): Timing {
    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    const resolveBuffer = device.createBuffer({ size: TIMESTAMP_BYTES, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    // The CPU can only read a buffer with MAP_READ, so we copy the timestamps into this one.
    const readBuffer = device.createBuffer({ size: TIMESTAMP_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    return {
        passDescriptor: { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } },
        resolve: encoder => {
            encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
            encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, TIMESTAMP_BYTES);
        },
        read: async () => {
            await readBuffer.mapAsync(GPUMapMode.READ);
            // slice(0) copies the bytes, because the mapped range is gone after unmap.
            const [begin, end] = new BigUint64Array(readBuffer.getMappedRange().slice(0));
            readBuffer.unmap();
            // Timestamps are in nanoseconds. Some GPUs give 0 or wrong values, so we skip those.
            return end > begin ? roundMs(Number(end - begin) / 1e6) : undefined;
        },
    };
}

/*
 * Opens a fresh device and returns run(iterations, timeoutMs), which submits one
 * dispatch as one command buffer and waits for it.
 */
export async function openGpu({ onUncapturedError }: { onUncapturedError: (message: string) => void }): Promise<OpenResult> {
    const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
    if (!adapter) {
        return { ok: false, error: 'no adapter (WebGPU missing or blocked for this site)' };
    }
    const hasTimestamps = adapter.features.has('timestamp-query');
    let device: GPUDevice;
    try {
        device = await adapter.requestDevice({ requiredFeatures: hasTimestamps ? ['timestamp-query'] : [] });
    } catch (error) {
        return { ok: false, error: errorText(error) };
    }
    device.addEventListener('uncapturederror', event => onUncapturedError(event.error.message));

    let lostInfo: GPUDeviceLostInfo | null = null;
    const lost = device.lost.then(info => { lostInfo = info; });

    const module = device.createShaderModule({ code: SHADER });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    // One u32: the iteration count for the shader.
    const paramsBuffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // One u32 per GPU thread.
    const outputBuffer = device.createBuffer({ size: WORKGROUPS * WORKGROUP_SIZE * 4, usage: GPUBufferUsage.STORAGE });
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
        ],
    });
    const timing = hasTimestamps ? createTiming(device) : null;

    // A 'lost' result if the device is lost by now, else null.
    const lostResult = (wallMs: number): RunResult | null =>
        lostInfo && { outcome: 'lost', wallMs, error: `${lostInfo.reason}: ${lostInfo.message}` };

    const submitAndWait = async (iterations: number, timeoutMs: number): Promise<RunResult> => {
        const lostBefore = lostResult(0);
        if (lostBefore) {
            return lostBefore;
        }
        device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([iterations]));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass(timing?.passDescriptor ?? {});
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(WORKGROUPS);
        pass.end();
        timing?.resolve(encoder);

        const start = performance.now();
        device.queue.submit([encoder.finish()]);
        // The first of three wins: the GPU finished, the device was lost, or our timeout ran out.
        const settled = await Promise.race([
            device.queue.onSubmittedWorkDone().then(() => 'ok' as const),
            lost.then(() => 'lost' as const),
            sleep(timeoutMs).then(() => 'hang' as const),
        ]);
        const wallMs = roundMs(performance.now() - start);
        const lostAfter = lostResult(wallMs);
        if (lostAfter) {
            return lostAfter;
        }
        if (settled !== 'ok') {
            return { outcome: settled, wallMs };
        }
        return { outcome: 'ok', wallMs, gpuMs: timing ? await timing.read() : undefined };
    };

    // A loss during the timestamp readback rejects mapAsync, so exceptions become outcomes too.
    const run = (iterations: number, timeoutMs: number): Promise<RunResult> => submitAndWait(iterations, timeoutMs)
        .catch(error => lostResult(0) ?? { outcome: 'error', wallMs: 0, error: errorText(error) });

    return { ok: true, gpu: { hasTimestamps, run, destroy: () => device.destroy() } };
}
