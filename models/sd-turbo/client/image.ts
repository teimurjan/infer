// Image helpers: the golden float image as bytes, and a byte compare.

// An image [3][512][512] in [-1, 1] as RGBA bytes, the same rule as image_to_rgba in Rust: (x / 2 + 0.5), clamp, times 255, round.
export function toRgba(values: Float32Array): Uint8Array {
    // Pixels in one plane.
    const plane = values.length / 3;
    // 4 bytes per pixel, alpha 255.
    const rgba = new Uint8Array(plane * 4);
    for (let pixel = 0; pixel < plane; pixel++) {
        for (let channel = 0; channel < 3; channel++) {
            rgba[pixel * 4 + channel] = Math.round(Math.min(Math.max(values[channel * plane + pixel] / 2 + 0.5, 0), 1) * 255);
        }
        rgba[pixel * 4 + 3] = 255;
    }
    return rgba;
}

export type ByteCompare = { differing: number; largest: number; total: number };

// How many bytes differ, and the largest difference.
export function compareBytes(actual: Uint8Array, expected: Uint8Array): ByteCompare {
    let differing = 0;
    let largest = 0;
    for (let index = 0; index < actual.length; index++) {
        const difference = Math.abs(actual[index] - expected[index]);
        if (difference > 0) differing++;
        largest = Math.max(largest, difference);
    }
    return { differing, largest, total: actual.length };
}
