// The image checks that both runtimes share. The float-to-bytes rule is in client/image.ts.

// The side of the image in pixels.
export const SIDE = 512;

// Stop with an error if the image is not a real image: the wrong size, or one flat color (for example all zeros from a lost GPU).
export function checkImage(rgba: Uint8Array): Uint8Array {
    if (rgba.length !== SIDE * SIDE * 4) {
        throw new Error(`the image has ${rgba.length} bytes, expected ${SIDE * SIDE * 4}`);
    }
    // Compare every color byte with the first one. Alpha is always 255, so it is skipped.
    const flat = rgba.every((byte, index) => index % 4 === 3 || byte === rgba[0]);
    if (flat) {
        throw new Error(`the image is one flat color (${rgba[0]})`);
    }
    return rgba;
}
