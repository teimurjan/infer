// Turns anything that was thrown into readable text.
// GPU errors and lost-device info are not Error objects, but they have a message too.
export function errorText(error: unknown): string {
    const hasMessage = typeof error === 'object' && error !== null && 'message' in error;
    return String(hasMessage ? error.message : error);
}
