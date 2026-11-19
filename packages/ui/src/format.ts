// Number formats that every page uses, so the same value looks the same everywhere.

// Bytes in one MB, as in the progress logs (10^6).
const MB = 1e6;

// Bytes as MB or GB, for example "866 MB" or "2.24 GB".
export const formatBytes = (bytes: number): string => (bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${(bytes / MB).toFixed(0)} MB`);

// Seconds with two digits, for example "1.79 s". Under one second, milliseconds.
export const formatSeconds = (seconds: number): string => (seconds < 1 ? `${(seconds * 1000).toFixed(0)} ms` : `${seconds.toFixed(2)} s`);

// A speed in MB per second.
export const formatRate = (bytesPerSecond: number): string => `${(bytesPerSecond / MB).toFixed(0)} MB/s`;
