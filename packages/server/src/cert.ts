// Makes the HTTPS certificate of the iPhone mode with mkcert, for the Wi-Fi address that the Mac has now.
// `bun run cert`. Run it again when the Wi-Fi address of the Mac changes.
// The iPhone trusts the certificate if it trusts the mkcert root (see models/sd-turbo/README.md), so nothing changes on the phone.

import { mkdir } from 'node:fs/promises';
import { networkInterfaces } from 'node:os';
import { join } from 'node:path';
import { FOLDER } from './tls.ts';

// en0 is the Wi-Fi of a MacBook. The phone reaches the Mac on that address.
const wifi = networkInterfaces().en0?.find(entry => entry.family === 'IPv4')?.address;
if (!wifi) {
    console.error('the Mac has no IPv4 address on en0 (Wi-Fi): join the same Wi-Fi as the iPhone first');
    process.exit(1);
}
await mkdir(FOLDER, { recursive: true });
const mkcert = Bun.spawn(['mkcert', '-cert-file', join(FOLDER, 'lan-cert.pem'), '-key-file', join(FOLDER, 'lan-key.pem'), wifi, 'localhost', '127.0.0.1'], { stdout: 'inherit', stderr: 'inherit' });
process.exit(await mkcert.exited);
