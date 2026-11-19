// The HTTPS certificate for the iPhone. Safari gives WebGPU and OPFS only to secure pages.
// mkcert makes these two files once. The README of SD-Turbo has the command.

import { X509Certificate } from 'node:crypto';
import { homedir, networkInterfaces } from 'node:os';
import { join } from 'node:path';

// The folder of the certificate, outside the repository, so the private key is never served or committed.
export const FOLDER = join(homedir(), '.config', 'inference-lan');

export type Tls = { cert: string; key: string; host: string };

// The IPv4 addresses of the Mac on its networks, without 127.0.0.1.
export const macAddresses = (): string[] =>
    Object.values(networkInterfaces()).flat().filter(entry => entry?.family === 'IPv4' && !entry.internal).map(entry => entry!.address);

// The certificate, the key, and the one Mac address that the certificate names. The phone must open that address,
// or Safari rejects the certificate. An error names the missing file or the address mismatch.
export async function loadTls(): Promise<Tls | Error> {
    const certFile = Bun.file(join(FOLDER, 'lan-cert.pem'));
    const keyFile = Bun.file(join(FOLDER, 'lan-key.pem'));
    for (const file of [certFile, keyFile]) {
        if (!(await file.exists())) return new Error(`missing ${file.name}: make it with: bun run cert`);
    }
    const cert = await certFile.text();
    // subjectAltName looks like "DNS:localhost, IP Address:10.120.1.125, IP Address:127.0.0.1".
    const named = (new X509Certificate(cert).subjectAltName ?? '').split(', ').filter(part => part.startsWith('IP Address:')).map(part => part.slice('IP Address:'.length));
    const host = macAddresses().find(address => named.includes(address));
    if (!host) return new Error(`the certificate names ${named.join(', ')}, but the Mac has ${macAddresses().join(', ')}: make a new certificate with: bun run cert`);
    return { cert, key: await keyFile.text(), host };
}
