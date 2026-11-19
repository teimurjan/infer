// Finds the iPhone that is connected to the Mac, with Xcode's devicectl.

import { tmpdir } from 'node:os';
import { join } from 'node:path';

export type Device = { udid: string; name: string; osVersion: string };

// The part of the devicectl JSON that this file reads.
type DeviceList = { result: { devices: { hardwareProperties?: { udid?: string; reality?: string; platform?: string }; deviceProperties?: { name?: string; osVersionNumber?: string }; connectionProperties?: { tunnelState?: string } }[] } };

// The first real (not simulated) iOS device whose connection is up, or an error that says what to fix.
export async function findIphone(): Promise<Device | Error> {
    // devicectl writes its JSON to a file, not to stdout.
    const file = join(tmpdir(), `devicectl-${process.pid}.json`);
    const listing = Bun.spawn(['xcrun', 'devicectl', 'list', 'devices', '--json-output', file], { stdout: 'ignore', stderr: 'pipe' });
    if ((await listing.exited) !== 0) return new Error(`devicectl failed: ${await new Response(listing.stderr).text()}`);
    const list = (await Bun.file(file).json()) as DeviceList;
    await Bun.file(file).delete();
    const phone = list.result.devices.find(device =>
        device.hardwareProperties?.reality === 'physical' && device.hardwareProperties.platform === 'iOS' && device.connectionProperties?.tunnelState === 'connected');
    if (!phone?.hardwareProperties?.udid) return new Error('no iPhone is connected: plug it in with a cable, unlock it, and tap Trust');
    return { udid: phone.hardwareProperties.udid, name: phone.deviceProperties?.name ?? 'iPhone', osVersion: phone.deviceProperties?.osVersionNumber ?? '?' };
}
