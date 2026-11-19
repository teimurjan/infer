// One engine for all tests of a page. The GPU opens once, at the first test that needs it.

import { openSession, type Session } from '../engine.ts';

let opened: Promise<Session> | null = null;

// The session of this page. Later calls get the same one.
export const session = (): Promise<Session> => (opened ??= openSession());
