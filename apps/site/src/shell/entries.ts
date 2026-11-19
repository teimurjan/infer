// Every inference that runs on the phone today. A new model adds its entry here, in alphabetical order.

import type { Entry } from '@infer/browser/entry';
import { entry as sdTurbo } from '@infer/sd-turbo/client/entry';

export const entries: readonly Entry[] = [sdTurbo];
