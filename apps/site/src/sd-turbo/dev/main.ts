// The developer page of SD-Turbo: its test suite, plus a button that deletes the stored weights.

import { clearCache } from '@infer/sd-turbo/client/cache';
import { suite } from '@infer/sd-turbo/client/tests';
import { button } from '@infer/ui/components/button';
import { mountSuite } from '../../dev/runner.ts';

// Deleting the cache makes the next weight load come from the network, to measure a first visit.
const clear = button('clear the weight cache', { kind: 'secondary' });
clear.addEventListener('click', async () => {
    clear.disabled = true;
    await clearCache();
    clear.textContent = 'cache cleared';
});

await mountSuite(suite, [clear]);
