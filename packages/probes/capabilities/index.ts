import { crumb } from '@infer/browser/breadcrumbs';
import { byId } from '@infer/browser/dom';
import { errorText } from '@infer/browser/errors';

const HIGHLIGHTED_FEATURES = ['shader-f16', 'subgroups', 'timestamp-query'];

// WebIDL attributes live on the prototype, so copy them with for...in.
function toPlainObject(source: object): Record<string, unknown> {
    const copy: Record<string, unknown> = {};
    for (const key in source) {
        // Reflect.get reads the attribute through its getter on the prototype.
        const value: unknown = Reflect.get(source, key);
        if (typeof value !== 'function') {
            copy[key] = value;
        }
    }
    return copy;
}

async function collectReport(): Promise<Record<string, unknown>> {
    const base = { userAgent: navigator.userAgent, secureContext: isSecureContext };
    // The types say navigator.gpu always exists. Old browsers do not have it.
    if (!navigator.gpu) {
        return { ...base, webgpu: false, reason: 'navigator.gpu is missing' };
    }
    const wgslLanguageFeatures = [...(navigator.gpu.wgslLanguageFeatures ?? [])].sort();
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        return { ...base, webgpu: false, reason: 'requestAdapter returned null', wgslLanguageFeatures };
    }
    return {
        ...base,
        webgpu: true,
        adapterInfo: toPlainObject(adapter.info ?? {}),
        features: Object.fromEntries(HIGHLIGHTED_FEATURES.map(name => [name, adapter.features.has(name)])),
        allFeatures: [...adapter.features].sort(),
        limits: toPlainObject(adapter.limits),
        wgslLanguageFeatures,
    };
}

const report = await collectReport().catch(error => ({ webgpu: false, reason: errorText(error) }));
byId('report', HTMLElement).textContent = JSON.stringify(report, null, 2);
crumb('gpu', 'capabilities', 'report', report);
