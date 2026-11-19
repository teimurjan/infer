// The URLs that the onnxruntime-web runner reads from.

import { MODEL_ROOT } from '../client/urls.ts';

// The onnxruntime-web .wasm files, which build.ts copies into dist/.
export const ORT_WASM = `${MODEL_ROOT}/dist/`;
// The fp16 ONNX models of the onnxruntime-web example. `?model=<url>` on the page address replaces it.
export const ONNX_MODELS = new URLSearchParams(location.search).get('model') ?? 'https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/main';
