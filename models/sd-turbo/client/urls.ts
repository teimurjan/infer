// The addresses of the files of SD-Turbo on the server. Every other module gets its URLs from here.

// The model folder on the server. The server serves the repository, so this is the path of models/sd-turbo.
export const MODEL_ROOT = '/models/sd-turbo';

export const urls = {
    // The WASM build of the engine, from `bun run build:wasm`.
    wasm: `${MODEL_ROOT}/engine/pkg/engine_bg.wasm`,
    // One weight file without its extension, for example weight('unet_q8') + '.bin'.
    weight: (name: string): string => `${MODEL_ROOT}/weights/${name}`,
    // One golden file, relative to golden/.
    golden: (path: string): string => `${MODEL_ROOT}/golden/${path}`,
    // One test file, relative to engine/testdata/.
    testdata: (path: string): string => `${MODEL_ROOT}/engine/testdata/${path}`,
};
