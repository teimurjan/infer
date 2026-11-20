// The addresses of the files of SD-Turbo on the server. Every other module gets its URLs from here.

// The model folder on the server. The server serves the repository, so this is the path of models/sd-turbo.
export const MODEL_ROOT = '/models/sd-turbo';

// The folder of the weight files on another server, for the deployed site. apps/site/build.ts puts the value of the
// SD_TURBO_WEIGHTS_URL environment variable here when it is set. Without it, this name does not exist.
declare const SD_TURBO_WEIGHTS_URL: string;
// typeof does not throw for a name that does not exist, so a build without the variable reads the weights from this server.
const WEIGHTS_ROOT = typeof SD_TURBO_WEIGHTS_URL === 'string' ? SD_TURBO_WEIGHTS_URL : `${MODEL_ROOT}/weights`;

export const urls = {
    // The WASM build of the engine, from `bun run build:wasm`.
    wasm: `${MODEL_ROOT}/engine/pkg/engine_bg.wasm`,
    // One weight file without its extension, for example weight('unet_q8') + '.bin'.
    weight: (name: string): string => `${WEIGHTS_ROOT}/${name}`,
    // One golden file, relative to golden/.
    golden: (path: string): string => `${MODEL_ROOT}/golden/${path}`,
    // One test file, relative to engine/testdata/.
    testdata: (path: string): string => `${MODEL_ROOT}/engine/testdata/${path}`,
};
