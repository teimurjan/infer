Browser: Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.6.2 Mobile/15E148 Safari/604.1
Plan: 10 runs per runtime, 3 warm images per run

| runtime | successes | cold s (median) | warm s (median) | peak GPU buffers MB (max) |
| --- | --- | --- | --- | --- |
| this engine (Rust + WGSL) | 10 of 10 | 5.36 | 5.64 | 1608.2 |

| runtime | run | ok | cold s | warm s | peak MB | error |
| --- | --- | --- | --- | --- | --- | --- |
| engine | 0 (warm-up) | no | - |  | 1364.4 | no GPU adapter: No suitable graphics adapter found; noop support not compiled in, vulkan support not compiled in, metal support not compiled in, dx12 support not compiled in, gl support not compiled in, webgpu found no adapters |
| engine | 1 | yes | 5.92 | 5.05 4.94 5.15 | 1450.9 |  |
| engine | 2 | yes | 5.06 | 5.18 5.35 5.38 | 1450.9 |  |
| engine | 3 | yes | 5.43 | 5.50 5.51 5.51 | 1608.2 |  |
| engine | 4 | yes | 5.52 | 5.38 5.47 5.67 | 1608.2 |  |
| engine | 5 | yes | 5.36 | 5.49 5.57 5.61 | 1608.2 |  |
| engine | 6 | yes | 5.42 | 5.51 5.82 6.64 | 1452.5 |  |
| engine | 7 | yes | 5.25 | 6.42 6.64 6.70 | 1608.2 |  |
| engine | 8 | yes | 5.22 | 6.71 6.70 7.16 | 1608.2 |  |
| engine | 9 | yes | 5.36 | 6.70 6.69 6.64 | 1608.2 |  |
| engine | 10 | yes | 5.24 | 6.56 6.61 6.61 | 1608.2 |  |
