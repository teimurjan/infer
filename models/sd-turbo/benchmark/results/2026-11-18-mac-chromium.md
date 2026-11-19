Browser: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36
Plan: 10 runs per runtime, 3 warm images per run

| runtime | successes | cold s (median) | warm s (median) | peak GPU buffers MB (max) |
| --- | --- | --- | --- | --- |
| this engine (Rust + WGSL) | 9 of 10 | 3.30 | 3.29 | 4134.6 |
| onnxruntime-web example | 10 of 10 | 7.73 | 1.58 | 5545.7 |

| runtime | run | ok | cold s | warm s | peak MB | error |
| --- | --- | --- | --- | --- | --- | --- |
| engine | 0 (warm-up) | yes | 3.64 | 3.27 3.35 3.22 | 2112.1 |  |
| engine | 1 | no | 3.30 | 3.34 3.27 | 3711.9 | text embeddings: one submit took 292.7 ms, the limit is 250.0 ms |
| engine | 2 | yes | 3.29 | 3.27 3.28 3.30 | 2179.3 |  |
| engine | 3 | yes | 3.32 | 3.32 3.28 3.27 | 2037.5 |  |
| engine | 4 | yes | 3.34 | 3.28 3.24 3.33 | 3779.1 |  |
| engine | 5 | yes | 3.28 | 3.32 3.31 3.32 | 2770.2 |  |
| engine | 6 | yes | 3.29 | 3.30 3.28 3.29 | 2770.2 |  |
| engine | 7 | yes | 3.29 | 3.29 3.30 3.31 | 2770.2 |  |
| engine | 8 | yes | 3.30 | 3.36 3.29 3.27 | 3644.7 |  |
| engine | 9 | yes | 3.33 | 3.32 3.33 3.26 | 4134.6 |  |
| engine | 10 | yes | 3.32 | 3.29 3.27 3.30 | 4115.0 |  |
| onnx | 0 (warm-up) | yes | 9.43 | 2.54 1.59 1.60 | 5545.7 |  |
| onnx | 1 | yes | 9.45 | 2.60 1.56 1.55 | 5545.7 |  |
| onnx | 2 | yes | 8.39 | 2.44 1.55 1.56 | 5545.7 |  |
| onnx | 3 | yes | 6.78 | 1.99 1.59 1.55 | 5545.7 |  |
| onnx | 4 | yes | 7.71 | 2.91 1.57 1.56 | 5545.7 |  |
| onnx | 5 | yes | 9.40 | 2.34 1.57 1.57 | 5545.7 |  |
| onnx | 6 | yes | 8.29 | 2.58 1.57 1.57 | 5545.7 |  |
| onnx | 7 | yes | 7.75 | 2.09 1.59 1.58 | 5545.7 |  |
| onnx | 8 | yes | 7.34 | 2.42 1.57 1.58 | 5545.7 |  |
| onnx | 9 | yes | 7.54 | 2.11 1.74 1.55 | 5545.7 |  |
| onnx | 10 | yes | 7.40 | 2.14 1.59 1.57 | 5545.7 |  |
