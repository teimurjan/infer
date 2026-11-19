Browser: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36
Plan: 10 runs per runtime, 3 warm images per run

| runtime | successes | cold s (median) | warm s (median) | peak GPU buffers MB (max) |
| --- | --- | --- | --- | --- |
| this engine (Rust + WGSL) | 10 of 10 | 3.33 | 3.32 | 4134.6 |

| runtime | run | ok | cold s | warm s | peak MB | error |
| --- | --- | --- | --- | --- | --- | --- |
| engine | 0 (warm-up) | yes | 3.96 | 3.36 3.32 3.41 | 3140.9 |  |
| engine | 1 | yes | 3.32 | 3.30 3.45 3.28 | 3342.4 |  |
| engine | 2 | yes | 3.33 | 3.31 3.31 3.29 | 2770.2 |  |
| engine | 3 | yes | 3.34 | 3.93 3.28 3.35 | 3846.3 |  |
| engine | 4 | yes | 3.30 | 3.31 3.31 3.29 | 2770.2 |  |
| engine | 5 | yes | 3.32 | 3.29 3.28 3.28 | 2770.2 |  |
| engine | 6 | yes | 3.33 | 3.29 3.44 3.29 | 3565.0 |  |
| engine | 7 | yes | 3.33 | 4.17 3.53 3.35 | 4134.6 |  |
| engine | 8 | yes | 3.35 | 3.33 3.44 3.97 | 3825.3 |  |
| engine | 9 | yes | 3.70 | 4.33 3.85 3.52 | 4131.8 |  |
| engine | 10 | yes | 3.34 | 3.30 3.54 3.34 | 3376.0 |  |
