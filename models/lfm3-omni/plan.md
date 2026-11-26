# Plan: LFM3-Omni-2B in Safari on the iPhone

Goal: a voice exchange with LFM3-Omni-2B in a web page on the iPhone. You speak, and the model answers in speech, in real time. Every number below is from `PROGRESS.md`.

## Where the work is now (2026-11-26)

| Measure | Value |
| --- | --- |
| Reference run | Done in Mac Python: "Paris" in 2 tokens. The golden set is `golden/text`. |
| Engine on the Mac | The word-by-word loop matches PyTorch: a 24-token answer to a 353-token prompt, scores within 6.72e-5 at every step. |
| Parameters | 2166928384 in the model, 73559090 in the codec |
| Language model at 4 bits, one scale per 32 | 972.2 MB, mean KL 0.2154 from float32 |
| Weights of a voice exchange, estimate | about 1186.7 MB: the language model at 4 bits, the speech parts at 8 bits |
| Speed for speech in real time | 12.5 passes of the language model per second |

## Steps, in order

1. Reference run and golden set of one text exchange (done).
2. The language model in the engine, on the Mac: the short convolution layer (done for layer 0, 2026-11-25: largest difference 2.26e-6), the two kinds of attention (done for layers 2 and 4, 2026-11-25: 2.86e-6 and 4.53e-6), and the output scores (done, 2026-11-25: all 24 layers, scores within 5.15e-5, best token "Paris"). Check each layer against `golden/text`.
3. The word-by-word loop with greedy choice and the memory of earlier tokens (done on the Mac, 2026-11-26: the 24-token answer of a 353-token prompt matches PyTorch, scores within 6.72e-5), then 4-bit weights with one float16 scale per 32 values (measured in Python, 2026-11-26: 972.2 MB, mean KL 0.2154 against 0.8778 with one scale per row, both answers still correct). Measure passes per second on the iPhone.
4. Fast matrix-vector kernels. The current matrix kernels reach 4.8 to 6.7 GB/s on the iPhone, against a copy speed of 47.7 GB/s. Real-time speech needs about 12.5 passes per second.
5. Speech in: the audio encoder and its projector, with a golden set of one spoken question.
6. Speech out: the depth decoder and the codec decoder, with a golden set of one spoken answer.
7. The voice page on the site, and "Talk 20" on the iPhone, like "Make 20" of SD-Turbo.
8. Images and video in, last.

## Open questions


- Whether the official ONNX export runs a text exchange in Safari on the iPhone. Not measured.
- The size of the codec decoder alone. The codec file also holds its encoder.
