# Plan: LFM3-Omni-2B in Safari on the iPhone

Goal: a voice exchange with LFM3-Omni-2B in a web page on the iPhone. You speak, and the model answers in speech, in real time. Every number below is from `PROGRESS.md`.

## Where the work is now (2026-11-24)

| Measure | Value |
| --- | --- |
| Reference run | Done in Mac Python: "Paris" in 2 tokens. The golden set is `golden/text`. |
| Parameters | 2166928384 in the model, 73559090 in the codec |
| Weights of a voice exchange, estimate | about 1186.7 MB: the language model at 4 bits, the speech parts at 8 bits |
| Speed for speech in real time | 12.5 passes of the language model per second |

## Steps, in order

1. Reference run and golden set of one text exchange (done).
2. The language model in the engine, on the Mac: the short convolution layer, the two kinds of attention, the memory of earlier tokens, and the output scores. Check each layer against `golden/text`.
3. The word-by-word loop with greedy choice, and 4-bit weights. Measure passes per second on the iPhone.
4. Fast matrix-vector kernels. The current matrix kernels reach 4.8 to 6.7 GB/s on the iPhone, against a copy speed of 47.7 GB/s. Real-time speech needs about 12.5 passes per second.
5. Speech in: the audio encoder and its projector, with a golden set of one spoken question.
6. Speech out: the depth decoder and the codec decoder, with a golden set of one spoken answer.
7. The voice page on the site, and "Talk 20" on the iPhone, like "Make 20" of SD-Turbo.
8. Images and video in, last.

## Open questions

- Whether the official ONNX export runs a text exchange in Safari on the iPhone. Not measured.
- The size of the codec decoder alone. The codec file also holds its encoder.
