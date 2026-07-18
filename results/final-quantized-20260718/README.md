# Final quantized-artifact probe

This directory contains the final lossless nibble-compression probe run on
Godel on 2026-07-18. The input is the real tensor-data region of the selected
quantized artifacts. The probe samples at most 64 MiB from each authoritative
weight shard; it is not a full-model compression result, so every reported row
includes the sampled-byte coverage of the available tensor payload.

The artifact selectors are deliberately narrow: the authoritative safetensors
index and all of its referenced shards for Gemma and Phi, the exact
`model.safetensors` file for Qwen, and only `latamgpt-Q4_K_M.gguf` for
LatamGPT. `prepared_samples.json` records the exact resolved revision,
selector, selected files, offsets, hashes, and sample fingerprint.

| Model | Selected shards/files | Sampled bytes | Available tensor bytes | Coverage |
|---|---:|---:|---:|---:|
| Gemma 4 31B-4bit | 4 | 268,435,456 | 19,081,358,329 | 1.406794% |
| Phi-4 14B-4bit | 2 | 134,217,728 | 9,088,455,359 | 1.476794% |
| Qwen3.5 27B-4bit | 1 | 67,108,864 | 17,898,198,238 | 0.374948% |
| LatamGPT-70B-4bit | 1 | 67,108,864 | 42,512,543,040 | 0.157857% |

Preparation used the four pinned revisions recorded in
`prepared_samples.json`. The measured phase was then run offline with CPU
affinity 12--23, 12 Huff0 threads, one verified warm-up, and one measured
repetition. Remote and local sample I/O are outside encoding time. Every one of
the eight result rows passed exact byte-for-byte round-trip verification. The
ratio covers symbols, raw fallbacks, referenced codebooks, and metadata.

- `quantized_nibble_compression.csv` and `.json`: eight measured rows (two
  compressors per model).
- `run_metadata.json`: host, affinity, backend fingerprints, and protocol.
- `prepared_samples.json`: artifact provenance and sample coverage.
- `prepare.log`, `run.log`, and the matching `.exit` files: successful final
  execution logs and exit codes.
