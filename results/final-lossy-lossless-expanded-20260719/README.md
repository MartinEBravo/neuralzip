# Expanded lossy-lossless evaluation

Final thesis artifacts produced on `godel` on 2026-07-19 with the compressor
implementation and model configuration published in commit `2a6a2d5`.

## Distilled checkpoints

The two output directories jointly contain four full BF16 distilled
checkpoints and both compressors for each checkpoint. They were split only to
reuse already materialized checkpoints; all eight rows use the same code digest
(`44cd05d85f8be148a6cb1f4461587d2c2e9fa2fd5a5e1b5d7a7893dd7c59c2fe`),
12 threads pinned to CPUs 0--11, one verified warm-up, and one measured run.
Every row completed an exact bitwise round-trip.

- `distilled-deepseek-current/`: the official DeepSeek-R1-Distill-Llama-8B
  and DeepSeek-R1-Distill-Qwen-7B checkpoints.
- `distilled-existing-current/`: the Gemma 2 9B and Qwen 2.5 7B distilled
  checkpoints rerun with the same current code rather than reusing older
  timings.

Across the four checkpoints, NeuralZip is faster in two and slower in two. Its
exponent ratio differs from ZipNN source by +0.007 to +0.084 percentage points.

## Quantized artifacts

`quantized-all-current/` contains one homogeneous rerun of all six artifacts:
three bitsandbytes safetensors checkpoints, one AWQ checkpoint, one GPTQ
checkpoint, and the LATAM-GPT GGUF checkpoint. The measured phase ran offline
with 12 threads pinned to CPUs 12--23, one verified warm-up, and one measured
run. All twelve rows passed exact byte-for-byte round-trip verification.

The probe samples at most 64 MiB from every authoritative weight shard and
reports 0.158--2.409% coverage of the available tensor-data regions. It is a
nibble-stream probe, not a full-checkpoint ratio. `run_metadata.json` records
the exact runner, configuration, Huff0 backend, CPU affinity, and sample
fingerprints used by the run.

Across the six artifacts, shared coding is nominally faster in four cases and
slower in two. Apart from one 1.79x improvement, time differences stay within
8%. Its ratio is 0.000--4.821 percentage points higher than local Huffman.

## Validation

Run from the repository root:

```bash
python examples/validate_lossy_lossless_expansion.py
```

The validator requires 8/8 distilled and 12/12 quantized rows, exact model and
compressor coverage, pinned revisions, Godel host provenance, the stated CPU
affinities, one warm-up and one measurement, matching source hashes, and exact
round-trips.
