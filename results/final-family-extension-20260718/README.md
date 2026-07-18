# Final language-family and MoE extension

Final thesis artifacts produced on `godel` on 2026-07-18 with the general codec
workers and the sequential tmpfs streaming orchestrator stored in this
directory.

- Scope: 20 pinned Hugging Face checkpoints: 13 language-family models for the
  model-storage suite and 7 mixture-of-experts models.
- Compression: official ZipNN source and NeuralZip, BF16 tensors, 12 pinned CPU
  threads (`0-11`), one verified warm-up, and one measured encoding run.
- Timing: encode versus encode. ZipNN's defensive input copy is included in its
  non-destructive encoding path. NeuralZip setup is reported separately and is
  excluded from post-setup encoding time.
- Ratio: compressed exponent symbols plus raw fallbacks and metadata divided by
  the raw exponent-field bytes.
- Completeness: 40 rows = 20 models x 2 compressors. Every row has status `ok`,
  passed the bitwise round trip, and completed the verified warm-up. The 20
  receipts bind both compressor rows to the same pinned checkpoint weights.
- Execution: checkpoints were streamed one at a time through tmpfs to avoid the
  degraded persistent array. Compressor order alternated across models; models
  were not executed concurrently.
- Provenance: `manifest.json` records the host, configuration, code digest,
  pinned revisions, selected weight artifacts, and weight fingerprints. The
  measured code digest is
  `96e6f8ee716ba729a61f6cce96897546d61cd70e165ce5221fb9416376d0b508`.
- Validation: `thesis-summary.json` contains the strict combined summary over
  the 56-row baseline and this 40-row extension. Validation found all 34
  model-storage and 10 MoE pairs, with no missing or duplicate models.

The durable artifacts were copied from
`/data/mbravo/neuralzip-final-family-20260718/results/extension-tmpfs-20260718`
on `godel`. `run.log` is the complete orchestration log and ends with
`complete=1 jobs=40`.

SHA-256 checksums of the primary artifacts after synchronization:

```text
fd9161387b34ecbd002f06f96ccb40ba5ef4bad88786d6ea8bd92d5ac1e8d294  manifest.json
6fec2c12c9f7b745b694130504cb07de1d4645333cd9bdac9d0e5d82d8f7d975  progress.json
4f1a08ac86ca2c2d58915ccf3e1c78164477322de20ff5a87850607890b158bd  results.json
7c2daccd8cdb021bcd8d1ade6b6c2d6d87a1444a3662c5bf3cffbeb99e3f05e3  results.csv
```
