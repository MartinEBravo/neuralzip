# Final general model-storage run

Final thesis artifacts produced on `godel` on 2026-07-17/18 with the general
model-storage runner.

- Scope: 28 checkpoints across phenomenon, vision-family, language-family,
  diffusion, distilled, and mixture-of-experts suites.
- Compression: official ZipNN source and NeuralZip, BF16 tensors, 12 pinned CPU
  threads (`0-11`), one verified warm-up and one measured encoding run.
- Timing: encode versus encode; ZipNN's defensive input copy is included in its
  non-destructive encoding path. NeuralZip setup is reported separately and is
  excluded from post-setup encoding time.
- Ratio: compressed exponent symbols plus raw fallbacks and metadata divided by
  the raw exponent-field bytes.
- Completeness: 56 rows = 28 models x 2 compressors. Every row has status `ok`,
  passed the bitwise round trip, and completed the verified warm-up.
- Provenance: `manifest.json` records the code, host, configuration, revision,
  selected weight artifacts, and weight fingerprints for the run. The files in
  this directory were copied byte-for-byte from
  `/tmp/neuralzip-general-final-runs-20260717/main` on `godel`; SHA-256 hashes
  for `manifest.json`, `progress.json`, `results.csv`, and `results.json` were
  checked against the remote originals after synchronization.

