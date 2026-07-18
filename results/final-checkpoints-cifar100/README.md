# Final CIFAR-100 checkpoint run

Final thesis artifacts produced on `relela-06` on 2026-07-17/18 with
`examples/run_final_checkpoint_suite.py`.

- Dataset: real CIFAR-100 (`uoft-cs/cifar100`, Parquet mirror).
- Training: four vision architectures trained from random initialization for
  50 epochs with seed 2026; epoch 0 and every completed epoch were saved.
- Compression: ZipNN source and NeuralZip, 12 pinned CPU threads, one verified
  warm-up and one measured encode, exponent-field accounting, bitwise-exact
  round-trip.
- NeuralZip: one plan built at epoch 0 and frozen across each trajectory; setup
  and mapping are excluded from direct encoding time.
- Completeness: 408 rows = 4 models x 51 checkpoints x 2 compressors; all rows
  passed `roundtrip_ok` and `input_unchanged`.

The original `.pt` checkpoints remain in
`/data-local/martinb/neuralzip-final-runs-20260717/checkpoints-cifar100` on the
experiment host and are intentionally omitted from Git because the final CSV,
metadata, training trajectories, and rendered figure are sufficient to audit
the reported results.
