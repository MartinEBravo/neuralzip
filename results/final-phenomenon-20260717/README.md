# Final phenomenon and ablation rerun

- Host: `godel`
- Date: 2026-07-17
- Threads: 12
- Models: GPT-2, BERT Base, ResNet-18, and ResNet-34
- Warm-up/repetitions: one complete warm-up followed by one measured pass where
  the corresponding experiment reports encoding time
- Verification: every codec row marked successful passed exact round-trip
- Runner: `examples/run_thesis_model_suite.py`
- ZipNN implementation rows were regenerated after commit `bdc292e` so their
  ratios cover only the exponent representation (symbols, raw fallback, and
  group metadata), while retaining full-checkpoint byte counts as diagnostics.

The large temporary MM-RePair integer streams (`*/mmrepair/`) remain on the
execution host and are intentionally excluded here.  The tracked CSV, JSON,
NPZ, PDF, and log artifacts are sufficient to reproduce every reported value
and figure.

`precomputed_suite.json` is retained as an ablation artifact comparing the
local ZipNN implementation.  The final main-table comparison against official
ZipNN source is produced separately by `run_general_codec_rerun.py`.
