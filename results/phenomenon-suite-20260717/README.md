# Phenomenon-model suite (2026-07-17)

This directory contains the measurements used to replace the previous
four-model ablation suite with GPT-2, BERT Base, ResNet-18, and ResNet-34.

## Provenance

- Host: `godel`
- Working-tree base: `28ea42bf5f672b230cb20edd78277ec5be4d6ed1`
- Branch: `experiment/phenomenon-suite`
- Dtype: BF16
- CPU threads: 12
- Timing protocol: one complete warm-up followed by one measured repetition
- Verification: every production-compressor and ZipNN implementation row was
  checked with a bitwise-exact round trip

The resumable entry point is
`examples/run_phenomenon_suite_godel.sh`. The copied `suite/` and `clustering/`
directories are the authoritative final outputs. `precomputed_suite.json`
contains the target-specific direct benchmark.

## Report inputs

- `precomputed_suite.json`: target-specific ZipNN/NeuralZip time and ratio
- `suite/component_all_models.csv`: component isolation
- `clustering/model_summary.csv`: representative-pair clustering analysis
- `suite/zipnn_implementation_all_models.csv`: local/source ZipNN comparison
- `suite/deferred_all_models.csv`: deferred techniques
- `suite/figures/`: source CSV files and rendered PDFs for the four figures

`run.log` preserves the complete initial execution, including an early
MM-RePair shape-selection failure for convolutional tensors. The selector was
then generalized to repeated tensors with at least two dimensions; the
successful replacement measurements are recorded in `deferred-rerun.log` and
the final aggregate CSV.
