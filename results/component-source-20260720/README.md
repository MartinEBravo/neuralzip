# Component isolation rerun

This directory contains the four-model component-isolation rerun executed on
`godel`. The baseline is the official ZipNN 0.5.4 implementation; report tables
label it simply as `ZipNN`.

- Models: GPT-2, BERT Base, ResNet-18, and ResNet-34 in BF16.
- Threads: 12 pinned CPU threads.
- Timing: one verified warm-up followed by one measured repetition.
- Verification: every variant passed a bitwise-exact round trip.
- ZipNN timing includes the defensive input copy required by version 0.5.4.
- Ratios cover the exponent payload plus each compressor's native metadata.

`component_all_models.csv` is the aggregate report input. Per-model CSV files,
metadata, and execution logs are retained in the same directory.
