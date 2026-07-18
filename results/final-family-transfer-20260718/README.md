# Final family-transfer run

Final thesis artifacts produced on `relela-06` on 2026-07-18 with commit
`442a846` and `examples/run_precompute_family_transfer.py`.

- Pairs: ImageNet-1K ResNet-34 to ResNet-18, Qwen 3.5 2B Base to 0.8B Base,
  and Gemma 3 4B PT to 1B PT.
- Model state: all non-empty floating entries in each `state_dict`, including
  persistent floating buffers, converted to BF16.
- Transfer: direct structural reuse of the frozen source plan. Target
  histograms, Jensen--Shannon matching, target mode probing and exact-target
  materialization are disabled.
- Measurement: 12 pinned CPU threads, one uncached verified warm-up and one
  uncached measured encode for each path. Family setup, structural mapping,
  decoding and metric accounting are excluded from direct encoding time.
- Reproducibility: Hugging Face pairs use immutable pinned commits; every
  authoritative shard and revision is recorded and fingerprinted in its JSON.
- Completeness: all three pairs mapped 100% of the target floating tensors and
  exponent symbols. ZipNN source, target-specific NeuralZip and transferred
  NeuralZip all passed bitwise round-trip verification.

The JSON files are the authoritative results. Logs and `run.log` preserve the
execution order and completion markers; model snapshots remain on the
experiment host and are intentionally omitted from Git.
