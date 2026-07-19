# Expanded final family-transfer run

Final thesis artifacts produced on `godel` on 2026-07-19 with the runner from
commit `00ec816` and the strict result validator completed in `f058158`.

- Coverage: 11 frozen source plans, 20 transferred targets, and 31 unique
  checkpoints across ResNet BasicBlock, ResNet Bottleneck, ConvNeXt,
  EfficientNet, Qwen 2.5, Qwen 3.5, OLMo-2, Gemma 3, Llama 3.2, Whisper v3,
  and SigLIP 2.
- Model state: all non-empty floating entries in the evaluated state dictionary
  were converted to BF16. Gemma 3 uses the common text backbone; every other
  row uses the complete floating checkpoint.
- Transfer: each source builds its structural plan once. Targets reuse that
  frozen plan without target histograms, Jensen--Shannon matching, target mode
  probing, or an exact-target cache. Source tensors are released before target
  encoding.
- Measurement: 12 threads pinned to CPUs 0--11, one uncached verified warm-up,
  and one measured encode for transferred NeuralZip, target-specific NeuralZip,
  and official ZipNN source.
- Reproducibility: Hugging Face revisions are immutable commits; torchvision
  revisions, model artifacts, code, native modules, dependencies, host, and
  source-plan semantics are fingerprinted in every family JSON.
- Completeness: all 20 targets map 100% of their floating tensors and exponent
  symbols. All 60 evaluated codec paths passed bitwise round-trip verification.
- Release note: the OLMo-2 source and 7B target use the 1124 release, while the
  1B target uses the later 0425 release.

The 11 family JSON files are the authoritative results. `summary.json` and
`summary.csv` are strict derived views used to populate the report; the paired
logs preserve execution order and the complete emitted payloads. Model weights
are intentionally omitted from Git.
