# Packed GPU decode acceptance

Baseline: NeuralZip main `453c06ea019b8d1d2566de0b4a7396b34807425c`.
Only GPU runtime/decoding may change. CPU setup, codebooks, archives, compressed
payloads, restart metadata, selected weights and model arithmetic stay frozen.

Current status: **acceptance passed; implementation integrated into main**.
All 24 own/transfer comparisons pass in
two complete, independent campaigns. The worst latency is 1.026630× DFloat11;
live and forward-peak allocation exactly match baseline and remain below BF16.
The same general functional-consumer hook correction is applied to
baseline, candidate and the DFloat11 control adapter. CPU setup, archives,
payload/restart bytes, coverage and arithmetic remain unchanged.
See [RESULTS.md](RESULTS.md) for every model's measurements, exact validation,
reproduction commands and lossless raw evidence, including preserved failures.
The packaged evidence uses anonymous paths and archive ownership metadata;
measurements and input/setup/payload/runtime-source hashes are unchanged.

## Checking the archived campaigns

The historical campaigns ran on physical GPU **0**, an NVIDIA RTX A5500.
Their source captures, inputs, frozen setup/payload hashes, individual trials
and both campaign identities remain in these archives. No weights or pickles
are distributed. The original measured harness/check sources are retained
inside the evidence where captured, not exposed as extra public scripts.

From the repository root, with the [experiment environment](../README.md):

```bash
python scripts/six_models_ablations.py --check
```

This reads both ZIPs directly, checks their CRCs, recomputes all medians and
24 acceptance comparisons, and checks the 48 sharing/packing controls. It does
not unpack files, download weights, compile CUDA or rerun inference.
Acceptance requires candidate latency ≤1.10× DFloat11, live and peak allocation
≤1.05× baseline and each below BF16, exact weights/outputs, frozen input/setup,
identical coverage, CPU archive/payload/restart identities and independent
correctness proofs. A speedup alone is not a pass.

## Fresh measurements of the current implementation

Use `scripts/six_models_ablations.py --device gpu` as documented in
[../README.md](../README.md), with one explicit `--gpu`, pinned inputs and frozen
CPU setups. Models are `gemma3`, `parakeet`, `dinov3-convnext`, `qwen35`,
`dinov3-vit` and `siglip2`. Run BF16 first, then own/transfer/DFloat11, each in a
fresh process. New runs are not historical baseline measurements and do not
overwrite the acceptance archives. The benchmark refuses an occupied GPU;
it never signals other processes. Use tmux for long runs.

The model/input adapter is now bundled in `neuralzip/_experiment_inputs.py`;
there is no dependency on a private `six_gpu_reference.py`. Official DFloat11
source remains pinned to `457733886ce6ebc6d8dda1621fad1ffa2661e028` with exact
Python/encoder/CUDA/PTX hashes checked. Its shape/functional-consumer injection
adapter is unchanged in scope; genuine singleton/LUT fallbacks remain BF16.

Batch one, eager attention, no cache/offload, eight CPU threads on affinity
0–15, 20 warmups, five synchronized 50-forward trials. Preparation, export,
compilation, validation, decoder-only measurement and profiling stay outside
full-forward timing. Decoder-only trials use separate CUDA events. Live/peak
allocation and reservation are separate, including all runtime metadata;
validation references stay on CPU and weights are decoded on every forward.
`--check-kernels` runs the CUDA boundary/tail/long-code and module/save-reload
checks. Preserve failed/intermediate runs and use new output directories.

## Preserved pilot evidence

`evidence/staging/pilot-records.zip` contains all 18 original JSON records from
the three-model pilot, including every timing trial and input/setup/payload
hash. ZIP CRCs and exact equality to each original JSON were independently
checked before packaging; the unchanged originals also remain on the server.
Verify the archive with `shasum -a 256 -c pilot-records.sha256` from its folder.
Pilot measurements are not counted toward the two full acceptance campaigns.

| Pilot model | DFloat11 ms | Own baseline → candidate ms | Transfer baseline → candidate ms |
| --- | ---: | ---: | ---: |
| Gemma3 | 19.541 | 29.161 → 18.605 | 29.204 → 18.963 |
| Parakeet | 12.703 | 20.132 → 11.992 | 20.296 → 12.068 |
| DINOv3 ConvNeXt | 3.513 | 5.202 → 3.630 | 5.136 → 3.520 |

Every pilot candidate has exactly its baseline's live and peak allocation,
both below BF16, with bitwise weights/outputs and unchanged compressed bytes.

## Sharing-only and packing-only GPU ablations

The ablation rerun uses the frozen own setups from the CPU campaign
`shared-table1-ablations-20261005`, which supplied the
current paper's CPU rows. Copy only the twelve `*-sharing-own.setup.pkl` /
`*-packing-own.setup.pkl` files and their original method JSONs to
the GPU host; do not rebuild a setup for this frozen comparison.
The benchmark verifies each pickle against the original CPU record's
`own_setup_sha256` and the exact pinned BF16 manifest before export.

For fresh current-code measurements, use the public six-suite script's
`--methods sharing` / `--methods packing`, supplying frozen CPU setups and a
same-campaign BF16 reference. For the recorded historical baseline/candidate
comparison, `--check` verifies all 48 preserved controls, recomputes medians,
checks bitwise weights/outputs, frozen input/setup/payload/coverage/source
identities, and live/peak memory.
DFloat11 ratios are reported honestly; the full-method 10% speed target is
not imposed on the isolated ablations. The original full-method acceptance
checker and evidence remain unchanged in scope.
