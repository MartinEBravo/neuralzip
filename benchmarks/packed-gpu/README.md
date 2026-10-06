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

## Inputs and environment

On the GPU host, select one physical GPU explicitly with `--gpu`.
The reported campaigns use physical GPU **0**, an NVIDIA RTX A5500.
The launcher never signals another process and waits if the selected GPU is
occupied. Use separate immutable baseline/candidate source
captures, the existing native extension, and the same installed dependencies.
Do not run historical launch/recovery scripts: some suspend unrelated queues.

The benchmark reads the existing server-local data:

* `DATA=/path/to/frozen-cpu-results`
* `INPUTS=/path/to/pinned-bf16-inputs`
* `DFLOAT=/path/to/pinned-dfloat11-source`
* `DEPS=/path/to/benchmark-dependencies`

Replace these paths with the corresponding local benchmark inputs.

Include `$DATA`, the chosen source capture, `$DFLOAT`, and `$DEPS` on `PYTHONPATH`.
Use the existing pinned HF cache, offline, without reading or copying credentials.
`six_gpu_reference.py` is used only for pinned model construction, exact tensor
checks and deterministic inputs. Its SHA256 is checked by the benchmark, as are
the official DFloat11 Python, encoder, CUDA source and PTX files. The official
source corresponds to commit `457733886ce6ebc6d8dda1621fad1ffa2661e028`.
The harness changes only the DFloat11 injection adapter to restore original
Linear/Conv2d/Conv1D shapes and hook functional MultiheadAttention projections
on their consuming parent. That same hook-placement adapter is applied to the
baseline and candidate NeuralZip controls, without altering baseline source,
packing, selected weights or persistent GPU storage. Codec and kernel are untouched. Any genuine
official singleton/LUT fallback remains BF16 and is reported explicitly.

## Campaigns

Run `scripts/check_packed_gpu_results.py --self-test` first (standard library only).
Run `scripts/bench_packed_gpu.py MODEL METHOD --variant VARIANT --campaign 1 --gpu 0
--data-root "$DATA" --inputs "$INPUTS" --output "$OUTPUT"` in a fresh process
for each control. Models: `gemma3`, `parakeet`, `dinov3-convnext`, `qwen35`,
`dinov3-vit`, `siglip2`. Methods/variants for each model:

1. `bf16/control` (creates a new output reference inside the campaign).
2. `dfloat11/control` (fresh official inference control).
3. `own/baseline`, `transfer/baseline` using unchanged main.
4. `own/candidate`, `transfer/candidate` using the candidate source capture.

Repeat **all six controls**, in separate processes, for campaign `2`. Reverse
the baseline/candidate order in the second campaign. Pilot records go under
`--campaign pilot`, never into the acceptance campaigns. Failed/intermediate
versions and historical data are retained, not overwritten or cherry-picked.

Batch one, eager attention, no cache/offload, eight CPU threads on affinity
0–15, 20 warmups, five synchronized 50-forward trials. Preparation, export,
compilation, validation, decoder-only measurement and profiling are excluded
from full-forward trials. Decoder-only trials use CUDA events separately.
Live allocation and peak allocation during measured forwards are reported
separately from reservation; validation references remain on CPU. Every
compressed weight is reconstructed on each forward. Runtime descriptor and
additional metadata allocations are included in CUDA memory measurements.

`python scripts/check_packed_gpu_results.py "$OUTPUT" --baseline-cpu "$BASELINE_CPU_JSON"` exits nonzero unless
all **24 comparisons** (12 own/transfer comparisons in each of two campaigns)
meet: candidate latency ≤1.10× DFloat11; candidate live and peak allocation
≤1.05× unchanged NeuralZip and each below BF16; exact weights/outputs, frozen
inputs/setup, byte-identical payloads/restart metadata/CPU implementation,
identical weight coverage, identical versions across campaigns.

Run the CUDA correctness script and existing supported-module/save-reload
checks before campaigns. CPU archive identity is checked against the baseline
fixture, not inferred from a latency improvement. A speedup alone is not a pass.

`scripts/run_packed_gpu_campaign.py` runs the edge/module checks and a serial
three-model pilot (`--phase pilot`), or both full campaigns (`--phase campaigns`).
Use it in tmux `kt` with explicit baseline/candidate, data, input, official
DFloat11, dependency and HF-cache paths. It holds a lock to prevent duplicate
launches and checks device occupancy before each subprocess. The benchmark
also rejects another process appearing during a timed control. Source captures
are immutable: use a new candidate directory for each optimization, not an
in-place edit of a running experiment. Measurements are never overwritten.

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

Use `run_packed_gpu_campaign.py --phase ablations` with the same explicit
source/input/environment arguments as above, a new output `--root`,
`--ablation-root` pointing to those frozen files, and `--reference-root`
pointing to the completed full campaigns. This runs only sharing/packing
baseline/candidate controls on all six models twice, reversing order in run 2.
The unchanged BF16 outputs and 72 validated full-suite records are reused,
not rerun or overwritten. CPU setup/compression measurements are unchanged.

`check_packed_gpu_results.py OUTPUT --ablations --reference-root FULL_OUTPUT
--baseline-cpu BASELINE_CPU_JSON` checks all 48 fresh ablation controls,
recomputes all medians, verifies bitwise weights/outputs and frozen
input/setup/payload/coverage/source identities, and checks live/peak memory.
DFloat11 ratios are reported honestly; the full-method 10% speed target is
not imposed on the isolated ablations. The original full-method acceptance
checker and evidence remain unchanged in scope.
