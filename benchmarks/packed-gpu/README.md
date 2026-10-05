# Packed GPU decode acceptance

Baseline: `MartinEBravo/nzip` main `453c06ea019b8d1d2566de0b4a7396b34807425c`.
Only GPU runtime/decoding may change. CPU setup, codebooks, archives, compressed
payloads, restart metadata, selected weights and model arithmetic stay frozen.
Do not merge or update the paper from this experiment.

Current status: **draft, incomplete**. Five models pass the first campaign's
ten own/transfer comparisons, with unchanged allocation and exact results.
SigLIP exposed a pre-existing functional-consumer hook problem in both the
unchanged NeuralZip baseline and the DFloat11 control adapter. Correcting that
baseline-control boundary needs approval before two new full campaigns.
See [RESULTS.md](RESULTS.md) for every completed measurement, the untimed
diagnosis and preserved failure/measurement archives. No failed or missing
control counts as passing.

## Inputs and environment

On `multi_vm`, select one physical GPU explicitly with `--gpu`. The original
plan selected GPU 1; Martin subsequently authorized GPU **0** and releasing
his idle Jupyter kernel's GPU memory. The experiment uses GPU 0 throughout.
The launcher never signals another process and waits if the selected GPU is
occupied. Use separate immutable baseline/candidate source
captures, the existing native extension, and the same installed dependencies.
Do not run historical launch/recovery scripts: some suspend unrelated queues.

The benchmark reads the existing server-local data:

* `DATA=/home/martin/neuralzip-runs/transfer-full-rerun-20261005-0244`
* `INPUTS=/dev/shm/martin-nzip-six-20261002`
* `DFLOAT=/home/martin/neuralzip-runs/chunk-rerun-20261002/dfloat-source`
* `DEPS=/home/martin/neuralzip-runs/chunk-rerun-20261002/dependencies`

Include `$DATA`, the chosen source capture, `$DFLOAT`, and `$DEPS` on `PYTHONPATH`.
Use the existing pinned HF cache, offline, without reading or copying credentials.
`six_gpu_reference.py` is used only for pinned model construction, exact tensor
checks and deterministic inputs. Its SHA256 is checked by the benchmark, as are
the official DFloat11 Python, encoder, CUDA source and PTX files. The official
source corresponds to commit `457733886ce6ebc6d8dda1621fad1ffa2661e028`.
The harness changes only the DFloat11 injection adapter to restore original
Linear/Conv2d/Conv1D shapes; its codec and kernel are untouched. Any genuine
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
