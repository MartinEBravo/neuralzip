# Packed GPU decoding: acceptance passed

All **24 own/transfer comparisons** passed in **two complete independent campaigns** (72 fresh control records). The worst candidate latency is **1.026630× DFloat11**, below the 1.10× limit. Candidate live and forward-peak allocation exactly equal baseline in every comparison, and both are below BF16. There are no remaining acceptance gaps on these six pinned models and this GPU.

Shared-memory exponent-byte staging was sufficient. No output-count cache, prefix-width change, packing removal, decoded-weight retention, offload or fused matrix-multiplication redesign was added. CPU setup, codebooks, CPU archives, GPU payload/restart bytes, selected/compressed coverage and model arithmetic remain unchanged. The only additional production correction routes functional `MultiheadAttention.out_proj` decoding through its consuming attention module; Martin approved the same adapter for original-main NeuralZip and the DFloat11 control.

## Full-forward latency

Each entry is the median of five synchronized 50-forward trials after 20 warmups, batch one, eager attention, no cache/offload. Every control ran in a fresh process. Campaign 2 reversed candidate/baseline order. Initialization, compilation, export, validation, decoder-only trials and profiling are outside full-forward timing. Units: milliseconds. Baseline means main `453c06ea019b8d1d2566de0b4a7396b34807425c` with the approved functional-consumer adapter; its decoder is unchanged.

### Campaign 1

| Model | BF16 | DFloat11 | Own baseline → candidate | Transfer baseline → candidate | Own / DFloat11 | Transfer / DFloat11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 41.944 | 44.070 | 45.033 → 42.028 | 45.157 → 42.631 | 0.9537× | 0.9673× |
| Gemma3-1B | 12.588 | 19.619 | 29.145 → 18.609 | 29.176 → 18.933 | 0.9485× | 0.9651× |
| DINOv3 ConvNeXt-B | 2.183 | 3.523 | 5.219 → 3.617 | 5.137 → 3.502 | 1.0266× | 0.9940× |
| DINOv3 ViT-B/16 | 2.321 | 3.246 | 4.453 → 3.110 | 4.471 → 3.097 | 0.9580× | 0.9541× |
| SigLIP2-B | 3.435 | 5.100 | 7.723 → 4.893 | 7.742 → 4.876 | 0.9594× | 0.9561× |
| Parakeet TDT-0.6B-v3 | 9.113 | 12.700 | 20.144 → 11.989 | 20.288 → 12.102 | 0.9441× | 0.9530× |

### Campaign 2

| Model | BF16 | DFloat11 | Own baseline → candidate | Transfer baseline → candidate | Own / DFloat11 | Transfer / DFloat11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 41.698 | 43.862 | 45.166 → 42.075 | 45.203 → 42.327 | 0.9593× | 0.9650× |
| Gemma3-1B | 12.738 | 19.604 | 29.413 → 18.575 | 29.303 → 18.921 | 0.9475× | 0.9651× |
| DINOv3 ConvNeXt-B | 2.194 | 3.519 | 5.246 → 3.613 | 5.132 → 3.497 | 1.0266× | 0.9938× |
| DINOv3 ViT-B/16 | 2.305 | 3.232 | 4.456 → 3.100 | 4.449 → 3.102 | 0.9592× | 0.9597× |
| SigLIP2-B | 3.567 | 5.095 | 7.772 → 4.868 | 7.782 → 4.845 | 0.9554× | 0.9509× |
| Parakeet TDT-0.6B-v3 | 9.234 | 12.749 | 20.377 → 11.858 | 20.435 → 12.013 | 0.9301× | 0.9423× |

## Memory and correctness

All allocation values below are identical across both campaigns. Live means PyTorch-allocated GPU tensors; peak is reset after warmups and measured during full-forward trials. Reserved allocation is separately recorded in every raw JSON and is not reported as live memory. Existing runtime descriptors are included; **zero additional persistent GPU metadata** was added. MiB are rounded here; raw records preserve exact bytes.

| Model / method | Baseline live → candidate live MiB | Baseline peak → candidate peak MiB | BF16 live / peak MiB | Memory / bitwise |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-0.8B / own | 1328.006 → 1328.006 | 1388.881 → 1388.881 | 1664.215 / 1725.090 | Pass / pass |
| Qwen3.5-0.8B / transfer | 1333.678 → 1333.678 | 1394.553 → 1394.553 | 1664.215 / 1725.090 | Pass / pass |
| Gemma3-1B / own | 1554.856 → 1554.856 | 1619.137 → 1619.137 | 1965.660 / 2029.941 | Pass / pass |
| Gemma3-1B / transfer | 1573.679 → 1573.679 | 1637.960 → 1637.960 | 1965.660 / 2029.941 | Pass / pass |
| DINOv3 ConvNeXt-B / own | 142.162 → 142.162 | 149.052 → 149.052 | 176.438 / 185.078 | Pass / pass |
| DINOv3 ConvNeXt-B / transfer | 143.718 → 143.718 | 150.609 → 150.609 | 176.438 / 185.078 | Pass / pass |
| DINOv3 ViT-B/16 / own | 137.139 → 137.139 | 141.645 → 141.645 | 179.799 / 184.877 | Pass / pass |
| DINOv3 ViT-B/16 / transfer | 138.630 → 138.630 | 143.136 → 143.136 | 179.799 / 184.877 | Pass / pass |
| SigLIP2-B / own | 635.402 → 635.402 | 641.521 → 641.521 | 740.030 / 746.150 | Pass / pass |
| SigLIP2-B / transfer | 638.274 → 638.274 | 644.394 → 644.394 | 740.030 / 746.150 | Pass / pass |
| Parakeet TDT-0.6B-v3 / own | 931.244 → 931.244 | 947.245 → 947.245 | 1205.285 / 1221.286 | Pass / pass |
| Parakeet TDT-0.6B-v3 / transfer | 938.216 → 938.216 | 954.217 → 954.217 | 1205.285 / 1221.286 | Pass / pass |

Every control passes bitwise weights/outputs and unchanged input checks. Baseline/candidate setup, payload/restart hashes and compressed coverage match, as do CPU/native sources and the exporter AST. The CUDA proof covers raw/singleton/packed routes, FP16/BF16/FP32, odd tails, segment boundaries, long codes through 56 bits, staged and direct-write fallback paths, and public compressed save/reload. Module fixtures cover Linear, Conv2d, Transformers Conv1D, CNNs and a packed functional attention projection with poisoned workspace and an existing user hook. The final checker also recomputes full-forward and decoder-only medians and rejects missing records, duplicate records, changed inputs/payloads, invalid samples and individual latency/memory failures.

## Decoder-only latency

Separate CUDA-event medians, milliseconds; not included as extra work in full-forward trials.

| Campaign / model | DFloat11 decoder | Own baseline → candidate decoder | Transfer baseline → candidate decoder |
| --- | ---: | ---: | ---: |
| 1 / Qwen3.5-0.8B | 6.455 | 15.225 → 5.546 | 15.268 → 5.734 |
| 1 / Gemma3-1B | 7.192 | 17.087 → 6.104 | 17.042 → 6.449 |
| 1 / DINOv3 ConvNeXt-B | 1.331 | 3.038 → 1.426 | 2.939 → 1.314 |
| 1 / DINOv3 ViT-B/16 | 1.033 | 2.280 → 0.888 | 2.278 → 0.888 |
| 1 / SigLIP2-B | 2.124 | 4.808 → 1.884 | 4.842 → 1.872 |
| 1 / Parakeet TDT-0.6B-v3 | 5.881 | 13.417 → 4.950 | 13.530 → 5.063 |
| 2 / Qwen3.5-0.8B | 6.463 | 15.289 → 5.559 | 15.264 → 5.752 |
| 2 / Gemma3-1B | 7.247 | 17.249 → 6.070 | 17.099 → 6.434 |
| 2 / DINOv3 ConvNeXt-B | 1.329 | 3.040 → 1.419 | 2.934 → 1.314 |
| 2 / DINOv3 ViT-B/16 | 1.029 | 2.290 → 0.886 | 2.278 → 0.889 |
| 2 / SigLIP2-B | 2.112 | 4.849 → 1.870 | 4.846 → 1.864 |
| 2 / Parakeet TDT-0.6B-v3 | 5.915 | 13.614 → 4.901 | 13.629 → 4.990 |

## Environment and provenance

Physical GPU **0**, NVIDIA RTX A5500, UUID `05b2e34a-1171-95f1-c409-01fc7bb80386`; driver `595.91.07`. PyTorch `2.9.1+cu128`, CUDA `12.8`, CuPy `13.6.0`. Eight CPU threads, affinity 0–15. GPU 1 and its existing job were untouched. Both campaigns have identical software/input/source identities. All experiment processes have finished; GPU 0 is idle.

- Baseline main: `453c06ea019b8d1d2566de0b4a7396b34807425c` (original source capture unchanged).
- Candidate production source: `64f6598a524344a5e457e8a3a1cf92a58e167e95`; subsequent changes are tests/docs/evidence only.
- Official DFloat11: `457733886ce6ebc6d8dda1621fad1ffa2661e028`; Python/encoder/CUDA/PTX hashes are checked by the harness, not modified.
- Benchmark SHA256: `503cfbc7878a927a3c5b4c8ef9d79224cb29897cb539d569faedf6c1977ede79`.
- Final checker SHA256: `8de567eae5e66129bf51646044b533334e682db3cf2b1e2efc1715a4d5dda02c`.
- Pinned input-loader SHA256: `d570384eb28a6a148929e1aef6ceeb64309f87c75cb44d158dcd1a45453885f3`.
- Approved control adapter: `multihead-attention-out-proj-v1`.

| Pinned model | Revision |
| --- | --- |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
| `google/gemma-3-1b-pt` | `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` |
| `facebook/dinov3-convnext-base-pretrain-lvd1689m` | `e619358a5fe272554f17d2fb6cc65a679326bac0` |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | `5931719e67bbdb9737e363e781fb0c67687896bc` |
| `google/siglip2-base-patch16-224` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |
| `nvidia/parakeet-tdt-0.6b-v3` | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` |

| Candidate GPU source | SHA256 |
| --- | --- |
| `gpu/inference.py` | `e1a54a809aa9a557a8a7634b2783230e84565eff854a6d7f878261548d92bccb` |
| `gpu/decoder.py` | `473737d8c1b0cec0c86cf25ce96fe9a150220a33daf9e118552bbe9c1a0498bd` |
| `gpu/cuda/decode.cu` | `4dd6e493f60ab77143444c7afcddc819164fca424966a267a55b0fa8d57d51ff` |

## Lossless evidence and reproduction

`evidence/staging-functional/campaign-records.zip` contains **75 members**: 72 original timing JSONs, both initial/final run logs and the initial small-fixture source. It decodes to **2305214 bytes**; SHA256 `438669785f919c17a4e18ef0521690f502d535f01c71389d2eabe0f63218e3dc`. Every ZIP CRC, member SHA256 and exact equality to server originals passed, independently verified again locally. No weights, setup pickles, dependencies or credentials were fetched or archived. Originals remain on the server. The prior pilot/incomplete-campaign archives and their hashes are unchanged.

`campaign-records-members.json` records every exact member hash/size. `correctness.json` and `module-correctness.json` bind the proofs to the measured candidate; `acceptance.json` contains all 24 unrounded comparisons. Independent remote and local execution of the final checker returned the identical passing result. Restore only absent files before checking:

```sh
cd benchmarks/packed-gpu/evidence/staging-functional
shasum -a 256 -c campaign-records.sha256
unzip -n campaign-records.zip
cd ../../../..
python scripts/check_packed_gpu_results.py benchmarks/packed-gpu/evidence/staging-functional \
  --baseline-cpu benchmarks/packed-gpu/evidence/baseline-cpu.json
```

PR #18 is ready for review once this evidence is committed. **Do not merge or update Overleaf automatically.** These measurements establish acceptance on the listed pinned models/GPU, not a universal guarantee on other hardware or models.

## Historical incomplete attempt (preserved)

The earlier staging-only run below preceded the approved functional-consumer correction. It is retained as failure/measurement history and does not count toward the two completed campaigns above.

### Approved functional-consumer correction

The candidate now hooks `MultiheadAttention.out_proj` decoding on its consuming
attention module. The benchmark applies that same adapter to original-main
NeuralZip and the official DFloat11 integration, preserving both original
source captures. The adapter identity is recorded in every new control and
required by the acceptance checker. No original measurement is replaced.

The separate `staging-functional` CUDA and module proofs pass, including a
packed functional projection checked at use with poisoned shared storage and
an existing user hook. CPU archives match the original baseline byte-for-byte.
The first small regression fixture did not select packing; that failed
untimed check is preserved remotely. Enlarging the fixture exercises the
unchanged setup's packed route. Production code and setup rules were not
changed for that test adjustment. Both fresh full campaigns passed, as reported above.

### First incomplete campaign

Physical GPU 0, NVIDIA RTX A5500. Baseline decoder: main
`453c06ea019b8d1d2566de0b4a7396b34807425c`. Candidate: staging runtime from
`f9f052d1322d15d76d5bbb4596d37e553643fcf9`. The pinned official DFloat11 codec
and kernel are unchanged. Batch one, eager attention, no cache/offload,
20 warmups and five synchronized 50-forward trials in fresh processes.
Export, initialization, compilation, validation and decoder-only trials are
outside full-forward timing. All times below are medians in milliseconds.

| Model | DFloat11 | Own baseline → candidate | Transfer baseline → candidate | Own / DFloat11 | Transfer / DFloat11 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 43.658 | 44.989 → 42.091 | 45.057 → 43.036 | 0.9641× | 0.9858× |
| Gemma3-1B | 19.597 | 29.224 → 18.584 | 29.252 → 18.929 | 0.9483× | 0.9659× |
| DINOv3 ConvNeXt-B | 3.515 | 5.209 → 3.618 | 5.132 → 3.502 | 1.0294× | 0.9965× |
| DINOv3 ViT-B | 3.250 | 4.462 → 3.119 | 4.471 → 3.113 | 0.9597× | 0.9578× |
| SigLIP2-B | Not valid | Not measured | Not measured | Pending | Pending |
| Parakeet TDT-0.6B-v3 | 12.730 | 20.149 → 11.996 | 20.308 → 12.102 | 0.9423× | 0.9507× |

All ten completed comparisons meet the 1.10× latency limit. The largest measured
ratio is 1.0294× (ConvNeXt own). Every completed cohort passes bitwise
weights/outputs, input/setup immutability, CPU/native source identity and
byte-identical GPU payload/restart metadata and selected/compressed coverage.

#### Allocation

Candidate live and peak allocation exactly equal their respective baseline in
all ten completed comparisons, and are below BF16. No persistent GPU metadata
was added. MiB are rounded here; the original JSON records contain exact bytes.

| Model / method | Baseline live → candidate live MiB | Baseline peak → candidate peak MiB | BF16 live / peak MiB | Memory / bitwise |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-0.8B / own | 1328.006 → 1328.006 | 1388.881 → 1388.881 | 1664.215 / 1725.090 | Pass / pass |
| Qwen3.5-0.8B / transfer | 1333.678 → 1333.678 | 1394.553 → 1394.553 | 1664.215 / 1725.090 | Pass / pass |
| Gemma3-1B / own | 1554.856 → 1554.856 | 1619.137 → 1619.137 | 1965.660 / 2029.941 | Pass / pass |
| Gemma3-1B / transfer | 1573.679 → 1573.679 | 1637.960 → 1637.960 | 1965.660 / 2029.941 | Pass / pass |
| DINOv3 ConvNeXt-B / own | 142.162 → 142.162 | 149.052 → 149.052 | 176.438 / 185.078 | Pass / pass |
| DINOv3 ConvNeXt-B / transfer | 143.718 → 143.718 | 150.609 → 150.609 | 176.438 / 185.078 | Pass / pass |
| DINOv3 ViT-B / own | 137.139 → 137.139 | 141.645 → 141.645 | 179.799 / 184.877 | Pass / pass |
| DINOv3 ViT-B / transfer | 138.630 → 138.630 | 143.136 → 143.136 | 179.799 / 184.877 | Pass / pass |
| Parakeet TDT-0.6B-v3 / own | 931.244 → 931.244 | 947.245 → 947.245 | 1205.285 / 1221.286 | Pass / pass |
| Parakeet TDT-0.6B-v3 / transfer | 938.216 → 938.216 | 954.217 → 954.217 | 1205.285 / 1221.286 | Pass / pass |

SigLIP's BF16 control completed, but its DFloat11 output failed exact validation
before timing. No DFloat11 or NeuralZip latency/memory result is fabricated for
that model. The second campaign was not started.

### SigLIP functional-hook blocker

`vision_model.head.attention.out_proj.weight` is consumed directly by
`torch.nn.MultiheadAttention`; the child Linear's `forward` is never called.
The existing control integration installs its decode hook on that child.
Although the weights validate immediately after explicit decoding, the shared
workspace is overwritten before the attention layer uses that projection.
The unchanged NeuralZip main baseline has the same pre-existing issue.

An **untimed, isolated diagnostic** checked all 149 selected weights at use:

| Control | Original hook | Same hook moved to consuming attention layer |
| --- | --- | --- |
| NeuralZip main, own | One incorrect projection; image/logit outputs differ | 149/149 exact weights; all outputs bitwise BF16 |
| Pinned official DFloat11 | One incorrect projection; image/logit outputs differ | 149/149 exact weights; all outputs bitwise BF16 |

This diagnostic changed only hook placement on temporary model copies. No
library source, codec, kernel, codebook, payload, arithmetic, frozen setup or
published result was changed. The general correction would use module type,
not model-name-specific shortcuts.

Martin approved applying the same general correction to the **baseline control**,
DFloat11 adapter and candidate, then running two fresh full campaigns. The baseline source captures,
first-campaign measurements and failure history remain preserved. Do not treat
these diagnostics or the three-model pilots as acceptance campaigns, mark the
PR ready, merge it or edit Overleaf.

### Verification and preserved raw evidence

CUDA correctness covers FP16/BF16/FP32, raw/singleton/packed routes, odd tails,
segment boundaries, exact 12/32/33/56-bit codes, staged and direct-write paths,
and compressed Hugging Face save/reload. Supported-module fixtures pass for
Linear, Conv2d and Transformers Conv1D. CPU archives match main byte-for-byte
for all three dtypes. Lint, formatting and the acceptance checker's negative
self-tests pass. The complete acceptance checker deliberately returns nonzero
for the missing SigLIP cohort and all second-campaign cohorts.

`evidence/staging/first-campaign-incomplete.zip` preserves the 31 original
first-campaign JSONs, the original failure log, both untimed hook diagnostic
logs and their exact diagnostic source. All ZIP CRCs, member byte equality
and remote SHA256 matches were verified. The 18 pilot records remain in the
separate, unchanged `pilot-records.zip`. Neither archive includes weights,
setups, dependencies or credentials.

Restore the first-campaign records without overwriting local files:

```sh
cd benchmarks/packed-gpu/evidence/staging
shasum -a 256 -c first-campaign-incomplete.sha256
unzip -n first-campaign-incomplete.zip
cd ../../../..
python scripts/check_packed_gpu_results.py benchmarks/packed-gpu/evidence/staging \
  --baseline-cpu benchmarks/packed-gpu/evidence/baseline-cpu.json
```

The final command is expected to fail closed until two complete compliant
campaigns are present. Its current result is saved in
`evidence/staging/incomplete-acceptance.json`.
