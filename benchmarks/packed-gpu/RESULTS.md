# Packed GPU decoding: incomplete acceptance

PR #18 remains draft. Shared-memory exponent-byte staging is implemented, but
the required two complete, independent six-model campaigns have **not** passed.
This report is a first-campaign result, not final acceptance.

## First campaign

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

### Allocation

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

## SigLIP functional-hook blocker

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

## Verification and preserved raw evidence

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
