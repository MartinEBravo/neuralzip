# NeuralZip thesis freeze

This directory documents the implementation and measurements frozen under the
annotated Git tag `thesis-neuralzip-v1`.

## Frozen method

The final NeuralZip codec uses:

1. ZipNN-compatible dtype-aware bit and byte grouping.
2. Jensen-Shannon clustering over tensor exponent histograms.
3. Shared canonical Huffman codebooks per cluster and byte group.
4. Time-budgeted packed exponent n-grams (`n` in `{2, 3, 4}`) encoded with
   Huff0, a Top-K vocabulary, escapes, and a raw tail.
5. A raw fallback for chunks that do not meet the compression threshold.

The final public entrypoint is `Compressor("neuralzip")`. ZipNN comparisons use
`Compressor("zipnn")`; both paths share the same tensor traversal, byte layout,
chunk size, archive accounting, and timing implementation wherever their
algorithms perform the same operation.

## Timing policy

- `clustering_seconds`: Jensen-Shannon clustering only.
- `codebook_setup_seconds`: codebook, vocabulary, and mode selection.
- `encode_seconds`: tensor encoding with setup already available.
- `compression_seconds`: codebook setup plus encoding.
- `total_compression_seconds`: clustering, codebook setup, and encoding.

The transfer and exponent-only benchmarks explicitly report setup separately.
Their hot measurements do not rebuild codebooks or packed vocabularies. Packed
tokenization and entropy encoding of the target bytes remain in the hot path,
because they are required to produce the compressed representation.

## Final experiment artifacts

The exact scripts and JSON outputs are tracked under
`examples/outputs/resnet18_comparison/` despite the general output ignore rule.

| Experiment | Script | Output |
|---|---|---|
| ResNet-18 same-model setup | `benchmark_resnet18_packed_same_model.py` | `resnet18_packed_same_model_20260712.json` |
| ResNet-18/34 transfer, A1 and torchvision | `benchmark_resnet_packed_bidirectional.py` | `resnet_packed_bidirectional_20260712.json`, `resnet_packed_bidirectional_tv_20260712.json` |
| BERT uncased bidirectional transfer | `benchmark_bert_packed_bidirectional.py` | `bert_packed_bidirectional_20260712.json` |
| BERT Large uncased to Base cased | `benchmark_bert_packed_transfer.py` | `bert_packed_transfer_20260712.json` |
| VGG16-BN to VGG11-BN | `benchmark_vgg_packed_transfer.py` | `vgg_packed_transfer_20260712.json` |
| Exponent-only timing and component ablation | `benchmark_resnet18_exponent_time.py` | `resnet18_exponent_time_20260713.json` |
| Shared-codebook transfer diagnostic | `benchmark_cross_model_codebooks.py` | `cross_model_codebooks_20260712.json` |
| Packed implementation regression check | `diagnose_packed_regression.py` | `packed_regression_diagnosis_20260712.json` |

All final roundtrip checks in these artifacts passed bitwise.

## Reproduction

From the repository root, after installing the experiment dependencies:

```bash
python -m pip install -e '.[experiments,test]'
python -m pytest -q

python examples/outputs/resnet18_comparison/benchmark_resnet_packed_bidirectional.py tv
python examples/outputs/resnet18_comparison/benchmark_resnet_packed_bidirectional.py a1
python examples/outputs/resnet18_comparison/benchmark_bert_packed_bidirectional.py
python examples/outputs/resnet18_comparison/benchmark_bert_packed_transfer.py
python examples/outputs/resnet18_comparison/benchmark_vgg_packed_transfer.py
python examples/outputs/resnet18_comparison/benchmark_resnet18_packed_same_model.py
python examples/outputs/resnet18_comparison/benchmark_resnet18_exponent_time.py
```

The measured runs used 32 PyTorch CPU threads. Each benchmark contains its own
warm-up and repetition policy; do not compare a setup-inclusive time with a hot
encoding time.

## Godel environment

- OS: Debian 12, Linux `6.12.85+deb12-amd64`, x86-64.
- CPU: 2 x AMD EPYC 7343, 32 physical cores, 64 logical CPUs.
- Python: `3.11.2`.
- PyTorch: `2.12.1+cu130`.
- NumPy: `2.4.4`.
- timm: `1.0.28`.
- Transformers: `5.12.1`.

## Model revisions

| Model | Hugging Face revision |
|---|---|
| `timm/resnet18.tv_in1k` | `bbd144b3e5565108aad885f145491d11bc6ce807` |
| `timm/resnet34.tv_in1k` | `1b7b21cca82ff974d341713f777bf740c2db38c4` |
| `timm/resnet18.a1_in1k` | `491b427b45c94c7fb0e78b5474cc919aff584bbf` |
| `timm/resnet34.a1_in1k` | `0e8757122c4c09b15fb9300c0fd09b3fe89e28bf` |
| `google-bert/bert-base-uncased` | `86b5e0934494bd15c9632b12f734a8a67f723594` |
| `google-bert/bert-large-uncased` | `6da4b6a26a1877e173fca3225479512db81a5e5b` |
| `google-bert/bert-base-cased` | `cd5ef92a9fb2f889e972770a36d4ed042daf221e` |
| `timm/vgg16_bn.tv_in1k` | `f2d1d9661fb46b2cc78f53774d4d62af156060d6` |
| `timm/vgg11_bn.tv_in1k` | `d1b4a6d788dbf60b7dd034952e34dfda1d52404d` |

The benchmark scripts identify models by repository name. To reproduce this
snapshot exactly, download the revisions above or use a cache containing those
snapshot hashes.
