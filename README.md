# NeuralZip

**Lossless compression for neural-network weights with reusable setup.**

NeuralZip is a Python library for compressing floating-point model weights. It
is designed for workloads that encode related tensors repeatedly, such as
training checkpoints, federated updates, model families, and Mixture-of-Experts
models.

NeuralZip learns a setup from one set of weights: it groups tensors with similar
exponent distributions and builds shared Huffman codebooks for them. The setup
can then be reused for later checkpoints or transferred to a compatible model,
keeping clustering and codebook construction outside the hot encoding path.

- Bitwise-exact round trips for `float16`, `bfloat16`, and `float32` tensors.
- A simple API for compression, `.nzip` archives, and restoration.
- Reusable setups for repeated compression and compatible family members.
- Direct support for PyTorch modules, state dictionaries, and Hugging Face
  models.
- Exact byte accounting and separate setup, encoding, and decoding timings.
- Included local and official ZipNN baselines for comparison.

## Installation

NeuralZip is currently installed from source:

```bash
git clone https://github.com/MartinEBravo/nzip.git
cd nzip
python -m pip install -e .
```

Python 3.10 or newer and a native C/C++ compiler are required. Installation
builds the bundled Huff0 extension; its required Zstd sources are included in
the repository.

To reproduce the experiments or use the official ZipNN baseline:

```bash
python -m pip install -e ".[experiments]"
```

## Quick start

```python
import torch

from neuralzip import Compressor

weights = {
    "encoder.weight": torch.randn(1024, 1024, dtype=torch.bfloat16),
    "decoder.weight": torch.randn(1024, 1024, dtype=torch.bfloat16),
}

compressor = Compressor(
    "neuralzip",
    keep_payloads=True,
    verify=True,
    threads=4,
)
compressed = compressor.compress(weights)

print(f"Stored size: {100 * compressed['compression_ratio']:.2f}% of the input")
print(f"Setup: {compressed['setup_seconds']:.3f} s")
print(f"Encode: {compressed['encode_seconds']:.3f} s")

Compressor.save(compressed, "weights.nzip")
restored = compressor.decompress_file("weights.nzip")

for name, tensor in weights.items():
    assert torch.equal(
        restored[name].view(torch.uint8),
        tensor.view(torch.uint8),
    )
```

`keep_payloads=True` is required when the result will be saved or decompressed.
The default, `False`, is useful for measurements that do not need to retain the
encoded bytes. `verify=True` is enabled by default and checks every included
tensor after compression.

The equivalent explicit load flow is:

```python
loaded = Compressor.load("weights.nzip")
restored = Compressor("neuralzip").decompress(loaded)
```

## Compressing a model

NeuralZip accepts a `torch.nn.Module` directly:

```python
compressed = Compressor(
    "neuralzip",
    keep_payloads=True,
).compress(model)
```

It also includes a small Hugging Face loader:

```python
from neuralzip import Compressor, HFModel

hf_model = HFModel("openai-community/gpt2", dtype="bfloat16")
compressed = Compressor(
    "neuralzip",
    keep_payloads=True,
).compress(hf_model)
```

For a module, NeuralZip includes its floating-point named parameters. Passing a
`state_dict()` instead also includes floating-point buffers:

```python
module = hf_model.model  # Or any torch.nn.Module.
state = module.state_dict()
compressor = Compressor("neuralzip", keep_payloads=True)
compressed = compressor.compress(state)
Compressor.save(compressed, "model-weights.nzip")

floating_state = compressor.decompress_file("model-weights.nzip")
state.update(floating_state)
module.load_state_dict(state)
```

An `.nzip` file is a tensor archive, not a serialized model object. It does not
contain the model architecture, configuration, non-floating state, optimizer,
or original Python container. Keep those separately when reconstructing a full
training checkpoint.

## Reusing a setup

`NeuralZipPlan` separates the one-time setup from encoding. This is the
recommended interface for checkpoint trajectories and other repeated workloads.

```python
from neuralzip import NeuralZipPlan

source = dict(source_model.named_parameters())
target = dict(target_model.named_parameters())

# Pay clustering and codebook construction once.
plan = NeuralZipPlan.fit(source, threads=8)
print(f"Setup: {plan.setup_report.setup_seconds:.3f} s")

# The same setup can encode its source immediately.
source_batch = plan.encode(source)
source_restored = plan.decode(source_batch.payloads)

# Save the reusable setup independently of any compressed checkpoint.
plan.save("family.setup.pkl", source)

# Later, map a compatible checkpoint or family member and encode it directly.
plan = NeuralZipPlan.load("family.setup.pkl")
mapping = plan.map_target(target)
target_batch = plan.encode(target)
target_restored = plan.decode(target_batch.payloads)

assert mapping["matched_tensors"] == mapping["target_tensors"]
```

Transfer is deterministic and value-independent: every target tensor must match
a source tensor by name, dtype, and byte stream. NeuralZip does not automatically
search for the nearest architecture; the application chooses the compatible
source setup. Plan files use Python pickle and should only be loaded from trusted
sources.

## Compression methods

| Method | Purpose | `.nzip` save and restore |
| --- | --- | --- |
| `neuralzip` | Main codec with reusable clustering, shared codebooks, and packed exponents | Yes |
| `zipnn-local` | Included ZipNN-compatible local baseline | Yes |
| `zipnn-src` | Adapter to the official `zipnn` package for benchmarking | No |

`zipnn-src` is installed by the `experiments` extra. Its retained payloads are
official per-tensor ZipNN frames and are intentionally outside NeuralZip's
model-level archive API.

## Result fields

`Compressor.compress()` returns a dictionary. The most useful fields are:

| Field | Meaning |
| --- | --- |
| `raw_bytes` | Bytes in the selected floating-point tensors |
| `compressed_bytes` | Exact serialized output size for the selected tensors |
| `compression_ratio` | `compressed_bytes / raw_bytes`; lower is better |
| `setup_seconds` | Clustering and codebook construction |
| `encode_seconds` | Target-dependent encoding work |
| `total_seconds` | `setup_seconds + encode_seconds` |
| `decode_seconds` | Verification decode time when `verify=True` |
| `roundtrip_ok` | Whether verification was bitwise exact |

For `neuralzip` and `zipnn-local`, `compressed_bytes` is the exact `.nzip` size
for the selected tensors. For `zipnn-src`, it is the size of the official
per-tensor frames. Verification time is reported separately and is not included
in `total_seconds`.

## Common options

```python
compressor = Compressor(
    "neuralzip",
    verify=True,
    keep_payloads=False,
    threads=0,
    chunk_size=256 << 10,
    compression_threshold=0.95,
    cluster_js_threshold=0.05,
)
```

- `verify` performs a bitwise round-trip check.
- `keep_payloads` retains encoded tensors for saving or later decompression.
- `threads` controls native encoding workers.
- `chunk_size` sets the source bytes processed per chunk.
- `compression_threshold` controls raw fallback for weakly compressible chunks.
- `cluster_js_threshold` controls exponent-histogram clustering.

Advanced controls for sampling, packed exponent n-grams, codebook sharing, and
time-aware mode selection are also exposed by `Compressor` and
`NeuralZipPlan.fit()`.

## How it works

1. NeuralZip rearranges floating-point bytes into dtype-aware streams, exposing
   the exponent field that carries most of the compressible structure.
2. It groups tensors using Jensen-Shannon distance between exponent histograms.
3. For each group and byte stream, it selects among raw storage, local Huff0,
   shared Huff0 codebooks, and packed exponent n-grams.
4. A native batched encoder applies the selected modes and falls back to raw
   storage whenever compression is not worthwhile.

The setup cost is visible and reported. NeuralZip is most useful when that cost
can be amortized across repeated checkpoints or compatible models; for an
unrelated one-shot model, compare `total_seconds`, not only `encode_seconds`.

## Reproducing the experiments

Install the experiment dependencies and inspect the available runner options:

```bash
python -m pip install -e ".[experiments]"

python examples/run_fixed_model_suite.py --help
python examples/run_setup_transfer_suite.py --help
python examples/run_thesis_model_suite.py --help
python examples/run_clustering_ablation.py --help
python examples/run_checkpoint_vision.py --help
python examples/run_federated_vision.py --help
```

The fixed-model suite uses the pinned revisions and fingerprints in
`examples/fixed_model_manifest.json`. Generated checkpoints, logs, tables, and
figures are written below `results/` and are intentionally not versioned.

Some deferred-compressor experiments require external MM-RePair or TurboRC
installations; they are not dependencies of the NeuralZip library.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The public package surface is intentionally small:

```python
from neuralzip import Compressor, HFModel, NeuralZipPlan
```
