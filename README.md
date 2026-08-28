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
- Persistent, directly loadable BF16 inference checkpoints for NVIDIA GPUs,
  with generation, CPU offload, and automatic multi-GPU placement.
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
from transformers import AutoModelForCausalLM

from neuralzip import Compressor

model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2",
    dtype=torch.bfloat16,
)

neuralzip = Compressor(
    "neuralzip",
    threads=4,
)
setup = neuralzip.fit(model)
encoded = neuralzip.encode(model, setup=setup)
restored = neuralzip.decode(encoded)

print(f"Encoded tensor size: {100 * encoded.compression_ratio:.2f}% of the input")
print(f"Setup: {setup.setup_report.setup_seconds:.3f} s")
print(f"Encode: {encoded.encode_seconds:.3f} s")

for original, recovered in zip(model.parameters(), restored.parameters()):
    assert torch.equal(
        original.detach().cpu().view(torch.uint8),
        recovered.detach().cpu().view(torch.uint8),
    )
```

`fit()` returns an opaque, data-only setup. It contains the clustering and
Huffman codebooks but has no encoding methods of its own. `Compressor` fits,
maps, encodes, and decodes; the resulting `EncodedModel` can additionally
prepare compressed GPU inference. The encoded value does not retain the source
model or any source tensor objects.
Its byte ratio covers encoded tensor streams and verbatim auxiliary tensor data;
it is not a serialized full-model file size. Use the `.nzip` archive or GPU
checkpoint APIs below when exact on-disk accounting is required.

Official ZipNN uses the same model-level interface, without a setup:

```python
zipnn = Compressor("zipnn")
encoded_zipnn = zipnn.encode(model)
restored_zipnn = zipnn.decode(encoded_zipnn)
```

## Inputs and reconstruction

`encode()` accepts Hugging Face models, ordinary `torch.nn.Module` instances,
and tensor mappings. A Hugging Face result stores its class and copied
configuration, so `decode()` can construct a new model automatically. A tensor
mapping decodes to a new flat `dict`.

There is no safe generic constructor for an arbitrary PyTorch module. Supply an
empty destination or factory when decoding one:

```python
encoded = neuralzip.encode(custom_model, setup=setup)
restored = neuralzip.decode(encoded, model_factory=CustomModel)
# Or: neuralzip.decode(encoded, destination=CustomModel(...))
```

For a module, NeuralZip compresses floating-point named parameters and
persistent floating-point buffers. Remaining state—including integer BatchNorm
counters and non-persistent runtime buffers—is stored verbatim so the decoded
module preserves its tensor state, aliases, gradient flags, and per-module
training modes. Arbitrary Python attributes, hooks, and custom `get_extra_state()`
values are outside this model-level contract. Passing a tensor mapping or
`state_dict()` encodes its floating tensor leaves and decodes them to a flat
`dict`.

## Tensor archives

The older `compress()` reporting interface remains available for exact
benchmark fields and `.nzip` tensor archives:

```python
archive_compressor = Compressor(
    "neuralzip", setup=setup, verify=True,
    keep_payloads=True,
)
parameters = dict(model.named_parameters())
compressed = archive_compressor.compress(parameters)
Compressor.save(compressed, "model-weights.nzip")

floating_state = archive_compressor.decompress_file("model-weights.nzip")
state = model.state_dict()
state.update(floating_state)
model.load_state_dict(state)
```

`keep_payloads=True` is required for saving or later decompressing a reporting
result. `verify=True` checks every included tensor bitwise.

An `.nzip` file is a tensor archive, not a serialized model object. It does not
contain the model architecture, configuration, non-floating state, optimizer,
or original Python container. Keep those separately when reconstructing a full
training checkpoint.

## Reusing a setup

Call `Compressor.fit()` when clustering and codebook construction should happen
once and be reused. This is the recommended flow for checkpoint trajectories
and other repeated workloads.

```python
from neuralzip import Compressor

source = dict(source_model.named_parameters())
target = dict(target_model.named_parameters())

# Pay clustering and codebook construction once.
compressor = Compressor("neuralzip", threads=8)
setup = compressor.fit(source)
print(f"Setup: {setup.setup_report.setup_seconds:.3f} s")

# The same setup can encode its source immediately.
source_batch = compressor.encode(source, setup=setup)
source_restored = compressor.decode(source_batch)

# Save the reusable setup independently of any compressed checkpoint.
compressor.save_setup("family.setup.pkl", setup=setup)

# Later, map a compatible checkpoint or family member and encode it directly.
compressor = Compressor("neuralzip", setup="family.setup.pkl", threads=8)
mapping = compressor.map_target(target)
target_batch = compressor.encode(target)
target_restored = compressor.decode(target_batch)

assert mapping["matched_tensors"] == mapping["target_tensors"]
```

`encode()` returns a model-level encoded value with byte accounting and separate
mapping and encoding times. Pass `return_stats=True` to receive the same values
as a compact statistics object alongside the encoded model. For the full legacy
experiment report and an archive-ready dictionary, call `compress()` on the
fitted or loaded compressor:

```python
compressor = Compressor(
    "neuralzip",
    setup="family.setup.pkl",
    keep_payloads=True,
)
compressed = compressor.compress(target)
Compressor.save(compressed, "target.nzip")
```

Transfer is deterministic and value-independent: target streams normally match
by tensor name, dtype, and stream index; shapes and values may differ. Explicit
structural aliases can reuse a source codebook when related architectures expose
an equivalent tensor under a different representation—for example, an untied
`lm_head.weight` can reuse the source input-embedding modes. Mapping reports
exact and fallback assignments separately. NeuralZip does not automatically
search for the nearest architecture; the application chooses the compatible
source setup. Setup files use Python pickle and must only be loaded from trusted
sources; never pass an untrusted setup file through `setup=`.

## GPU inference

NeuralZip includes a lossless BF16 inference runtime for NVIDIA GPUs. It keeps
shared-Huffman exponent streams and verbatim sign/mantissa bytes compressed,
decodes each block immediately before execution, and reuses one BF16 workspace
per device. PyTorch still receives the original BF16 values for its linear and
embedding operations.

The runtime requires CUDA 12 and an NVIDIA GPU with BF16 support. Install its
optional dependency with:

```bash
python -m pip install -e ".[gpu]"
```

### Save and load a compressed model

Fit a sharing-only setup once, then save a complete checkpoint. The resulting
directory contains the model configuration, generation configuration, a
versioned NeuralZip export, and sharded Safetensors for the state that remains
resident. Original uncompressed copies of the encoded weights are omitted from
the resident shards, so loading never downloads or materializes them.

```python
import torch

from neuralzip import Compressor, NeuralZipModel

model = model.to(dtype=torch.bfloat16).eval()
compressor = Compressor(
    "neuralzip",
    ngram_values=(),
    packed_source_budget_ratio=0.0,
    packed_time_budget_ratio=0.0,
)
compressor.fit(model)

# Persist configuration + resident shards + compressed BF16 weights.
NeuralZipModel.save_pretrained(
    model,
    compressor,
    "my-model-neuralzip",
    tokenizer=tokenizer,  # Optional.
    max_shard_size="5GB",
)

# This reconstructs the architecture without first allocating all original
# weights, streams resident shards to their destinations, and attaches the
# compressed runtime only after placement succeeds.
model = NeuralZipModel.from_pretrained(
    "my-model-neuralzip",  # A local directory or Hugging Face Hub repo.
    device_map="auto",
)

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=64)
```

`from_pretrained()` recognizes the saved Transformers AutoModel class instead
of guessing from a class-name suffix. It supports causal and seq2seq language
models, masked language models, speech, vision, audio, classification, and
question-answering AutoModel families. Generation configuration and KV cache
behavior are preserved.

Checkpoints use canonical JSON, SHA-256 verification, and Safetensors only;
they never deserialize pickle. Resident state is always indexed and may be
split into multiple shards. Exact tied-weight aliases are stored once and
restored when loading. Automatic loading does not execute checkpoint-supplied
Python by default; `trust_remote_code=True` is an explicit trusted-code opt-in.

### Placement and offload

Automatic placement sizes encoded weights by their compressed payload instead
of their original BF16 allocation, then reserves memory for the shared
workspace, metadata, codebooks, launch descriptors, and one transient block.
You can also set explicit per-module placement through the standard Accelerate
map:

```python
model = NeuralZipModel.from_pretrained(
    "my-model-neuralzip",
    device_map="auto",
    max_memory={0: "22GiB", 1: "22GiB", "cpu": "64GiB"},
    cpu_offload=True,
)
```

`cpu_offload=True` keeps compressed exponent and sign/mantissa payloads on
pinned host memory and transfers only the current block's payload. If the
device map places resident blocks on CPU, their NeuralZip payloads are
offloaded automatically. To keep the complete resident model on CPU and
execute one block at a time on a GPU:

```python
model = NeuralZipModel.from_pretrained(
    "my-model-neuralzip",
    device_map={"": "cpu"},
    execution_device="cuda:0",
    cpu_offload=True,
)
```

### Diffusion components and custom architectures

Install `.[gpu,diffusion]` to save and load Diffusers model components such as
transformers used by Flux-like pipelines. NeuralZip stores the Diffusers class
and config, then the component can be injected into its ordinary pipeline:

```python
NeuralZipModel.save_pretrained(
    transformer,
    compressor,
    "flux-transformer-nzip",
)

transformer = NeuralZipModel.from_pretrained(
    "flux-transformer-nzip",
    device_map="auto",
)
pipeline = FluxPipeline.from_pretrained(base_model_id, transformer=transformer)
```

For other `torch.nn.Module` architectures, use a `RegexGPUAdapter` when block
discovery is not automatic. A trusted `model=` instance or `model_factory=` can
reconstruct custom classes without embedding executable objects in the
checkpoint.

### In-memory attachment

Compression and execution preparation are separate operations. `fit()` learns
the reusable setup, `encode()` creates the compressed model, and the encoded
value converts its payloads to the CUDA runtime layout before attachment:

```python
setup = compressor.fit(model)
encoded = compressor.encode(model, setup=setup)
model = encoded.prepare_inference(model.to("cuda:0"))

with torch.inference_mode():
    output = model(**inputs)
```

The lower-level flow remains available to benchmarks that need to measure the
GPU export and attachment independently:

```python
from neuralzip import NeuralZipRuntime, export_model_for_gpu

exported = export_model_for_gpu(model, compressor)

model = model.to("cuda:0")
NeuralZipRuntime.attach(model, exported)

with torch.inference_mode():
    output = model(**inputs)

runtime = model.neuralzip_runtime
print(runtime.export.compressed_bytes)  # Payload, stream metadata, and LUTs.
print(runtime.export.workspace_bytes)  # Reused temporary BF16 workspace.
print(runtime.export.runtime_candidate_bytes)  # Includes workspace and descriptors.

# Recover an ordinary model with registered BF16 Parameters when needed.
model = runtime.restore()
```

An attached model using direct `device=` placement can be copied without
materializing its compressed weights:

```python
NeuralZipModel.save_pretrained(model, "another-neuralzip-checkpoint")
```

Attachment uploads the compressed buffers, compiles the CUDA kernel, and
preflights every block before removing any parameter. A CuPy, NVRTC, upload, or
launch failure therefore leaves the model intact. Weights whose exponent mode
is `raw`, non-BF16 weights, noncontiguous weights, and aliased weights remain
resident. This transactional order creates a transient attachment peak where
the original weights, compressed buffers, launch descriptors, and workspace
coexist; reserve device headroom beyond `runtime_candidate_bytes` for it.

The runners' `payload_ratio` uses `compressed_bytes / raw_bytes`: it includes
the compressed streams, restart metadata, and shared LUTs, but excludes the
reusable workspace and GPU launch descriptors. `runtime_candidate_bytes`
includes both of those runtime allocations as well as eligible weights that
remain resident.

While attached, compressed weights are intentionally absent from
`named_parameters()` and the ordinary `state_dict()`. Use
`NeuralZipModel.save_pretrained()` for a compressed checkpoint, or call
`runtime.restore()` before ordinary PyTorch serialization. Restore before
calling `.to()`, changing dtype, or mutating module structure.

Accelerate-managed `device_map` and CPU-offload hooks own their resident tensor
placement. Re-saving or restoring registered Parameters while those hooks are
active is rejected; copy the original checkpoint, call `runtime.materialize()`
for independent decoded tensors, or load once with direct `device=` placement.

The runtime is inference-only and sequential within each device. It currently
compresses BF16 `nn.Linear` and optionally `nn.Embedding` weights using
sharing-only exponent modes. Packed exponents, training, concurrent forwards,
and `torch.compile` are not supported. The decoder reconstructs each active
block into a temporary workspace; it is not a fused compressed matmul kernel.

## Compression methods

| Method | Purpose | `.nzip` save and restore |
| --- | --- | --- |
| `neuralzip` | Main codec with reusable clustering, shared codebooks, and packed exponents | Yes |
| `zipnn-local` | Included ZipNN-compatible local baseline | Yes |
| `zipnn` | Adapter to the official `zipnn` package for benchmarking | No |

`zipnn` is installed by the `experiments` extra. Its retained payloads are
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
| `mapping_seconds` | Metadata-only target mapping when a setup is reused |
| `encode_seconds` | Target-dependent encoding work |
| `total_seconds` | Setup or mapping performed by the call, plus encoding |
| `decode_seconds` | Verification decode time when `verify=True` |
| `roundtrip_ok` | Whether verification was bitwise exact |

For `neuralzip` and `zipnn-local`, `compressed_bytes` is the exact `.nzip` size
for the selected tensors. For `zipnn`, it is the size of the official
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
- `threads` controls native encoding and decoding workers.
- `chunk_size` sets the source bytes processed per chunk.
- `compression_threshold` controls raw fallback for weakly compressible chunks.
- `cluster_js_threshold` controls exponent-histogram clustering.

Advanced controls for sampling, packed exponent n-grams, codebook sharing, and
time-aware mode selection are also exposed by `Compressor`.

For latency-sensitive decoding, use the native sharing-only path:

```python
compressor = Compressor(
    "neuralzip",
    ngram_values=(),
    packed_source_budget_ratio=0.0,
    packed_time_budget_ratio=0.0,
)
compressor.fit(tensors)
```

This path decodes all raw and shared-Huff0 chunks through one native job queue.
Packed exponents remain available as an optional storage-oriented mode when a
small additional ratio reduction matters more than decode latency.

## How it works

1. NeuralZip rearranges floating-point bytes into dtype-aware streams, exposing
   the exponent field that carries most of the compressible structure.
2. It groups tensors using Jensen-Shannon distance between exponent histograms.
3. For each group and byte stream, it selects among raw storage, precomputed
   shared Huff0 codebooks, and precomputed packed-exponent codebooks.
4. A native batched encoder applies the selected modes and falls back to raw
   storage whenever compression is not worthwhile.

The setup cost is visible and reported. NeuralZip is most useful when that cost
can be amortized across repeated checkpoints or compatible models; for an
unrelated one-shot model, compare `total_seconds`, not only `encode_seconds`.

## Scope and future work

NeuralZip's stable path targets storage, transfer, and repeated checkpoint
compression. The GPU inference runtime is deliberately narrower: it re-encodes
exponent streams with the selected precomputed Huff0 codebooks into a
continuous GPU bitstream rather than trying to consume the archive's
four-stream frames directly.

The CUDA decoder is adapted from the
[official DFloat11 implementation](https://github.com/LeanModels/DFloat11)
under Apache-2.0, with a collision-free lookup table for all 256 BF16 exponent
symbols and additional bounds checks. The current checkpoint path supports
direct loading, sharded resident state, Transformers generation, Diffusers
components, CPU payload offload, and automatic or explicit multi-GPU placement.

The compressed GPU export is currently one Safetensors file and is retained in
host memory while attached. Future work includes sharded streaming for that
export, a GPU decoder for packed exponent n-grams, prefetching and concurrent
forwards, disk offload, precompiled CUDA wheels, broader end-to-end pipeline
integrations, and fused kernels that consume compressed weights without
reconstructing each active block in a temporary BF16 workspace.

## Reproducing the experiments

Install the experiment dependencies and inspect the available runner options:

```bash
python -m pip install -e ".[experiments]"

python -m examples.experiments.run_fixed_model_suite --help
python -m examples.experiments.run_setup_transfer_suite --help
python -m examples.experiments.run_thesis_model_suite --help
python -m examples.experiments.run_clustering_ablation --help
python -m examples.experiments.run_checkpoint_vision --help
python -m examples.experiments.run_federated_vision --help
python -m examples.experiments.run_dfloat11_comparison --help
```

The fixed-model suite uses the pinned revisions and fingerprints in
`examples/experiments/fixed_model_manifest.json`. Generated checkpoints, logs,
tables, and figures are written below `results/` and are intentionally not
versioned.

Some deferred-compressor experiments require external MM-RePair or TurboRC
installations; they are not dependencies of the NeuralZip library.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The public package surface is intentionally small:

```python
from neuralzip import (
    Compressor,
    NeuralZipModel,
    NeuralZipRuntime,
    export_model_for_gpu,
    load_gpu_pretrained,
    save_gpu_pretrained,
)
```
