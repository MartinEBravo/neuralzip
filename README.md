# NeuralZip

NeuralZip is a lossless compressor for floating-point model weights. It learns
a reusable setup—tensor groups and Huffman codebooks—from one model, then uses
that setup to encode later checkpoints or compatible members of the same model
family without repeating the statistical analysis.

It is designed for repeated workloads such as training checkpoints, federated
updates, model families, and Mixture-of-Experts models.

- Bitwise-exact round trips for FP16, BF16, and FP32 tensors.
- Reusable and transferable setups.
- PyTorch modules, state dictionaries, and Hugging Face models.
- Separate setup, mapping, encoding, and decoding timings.
- Directly loadable BF16 inference checkpoints for NVIDIA GPUs.

## Installation

```bash
git clone https://github.com/MartinEBravo/nzip.git
cd nzip
python -m pip install -e .
```

Python 3.10 or newer and a C/C++ compiler are required when installing from
source. The native extension compiles the Huff0 slice from a pinned official
[Zstd revision](https://github.com/facebook/zstd/commit/5233c58e6ca0b1c4c6b353ad79649191ed195bdc).
Source builds download that revision, verify the archive's SHA-256, and extract
only the required files into the temporary build directory. Built wheels
contain the compiled code and do not download or dynamically link Zstd at
runtime.

For an offline source build, download the pinned archive once and point the
build at it; the same SHA-256 check still applies:

```bash
NEURALZIP_ZSTD_ARCHIVE=/path/to/zstd-5233c58.tar.gz \
  python -m pip install .
```

Install the GPU runtime when compressed inference is needed:

```bash
python -m pip install -e ".[gpu]"
```

## Compressed inference in one call

`NeuralZip.from_pretrained()` loads an ordinary Hugging Face model in BF16,
obtains a sharing-only setup, compresses its eligible weights, and returns the
native Transformers model ready for inference:

```python
from transformers import AutoTokenizer, pipeline
from neuralzip import NeuralZip

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = NeuralZip.from_pretrained(model_id, device="cuda:0")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=model.neuralzip_runtime.device,
)
print(generator("Machine learning is", max_new_tokens=20, do_sample=False))
```

The default loader is `AutoModelForCausalLM`. Pass another Hugging Face auto
class through `model_class=` for vision, audio, classification, or seq2seq
architectures.

With `setup=None`, NeuralZip fits the source model automatically. Pass an
existing setup object or trusted setup file to skip fitting and transfer its
codebooks:

```python
model = NeuralZip.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    setup="qwen-family.setup.pkl",
    device="cuda:0",
)
```

An explicit incompatible setup fails instead of silently fitting another one.
Automatic fitting is intentionally not cached: persist the compressed model if
future processes should avoid loading and recompressing the original BF16
checkpoint.

For larger models, `device_map="auto"`, explicit Accelerate maps,
`max_memory`, and CPU payload offload are available:

```python
model = NeuralZip.from_pretrained(
    model_id,
    device_map="auto",
    max_memory={0: "22GiB", 1: "22GiB", "cpu": "64GiB"},
    cpu_offload=True,
)
```

## Encode and decode manually

Use the explicit setup API when setup, encoding, and decoding must be
controlled or measured separately. `NeuralZip` is a stateless namespace: the
setup is the only reusable compression state and is always passed explicitly.

```python
import torch
from transformers import AutoModelForCausalLM
from neuralzip import NeuralZip

model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2",
    dtype=torch.bfloat16,
)

setup = NeuralZip.fit(model, threads=4)
encoded, stats = NeuralZip.encode(
    model,
    setup=setup,
    threads=4,
    return_stats=True,
)
restored, decode_stats = NeuralZip.decode(
    encoded,
    threads=4,
    return_stats=True,
)

print(f"Ratio: {encoded.compression_ratio:.3f}")
print(f"Setup: {setup.setup_report.setup_seconds:.3f} s")
print(f"Encode: {stats.encode_seconds:.3f} s")
print(f"Decode: {decode_stats.decode_seconds:.3f} s")

for original, recovered in zip(model.parameters(), restored.parameters()):
    assert torch.equal(
        original.detach().cpu().view(torch.uint8),
        recovered.detach().cpu().view(torch.uint8),
    )
```

`fit()` returns an immutable setup; it does not mutate global or object state.
It also accepts a Hugging Face model identifier when only the reusable setup
is needed: `setup = NeuralZip.fit("openai-community/gpt2")`.
The default is sharing-only so the setup also works with GPU inference. For a
storage-only setup with packed exponent n-grams, pass
`ngram_values=(2, 3, 4)` explicitly.
`encode()` returns an immutable, weight-free encoded value containing its
payloads, codebooks, reconstruction metadata, byte counts, and timing. It does
not retain the source model or its tensor objects. `decode()` is self-contained:
the encoded value carries the codebooks and metadata needed for reconstruction,
so the original setup is not required.

### Reuse and transfer a setup

```python
setup = NeuralZip.fit(reference_model, threads=8)
NeuralZip.save_setup(setup, "family.setup.pkl")

# Same process.
encoded_checkpoint = NeuralZip.encode(
    next_checkpoint,
    setup=setup,
    threads=8,
)

# Another process or a compatible family member.
setup = NeuralZip.load_setup("family.setup.pkl")
mapping = NeuralZip.map_target(related_model, setup=setup)
encoded_related = NeuralZip.encode(
    related_model,
    setup=setup,
    threads=8,
)

assert mapping["matched_tensors"] == mapping["target_tensors"]
```

Mapping is deterministic and does not inspect target values. Streams match by
tensor name, dtype, and stream index, with a small set of explicit aliases for
equivalent structures. NeuralZip does not search for a related architecture;
the caller chooses the source setup. Setup files use pickle and must come from
a trusted source.

### Inputs and reconstruction

`encode()` accepts Hugging Face models, ordinary `torch.nn.Module` objects, and
nested tensor mappings. Hugging Face models carry enough copied configuration
for `decode()` to create a new model. Mappings decode to a flat dictionary.

An arbitrary PyTorch module has no universal safe constructor, so provide an
empty destination or factory:

```python
setup = NeuralZip.fit(custom_model)
encoded = NeuralZip.encode(custom_model, setup=setup)
restored = NeuralZip.decode(encoded, model_factory=CustomModel)
# Or: NeuralZip.decode(encoded, destination=CustomModel(...))
```

Floating parameters and persistent floating buffers are encoded. Remaining
tensor state is retained verbatim so aliases, gradient flags, and module
training modes can be reconstructed. Hooks, arbitrary Python attributes, and
custom `get_extra_state()` values are outside this contract.

## Persist a compressed inference checkpoint

An attached model can be stored without restoring its BF16 weights:

```python
NeuralZip.save_pretrained(
    model,
    "my-model-neuralzip",
    tokenizer=tokenizer,
    max_shard_size="5GB",
)

model = NeuralZip.load_pretrained(
    "my-model-neuralzip",  # Local directory or Hugging Face Hub repository.
    device_map="auto",
)
```

The directory contains model configuration, generation configuration, a
versioned compressed export, and sharded Safetensors for resident state.
Loading reconstructs the architecture without allocating the omitted BF16
weights. Manifests are canonical JSON and every file is SHA-256 verified; the
checkpoint format never deserializes pickle.

`from_pretrained()` starts from an ordinary Hugging Face checkpoint and
compresses it; `load_pretrained()` starts from this NeuralZip artifact and does
neither fitting nor recompression. The trusted setup pickle is deliberately not
embedded in the safe checkpoint, so save it separately when it will be reused.

Saving an attached model is supported with direct `device=` placement.
Accelerate-managed models should be saved before dispatch or copied from their
existing compressed checkpoint because Accelerate owns their resident hooks.

## How compression works

1. Floating-point bytes are rearranged into dtype-aware streams so the
   compressible exponent field is explicit.
2. Tensors are grouped by Jensen–Shannon distance between exponent histograms.
3. Each group reuses a precomputed Huffman codebook; storage-oriented setups may
   additionally use packed exponent n-grams.
4. The native batched encoder applies those choices and falls back to raw bytes
   when compression is not worthwhile.

Setup cost is explicit. NeuralZip is most useful when it is amortized across
checkpoints or transferred between compatible models. For an unrelated
one-shot model, include setup time in the comparison.

The GPU path uses sharing-only BF16 exponent streams. It decodes the active
block into a reusable workspace immediately before execution, so ordinary
PyTorch layers receive the exact original values. The CUDA decoder is adapted
from the [official DFloat11 implementation](https://github.com/LeanModels/DFloat11)
under Apache-2.0.

## Current GPU constraints

- NVIDIA CUDA device with BF16 support.
- BF16 `nn.Linear` weights; embeddings are optional.
- Inference only: no gradients, concurrent forwards, `.to()` after attachment,
  or `torch.compile`.
- Packed exponent n-grams are supported for storage, not by the current GPU
  decoder.
- The runtime reconstructs each block in a temporary BF16 workspace; it is not
  yet a fused compressed matrix-multiplication kernel.

Restore a directly placed model to ordinary registered BF16 parameters when
needed:

```python
model = model.neuralzip_runtime.restore()
```
