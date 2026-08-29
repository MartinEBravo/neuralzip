# NeuralZip

NeuralZip is a lossless compressor for floating-point model weights. It learns
a reusable setup—tensor groups and Huffman codebooks—from one model, then uses
that setup to encode later checkpoints or compatible models without repeating
the statistical analysis.

It is intended for repeated workloads such as training checkpoints, federated
updates, model families, and Mixture-of-Experts models.

- Bitwise-exact FP16, BF16, and FP32 round trips.
- Reusable and transferable setups.
- PyTorch modules, nested tensor mappings, and Hugging Face Transformers.
- Separate setup, mapping, encoding, and decoding timings.
- Sequential compressed BF16 inference on one NVIDIA GPU.

## Installation

```bash
git clone https://github.com/MartinEBravo/nzip.git
cd nzip
python -m pip install -e .
```

Python 3.10 or newer and a C/C++ compiler are required for a source install.
The build downloads a pinned official
[Zstd revision](https://github.com/facebook/zstd/commit/5233c58e6ca0b1c4c6b353ad79649191ed195bdc),
verifies its SHA-256, and compiles only the Huff0 sources used by NeuralZip.
Built wheels contain the native extension and do not download Zstd at runtime.

For an offline source build, provide the same pinned archive explicitly:

```bash
NEURALZIP_ZSTD_ARCHIVE=/path/to/zstd-5233c58.tar.gz \
  python -m pip install .
```

Install the optional CUDA runtime for compressed inference:

```bash
python -m pip install -e ".[gpu]"
```

## Compressed inference

`NeuralZip.from_pretrained()` loads an ordinary local or Hugging Face
Transformers checkpoint in BF16, fits a sharing-only setup, compresses eligible
weights, and returns the native model ready for inference on one GPU.

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

The default loader is `AutoModelForCausalLM`. Use `model_class=` when another
Transformers auto class is needed. Pass `include_embeddings=True` to include
eligible block-local embeddings.

When `setup=None`, NeuralZip fits the source model automatically. An existing
setup object or trusted setup file skips fitting and reuses its codebooks:

```python
model = NeuralZip.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    setup="qwen-family.setup.pkl",
    device="cuda:0",
)
```

An incompatible explicit setup fails instead of fitting a replacement. Save a
compressed checkpoint when another process should avoid loading and
recompressing the original BF16 model.

## Fit, encode, and decode

Use the explicit API when each phase must be controlled or measured separately.
`NeuralZip` is stateless: the setup is the reusable compression state and is
passed explicitly to later operations.

```python
import torch
from transformers import AutoModelForCausalLM
from neuralzip import NeuralZip

model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2",
    dtype=torch.bfloat16,
).eval()

setup = NeuralZip.fit(model, threads=4)
encoded, encode_stats = NeuralZip.encode(
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
print(f"Mapping: {encode_stats.mapping_seconds:.3f} s")
print(f"Encode: {encode_stats.encode_seconds:.3f} s")
print(f"Decode: {decode_stats.decode_seconds:.3f} s")

for original, recovered in zip(model.parameters(), restored.parameters(), strict=True):
    assert torch.equal(
        original.detach().cpu().view(torch.uint8),
        recovered.detach().cpu().view(torch.uint8),
    )
```

`fit()` also accepts a model identifier when the loaded model itself is not
needed: `setup = NeuralZip.fit("openai-community/gpt2")`.

The default setup is sharing-only and can be used by both the storage and GPU
paths. For storage-only packed exponent n-grams, pass
`ngram_values=(2, 3, 4)` to `fit()`. `decode()` is self-contained because the
encoded value carries its codebooks and reconstruction metadata; it does not
need the original setup.

### Reuse or transfer a setup

```python
setup = NeuralZip.fit(reference_model, threads=8)
NeuralZip.save_setup(setup, "family.setup.pkl")

# Reuse it for another checkpoint in this process.
encoded_checkpoint = NeuralZip.encode(
    next_checkpoint,
    setup=setup,
    threads=8,
)

# Or load it for a compatible family member.
setup = NeuralZip.load_setup("family.setup.pkl")
mapping = NeuralZip.map_target(related_model, setup=setup)
encoded_related = NeuralZip.encode(
    related_model,
    setup=setup,
    threads=8,
)

assert mapping["matched_tensors"] == mapping["target_tensors"]
```

Mapping reads target metadata, not tensor values. Streams match by tensor name,
dtype, and stream index, with explicit aliases for equivalent structures. The
caller selects the source setup; NeuralZip does not search for a related model.
Setup files use pickle and must come from a trusted source.

### Inputs and reconstruction

`encode()` accepts Transformers models, ordinary `torch.nn.Module` objects, and
nested tensor mappings. A Transformers encode carries enough configuration for
`decode()` to construct a fresh model. Mappings decode to a flat dictionary.

For an arbitrary module, provide an empty destination or a factory:

```python
setup = NeuralZip.fit(custom_model)
encoded = NeuralZip.encode(custom_model, setup=setup)
restored = NeuralZip.decode(encoded, model_factory=CustomModel)
# Or: NeuralZip.decode(encoded, destination=CustomModel(...))
```

Floating parameters and persistent floating buffers are encoded. Remaining
tensor state, aliases, gradient flags, and module training modes are restored.
Hooks and arbitrary Python attributes are outside this contract.

## Save and load a compressed checkpoint

An attached Transformers model can be saved without restoring its compressed
weights:

```python
NeuralZip.save_pretrained(
    model,
    "my-model-neuralzip",
    tokenizer=tokenizer,
)

model = NeuralZip.load_pretrained(
    "my-model-neuralzip",
    device="cuda:0",
)
```

`load_pretrained()` also accepts the repository ID of a NeuralZip checkpoint on
the Hugging Face Hub:

```python
model = NeuralZip.load_pretrained(
    "account/my-model-neuralzip",
    revision="main",
    device="cuda:0",
)
```

The checkpoint contains a small manifest, the Transformers configuration, one
resident Safetensors file, and a primitive-only compressed export loaded with
PyTorch's `weights_only=True` mode. The model files are SHA-256 checked. It is a
simple research format kept straightforward for local inspection.

`from_pretrained()` starts from an ordinary Transformers checkpoint and
compresses it. `load_pretrained()` starts from an already compressed NeuralZip
checkpoint and performs neither fitting nor recompression. The setup is not
embedded, so save it separately when it will be transferred again.

## How it works

1. Floating-point bytes are rearranged into dtype-aware streams that expose the
   compressible exponent field.
2. Tensors are grouped by Jensen–Shannon distance between exponent histograms.
3. Each group reuses a precomputed Huffman codebook. Storage-oriented setups may
   additionally use packed exponent n-grams.
4. The native batched encoder applies those choices and falls back to raw bytes
   when compression is not worthwhile.

Setup cost is explicit. NeuralZip is most useful when it is amortized across
checkpoints or transferred between compatible models. Include setup time when
evaluating an unrelated one-shot model.

For GPU inference, NeuralZip keeps sharing-only BF16 exponent streams and
decodes the active transformer block into a reusable workspace immediately
before execution. Ordinary PyTorch layers therefore receive the exact original
weights. The CUDA decoder is adapted from the
[official DFloat11 implementation](https://github.com/LeanModels/DFloat11)
under Apache-2.0.

## GPU constraints

- One NVIDIA CUDA device with BF16 support.
- Sequential, inference-only execution.
- BF16 `nn.Linear` weights; block-local embeddings are optional.
- No concurrent forwards, training, device changes after attachment, or
  `torch.compile`.
- Packed exponent n-grams are storage-only; GPU inference uses sharing-only
  codebooks.
- Weights are decoded into a temporary workspace; this is not a fused compressed
  matrix-multiplication kernel.

Restore ordinary registered BF16 parameters when compressed inference is no
longer needed:

```python
model = model.neuralzip_runtime.restore()
```
