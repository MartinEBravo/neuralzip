# NeuralZip

NeuralZip is a lossless model-weight compressor with a transparent Hugging Face
wrapper. Activate it once for the current Python process, then keep using the
ordinary `from_pretrained()` and `save_pretrained()` APIs; the wrapper performs
the encode automatically.

It is a research implementation for sequential compressed floating-point inference on
one NVIDIA GPU. Eligible weights are stored as lossless compressed streams and
reconstructed only when their block or individual layer runs.

- Bitwise-exact weights in their loaded dtype during inference.
- Optional precomputed global codebooks, declared per format in `dtypes.py`.
- Reusable chunk-level Jensen--Shannon clusters and adaptive pair/singleton Huffman.
- Standard Hugging Face directories and Safetensors shards.
- No NeuralZip-specific model loader or checkpoint export command.

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

## Fast path

Call `neuralzip()` once before loading a model. The activation applies to
subsequent compatible Transformers loads and saves in that Python process.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from neuralzip import neuralzip

neuralzip()

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device="cuda:0",
)
print(generator("Machine learning is", max_new_tokens=20, do_sample=False))

stats = model.neuralzip_stats
print(stats.representation)  # "gpu" before save, "storage" after save
print(f"Encode: {stats.encoding_seconds:.3f} s")
print(f"Ratio: {stats.compression_ratio:.3f}")
print("Exponent ratio:", stats.exponent_ratio)
print("Compressed-stream ratio:", stats.stream_ratio)
```

For an ordinary checkpoint, Hugging Face loads the model and NeuralZip then
compresses eligible weights in memory, preserving the dtype chosen by HF.
Without a setup, this path uses any available global codebooks declared in
`dtypes.py`; it does not analyze the model or construct a setup. Missing global
files are skipped. Weights without a compatible codebook remain uncompressed:
if no weights are exported, there is no compressed GPU runtime and the GPU
representation ratio is 1.0. Build and supply a setup to compress those weights.
The wrapper stores setup time, encode time, raw and compressed payload bytes,
and total and exponent-only ratios in `model.neuralzip_stats`. Its
`representation` field distinguishes the in-memory GPU layout from the storage
layout written by `save_pretrained()`. Byte counts use NeuralZip's canonical
binary layout, excluding filesystem and Safetensors/JSON container overhead.
Exponent ratio is `None` when the model's layouts do not expose pure exponent
byte streams; stream ratio describes the lanes actually compressed and is
`None` when none are selected.

## Reusable family setup

Build a family-specific setup once from an ordinary reference model, before
activating NeuralZip:

```python
from transformers import AutoModelForCausalLM
from neuralzip import neuralzip

reference = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    dtype="auto",
)
setup = neuralzip.setup(reference)
setup.save("qwen-family.setup.pkl")
print(setup.setup_seconds)
```

`neuralzip.setup(reference)` delegates to `NeuralZipSetup().build(reference)`:

```python
from neuralzip.setup import NeuralZipSetup

setup = NeuralZipSetup().build(reference)
setup.save("qwen-family.setup.pkl")
setup = NeuralZipSetup.load("qwen-family.setup.pkl")
```

The constructor only configures defaults. `build()` learns the setup, replaces
any previous learned state, and returns the same instance. Saved setups contain
configuration, codebooks, assignments, and timing results, not training samples
or native handles. Rebuild tensor-level setups for this algorithm. Storage format
version 2 and HF checkpoint version 3 replace the old fixed-pair format;
old setups and compressed checkpoints are not supported on this branch.

The setup analyzes floating parameters and persistent floating buffers in the
formats registered in `dtypes.py`. Source-weight chunks default to 256 KiB.
Each chunk joins the nearest same-dtype exponent centroid within JS distance
0.05, or starts a new cluster. Different chunks of one tensor can use different
tables; chunks from different tensors can share them.

For each cluster, up to 8 MiB of stream data are sampled without joining chunk
boundaries. Singleton Huffman is the baseline. A mixed candidate starts with up
to 254 frequent adjacent exponent pairs and greedily parses matched pairs,
otherwise singletons. Pairs whose codes are longer than their two singleton
codes are removed and the candidate is rebuilt until that condition holds.
All singleton bytes remain encodable, with pseudocount one.

During setup, the two encoders are compared on up to four chunks spread across the samples, using
one worker, one warmup and five alternating measured repetitions. Mixed coding
is kept only when its estimated cluster size, including table, vocabulary and
fragment metadata, is smaller. Encoding time is recorded for diagnostics and
does not affect selection. The choice is frozen; encode performs no fitting
or benchmarking. `setup.packing_report` records size estimates, timings and
selected modes. Sample measurements do not guarantee savings for unseen
weights or full-model timings. Their cost is included in setup time.

At encode time, each chunk falls back to raw if its payload plus fragment
record is not below 95% of its uncompressed stream size. There are no escape
streams or raw tails.

Disable packing with `enable_packed_codebooks=False` for sharing-only, or sharing
with `enable_shared_codebooks=False` for one selected table per chunk. Disabling
both stores raw streams. Pair packing applies only to pure exponent lanes;
FP16 currently retains its reversible byte layout and singleton-only coding.
Singleton tables use upstream Zstd Huff0; mixed alphabets use the native
canonical Huffman encoder with four independently delimited streams.
Mixed encoding parses and writes in one pass. Its four stream boundaries may
cross a source pair, but never split or restart its greedy tokenization.

The decoder retains tables for matching codebook contents across calls, not
just matching IDs. It owns copies of the small headers and vocabularies, not
the checkpoint buffer. A fresh decoder still constructs tables from the archive;
cold and repeated decoding should therefore be measured separately.

Matching tensor names, dtypes, and element counts reuse frozen chunk assignments.
For differently sized tensors, the setup also builds a singleton projection by
pooling original source counts per tensor. This projection requires no target
histogram or fitting, but discards chunk distinctions and pair packing. Unmapped
tensors use compatible global codebooks or raw storage.

Select the resulting object or file once in each inference process, then use
Transformers normally:

```python
neuralzip("qwen-family.setup.pkl")  # neuralzip(setup) also works
model = AutoModelForCausalLM.from_pretrained("account/related-qwen-model")
```

For GPU inference, the wrapper uses one compatible shared-Huffman byte stream
per group, preferring a pure exponent stream when the layout provides one.
It consumes the frozen chunk assignments, including adaptive singleton/pair
codebooks, and expands pair symbols during CUDA decoding. Tensors absent
from the setup use an available global codebook or remain uncompressed. Setup
files use pickle and must come from a trusted source.

## Save and load

Use the normal Hugging Face methods. Saving encodes every floating parameter and
persistent floating buffer with the full storage codec. It preserves raw,
singleton-Huffman, and mixed pair/singleton modes in ordinary Safetensors shards, with
the small format marker and statistics in `config.json`.

```python
model.save_pretrained("qwen-neuralzip")
tokenizer.save_pretrained("qwen-neuralzip")
print(model.neuralzip_stats)

loaded_model = AutoModelForCausalLM.from_pretrained("qwen-neuralzip")
print(loaded_model.neuralzip_stats.decoding_seconds)
```

The same call can load a NeuralZip checkpoint from the Hugging Face Hub:

```python
loaded_model = AutoModelForCausalLM.from_pretrained(
    "account/my-model-neuralzip",
    revision="main",
)
```

A compressed checkpoint contains the codebooks and metadata needed to restore
every tensor bit-for-bit. NeuralZip detects the format, decodes the storage
representation, and converts eligible weights to its GPU runtime
layout. This never runs setup or rebuilds storage codebooks. The conversion is
necessary because packed storage streams and the blockwise CUDA decoder use
different layouts. `model.neuralzip_stats` retains the original storage encode
measurements and adds `decoding_seconds` for the restoration itself.
`runtime_encoding_seconds` records the subsequent conversion to the GPU layout.
With no explicit dtype, or `dtype="auto"`, a compressed checkpoint restores
each tensor's stored dtype and bits, including mixed-dtype state. An explicit
`dtype` requests HF's normal conversion and can change the original precision.

Call `neuralzip()` once at the start of every new process before loading either
ordinary or compressed models.

No setup file is needed to restore a saved checkpoint's weights: its required
codebooks are included. Compressing them again for GPU inference uses the active setup or
available globals; otherwise they stay resident and uncompressed.

## Tensor checkpoints

For explicit CPU checkpoint encoding and decoding, use one compressor interface:

```python
from neuralzip.compressor import NeuralZipCompressor

compressor = NeuralZipCompressor(setup)
blob, metadata, sizes, seconds = compressor.encode(reference.named_parameters())
restored = compressor.decode(blob, metadata)  # dictionary of named CPU tensors
```

Pass an iterable of `(name, floating_tensor)` pairs. Encoding uses the native
chunk job queue and returns the byte tensor, JSON-safe metadata,
canonical size counts, and encoding time. The time includes encoding, assembly,
and size accounting, but excludes codebook preparation and target mapping.
Decode reads its codebooks from the checkpoint; a fresh `NeuralZipCompressor()`
can decode the same blob without the original setup. HF saving uses this pair
internally and additionally preserves persistent buffers and shared-weight aliases.

Execution options belong to the compressor, not the setup:

```python
compressor = NeuralZipCompressor(setup, threads=4, direct_batch_bytes=64 << 20)
```

Defaults are `threads=0` (automatic) and `direct_batch_bytes=128 << 20` for
the decode batch budget. Setup uses one `prior_count`
(default `1`) to smooth every Huffman table, including mixed-token tables.
These execution options are no longer accepted by `NeuralZipSetup`, and
`prior_count` replaces the separate `exponent_prior_count` option.

## How it works

1. `neuralzip()` installs the transparent Transformers integration for the
   current process and selects the active setup and CUDA device.
2. An ordinary checkpoint is loaded through Transformers. The wrapper then
   separates eligible weights into the byte streams described by their dtype
   and encodes a compatible stream with an available precomputed codebook.
3. During inference, the active block or layer is reconstructed bit-for-bit
   in a reusable GPU workspace immediately before execution.
4. `save_pretrained()` runs the full storage codec and stores its streams and
   codebooks in Safetensors; `config.json` identifies the checkpoint format.
5. A later `from_pretrained()` recognizes that format, restores the exact model
   state, and prepares its eligible weights for blockwise GPU decoding.

The CUDA reconstruction path is adapted from the
[official DFloat11 implementation](https://github.com/LeanModels/DFloat11)
under Apache-2.0.

## Research limitations

- GPU weight modules: untied `Linear`, `Conv2d`, and Transformers `Conv1D`.
  `Sequential` and standalone layers use per-layer hooks; existing transformer
  blocks retain batched reconstruction.
- One NVIDIA CUDA device supporting the model's arithmetic and decode layout.
- Sequential, inference-only execution.
- Eligible weights in registered formats are compressed; other state remains
  resident.
- Models without compatible weight modules still use the lossless
  storage format, but load as ordinary models without a compressed GPU runtime.
- No concurrent forwards, training, device changes after loading, or
  `torch.compile`.
- Weights use a temporary workspace; this is not a fused compressed
  matrix-multiplication kernel.
- Decode shared-memory requirements depend on the layout and encoded stream;
  groups exceeding the device limit are rejected before removing model weights.
- Compressed checkpoints require Safetensors.
- Loading a compressed checkpoint still constructs its architecture on CPU,
  so peak host memory is higher than the final resident model.
- Activation is process-wide and affects subsequent compatible Transformers
  model loads and saves.
