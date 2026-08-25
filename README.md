# Checkpoint Compression Experiments

Install the project in editable mode before running notebooks:

```bash
pip install -e ".[experiments]"
```

The codebase keeps only the minimal model-loading and compression surface:

```python
from neuralzip import Compressor, HFModel

hf_model = HFModel("gpt2", dtype="bfloat16")
results = Compressor("zipnn-local").compress(hf_model)
```

`HFModel` wraps Hugging Face model loading. `Compressor("zipnn-local")` applies
the local ZipNN-compatible implementation.
For each supported floating-point tensor, the codec applies the same high-level
strategy as ZipNN: dtype-aware bit/byte reordering, byte-position grouping,
chunked Huffman compression, and raw fallback when a chunk does not pass the
compression threshold. Huff0 is implemented in the compiled C++ extension
`neuralzip._huff0_cpp`.

`Compressor("zipnn-src")` calls the official `zipnn` package from the
authors and treats its binary output as the compressed payload. Use it as the
external baseline.

`Compressor("neuralzip")` is the single NeuralZip entrypoint and produces a
reversible full-checkpoint representation. It keeps ZipNN's dtype-aware byte
layout and clusters tensors by Jensen-Shannon distance over exponent histograms.
For each cluster and byte stream, a bounded sample evaluates raw storage, local
Huff0, a shared Huff0 codebook, and packed-exponent Huff0 with `n` in
`{2, 3, 4}`. Packed modes use a Top-K vocabulary, an escape symbol for unseen
n-grams, and a raw tail, so reconstruction remains bitwise exact. New shared
and packed payloads use explicit four-stream Huff0 frames.

The JS threshold is controlled by `cluster_js_threshold`; sample size by
`max_codebook_sample_bytes`; and the packed candidates by `ngram_values` and
`top_k`. Two-dimensional exponent streams use a deterministic transposed layout
when selected with `exponent_layout="transpose2d"`; the default layout is
`"original"`. Codebooks and vocabularies are stored once in the compressed
header and referenced by chunks.

By default, encoding retains no exact-model cache. Once modes and codebooks are
available, a fused native pipeline reads each target tensor, performs ZipNN
reordering and stream extraction, tokenizes packed exponents, applies Huff0 and
its local/raw fallback, and returns only the final payload. Reordered streams
and tokens are transient. Tensors are submitted through input-size-windowed
native batches whose jobs share one queue per window. The default threshold is
128 MiB; an individual larger tensor forms its own window. Encoding a new
checkpoint therefore does not retain a second transformed copy of its
floating-point parameters in memory.

An exact-target cache remains available only as an explicit repeated-encoding
option with `enable_prepacked_cache=True`. It is valid while the same named
tensor objects remain immutable, with the same pointers and PyTorch version
counters. It is never transferred between family members, serialized, or
counted in the compression ratio. Writes through `tensor.data` or external
NumPy/storage aliases bypass the version counter and require explicit cache
invalidation.

Mode selection is time-aware. Among modes within `speed_ratio_tolerance`
(default 3% relative compressed bytes) of the best estimated ratio, it chooses
the fastest direct encoding path. Packed opportunities are ranked by bytes
saved per additional second and admitted under `packed_time_budget_ratio`
(default 100% of estimated base time), with `packed_source_budget_ratio` retained
as a coverage cap.

Timing is reported as `clustering_seconds`, `codebook_setup_seconds`, and
`direct_encode_seconds` (`encode_seconds`). The encode measurement includes all
target-dependent work. `prepack_seconds` and exact-cache memory remain zero
unless repeated-target caching was explicitly enabled. Results also record
selected modes, packed chunks, escape rate, and round-trip status.

```python
compressed = Compressor(
    "neuralzip",
    verify=True,
    keep_payloads=True,
).compress(hf_model)
Compressor.save(compressed, "model.nzip")
restored = Compressor("neuralzip").decompress_file("model.nzip")
```

Direct family transfer reuses only the structural family plan:

```python
codec = Compressor("neuralzip").codec
codec.prepare_clusters(source_parameters)
codec.build_codebooks_and_select_modes(source_parameters)
codec.prepare_direct_transfer_target(target_parameters)
target_payloads = codec.compress_tensors(target_parameters)
```

The mapping step reads target metadata only. It does not compute target
histograms, perform JS matching or mode probing, or retain target data.

The `compressed_bytes` metric is the exact size of this versioned archive, not
an entropy estimate or a sum that omits container metadata.

The same compressor can also encode checkpoint deltas:

```python
delta = Compressor("zipnn-local", keep_payloads=True).compress_delta(base_model, next_model)
restored = Compressor("zipnn-local").decompress_delta(delta, base_model)
```

The main notebook is `examples/1-model-hubs.ipynb`.
