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
compression threshold. Huffman is implemented in the compiled C++ extension
`neuralzip._huffman_cpp`.

`Compressor("zipnn-src")` calls the official `zipnn` package from the
authors and treats its binary output as the compressed payload. Use it as the
external baseline.

`Compressor("neuralzip")` is the single NeuralZip entrypoint and produces a
reversible full-checkpoint representation. It keeps ZipNN's dtype-aware byte
layout and clusters tensors by Jensen-Shannon distance over exponent histograms.
For each cluster and byte stream, a bounded sample selects one of three modes:
raw storage, a shared canonical Huffman codebook, or a shared packed-exponent
Huff0 codebook with `n` in `{2, 3, 4}`. Packed modes use a Top-K vocabulary,
an escape symbol for unseen n-grams, and a raw tail, so reconstruction remains
bitwise exact.

The JS threshold is controlled by `cluster_js_threshold`; sample size by
`max_codebook_sample_bytes`; and the packed candidates by `ngram_values` and
`top_k`. Two-dimensional exponent streams use a deterministic transposed layout
when selected with `exponent_layout="transpose2d"`; the default layout is
`"original"`. Codebooks and vocabularies are stored once in the compressed
header and referenced by chunks.
The result reports clustering as `precompute_seconds`, codebook and mode setup as
`codebook_setup_seconds`, and tensor encoding as `encode_seconds`.
`compression_seconds` includes codebook setup and encoding, while
`total_compression_seconds` additionally includes clustering. Results also
record selected modes, packed chunk counts, escape rate, and round-trip status.

```python
compressed = Compressor(
    "neuralzip",
    verify=True,
    keep_payloads=True,
).compress(hf_model)
Compressor.save(compressed, "model.nzip")
restored = Compressor("neuralzip").decompress_file("model.nzip")
```

The `compressed_bytes` metric is the exact size of this versioned archive, not
an entropy estimate or a sum that omits container metadata.

The same compressor can also encode checkpoint deltas:

```python
delta = Compressor("zipnn-local", keep_payloads=True).compress_delta(base_model, next_model)
restored = Compressor("zipnn-local").decompress_delta(delta, base_model)
```

The main notebook is `examples/1-model-hubs.ipynb`.
