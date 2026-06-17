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

`Compressor("neuralzip-fast")` is the fast NeuralZip entrypoint. It
keeps the same ZipNN tensor byte layout, clusters tensors by Jensen-Shannon
distance over exponent histograms, builds one Huffman codebook per tensor
cluster and byte stream, stores those codebooks in the compressed header, and
then reuses them during chunk encoding with raw fallback. The JS threshold is
controlled by `cluster_js_threshold`; `clustering="architecture-aware"` switches
back to layer-signature clusters. Codebooks are estimated from a bounded sample
per cluster/stream, controlled by `max_codebook_sample_bytes`. During precompute,
the codec also tests sampled streams and pre-marks poorly compressible
cluster/stream pairs as raw, avoiding repeated failed chunk-encoding attempts in
the hot path. This can be disabled with `test_cluster_compressibility=False` for
ablations. The reported `precompute_seconds` is the cold cost of building those
codebooks; `encode_seconds` is the hot compression cost when the codebooks are
already available.
The test is enabled by default because it improves hot compression time on the
GPT-2 BF16 benchmark despite increasing setup time.

`Compressor("neuralzip-ratio")` is the codebook-aware ratio experiment used in
the thesis. It isolates the BF16 exponent stream and, for each Jensen-Shannon
cluster, selects between local ZipNN-style Huffman and full-vocabulary n-gram
arithmetic for `n in {2,3,4}`, including the n-gram codebook cost. This is an
exponent-stream experiment rather than a full-checkpoint codec; its result
explains the observed ratio gain and the codebook boundary. Its arithmetic
backend is isolated in `neuralzip._arithmetic_cpp`.

`Compressor("global-huffman")` is an exponent-stream baseline. It extracts the
ZipNN exponent stream, builds a single Huffman codebook from one representative
chunk with `prior_count=1`, and reuses that codebook for all remaining exponent
chunks with raw fallback.

The same compressor can also encode checkpoint deltas:

```python
delta = Compressor("zipnn-local", keep_payloads=True).compress_delta(base_model, next_model)
restored = Compressor("zipnn-local").decompress_delta(delta, base_model)
```

The main notebook is `examples/1-model-hubs.ipynb`.

On GPT-2 BF16, longer packed n-grams reduce the arithmetic-coded exponent
payload, but explicit n-gram codebooks dominate the final size. NeuralZip-Ratio
therefore reports the best codebook-aware selector, while the larger theoretical
n-gram gains remain a payload-only upper bound unless the codebooks can be
amortized or represented more compactly.
