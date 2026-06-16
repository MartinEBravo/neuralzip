# Checkpoint Compression Experiments

Install the project in editable mode before running notebooks:

```bash
pip install -e ".[experiments]"
```

The codebase keeps only the minimal model-loading and compression surface:

```python
from neuralzip import Compressor, HFModel

hf_model = HFModel("gpt2", dtype="bfloat16")
results = Compressor("zipnn").compress(hf_model)
```

`HFModel` wraps Hugging Face model loading. `Compressor("zipnn")` applies ZipNN.
For each supported floating-point tensor, the codec applies the same high-level
strategy as ZipNN: dtype-aware bit/byte reordering, byte-position grouping,
chunked Huffman compression, and raw fallback when a chunk does not pass the
compression threshold. The Huffman backend is a compiled C++ extension
(`neuralzip._huffman_cpp`).

`Compressor("neuralzip")` is the experimental NeuralZip entrypoint. For now its
codec is a literal copy of the local ZipNN codec with the name changed, so
future changes can evolve without changing the ZipNN baseline.

For comparisons against the authors' implementation, use
`Compressor("zipnn-src")`. This method calls the official `zipnn` package
directly and treats its binary output as the compressed payload.

The same compressor can also encode checkpoint deltas:

```python
delta = Compressor("zipnn", keep_payloads=True).compress_delta(base_model, next_model)
restored = Compressor("zipnn").decompress_delta(delta, base_model)
```

The main notebook is `examples/1-model-hubs.ipynb`.
