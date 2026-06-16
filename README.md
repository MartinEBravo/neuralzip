# Checkpoint Compression Experiments

Install the project in editable mode before running notebooks:

```bash
pip install -e ".[experiments]"
```

The codebase keeps only the minimal model-loading and ZipNN compression surface:

```python
from neuralzip import Compressor, HFModel

hf_model = HFModel("gpt2", dtype="bfloat16")
results = Compressor("zipnn").compress(hf_model)
```

`HFModel` wraps Hugging Face model loading. `Compressor("zipnn")` applies the
local ZipNN-inspired implementation: each floating-point tensor is split into
byte-position streams, profitable streams are encoded with canonical Huffman,
and the remaining streams are stored raw so decompression reconstructs the
original tensors bit by bit.

The main notebook is `examples/1-model-hubs.ipynb`.
