# NeuralZip Checkpoint Experiments

Install the project in editable mode before running notebooks:

```bash
pip install -e ".[experiments]"
```

The main experiment entrypoint is `examples/1-model-hubs.ipynb`.
That notebook contains the complete model-hub workflow used by the thesis:
model storing, checkpoint download, experiment execution, and result upload.

The current thesis checkpoint suite targets:

- Llama 3.3 70B Instruct
- Gemma 4 31B
- Phi-4 14B
- Qwen3.5 27B

The MoE-specific notebook (`examples/4-moe.ipynb`) uses a separate sparse-model
suite:

- `gpt-oss-120b`
- DeepSeek V4 Pro
- Kimi K2.6
- GLM-5.1 744B

Run `examples/1-model-hubs.ipynb` from top to bottom for the notebook-first
pipeline. The notebook keeps destructive or expensive operations explicit:
download and upload cells should be reviewed before execution, and local model
storage defaults should be configured there.

Legacy notebooks can still be used for small local debugging, but thesis tables
should come from the model-hub notebook so storing, download, experiment, and
upload provenance stay in one place.

The importable library lives under `neuralzip/`; for example:

```python
from neuralzip import Compressor, HFModel

hf_model = HFModel("bert-base-uncased", dtype="bfloat16")
zipnn_results = Compressor("ZipNN").compress(hf_model)
```

Compression backends live in `neuralzip/algorithms/`.
Model-suite selection, download planning, experiment execution, and upload
manifests live in the notebooks.
