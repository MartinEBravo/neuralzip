# NeuralZip Checkpoint Experiments

Build native compression backends:

```bash
make native
```

The main experiment entrypoint is `examples/1-model-hubs.ipynb`.
That notebook contains the complete model-hub workflow used by the thesis:
model storing, checkpoint download, experiment execution, and result upload.

The current thesis checkpoint suite targets:

- Llama 3.3 70B Instruct
- Gemma 4 31B
- `gpt-oss-120b`
- Qwen2.5 72B Instruct

Run `examples/1-model-hubs.ipynb` from top to bottom for the notebook-first
pipeline. The notebook keeps destructive or expensive operations explicit:
download and upload cells should be reviewed before execution, and local model
storage defaults should be configured there.

Legacy notebooks can still be used for small local debugging, but thesis tables
should come from the model-hub notebook so storing, download, experiment, and
upload provenance stay in one place.

Compression backends live in `compression_algorithms/`. Shared model-suite
constants and selectors live in small Python modules, but the experiment
workflow itself belongs in notebooks.
