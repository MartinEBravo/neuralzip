# Model Compression Experiments

Build native compression backends:

```bash
make native
```

Run experiments through the single entrypoint:

```bash
python3 experiments.py --mode full --models "Gemma 4 E2B"
python3 experiments.py --mode zipnn --models "Gemma 4 E2B"
python3 experiments.py --mode exponent --only ngram-sweep --models "Gemma 4 E2B"
python3 experiments.py --mode entropy --field exponent --models "Gemma 4 E2B"
python3 experiments.py --mode mm-repair --models "Gemma 4 E2B"
```

Compression backends live in `compression_algorithms/`. The rest of the model
loading, field extraction, and experiment orchestration stays at the repository
root.
