from config import MODEL_SUITE as _MODEL_SUITE, ZIPNN_METHOD

# First remote model-storage pass: open or already usable checkpoints up to 14B.
MODEL_SUITE = [
    spec for spec in _MODEL_SUITE
    if float(spec.get("parameters_b", 0.0)) <= 14.0
]
