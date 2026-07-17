from config import MODEL_SUITE as _MODEL_SUITE, ZIPNN_METHOD

MODEL_SUITE = [spec for spec in _MODEL_SUITE if spec["name"] == "Qwen2.5-1.5B"]
