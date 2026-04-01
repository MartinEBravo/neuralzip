from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "imagenes" / "plots"
RESULTS_CSV = ROOT / "compression_results.csv"
DEFAULT_METHODS = ["ZipNN", "ZipNN+", "ZipNN++"]
MODEL_LABELS = {
    "bert-base-uncased": "BERT-base",
    "roberta-base": "RoBERTa-base",
    "google-t5/t5-small": "T5-small",
    "gpt2": "GPT-2",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeek-R1-Qwen 7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1-Qwen 14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepSeek-R1-Qwen 32B",
    "google/gemma-2-2b": "Gemma 2 2B",
    "google/gemma-2-9b": "Gemma 2 9B",
    "google/gemma-2-27b": "Gemma 2 27B",
    "meta-llama/Llama-3.2-1B": "Llama 3.2 1B",
    "meta-llama/Llama-3.2-3B": "Llama 3.2 3B",
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
    "Qwen/Qwen2.5-Coder-7B": "Qwen2.5-Coder 7B",
}
MODEL_PRESETS = {
    "baseline": [
        "gpt2",
        "bert-base-uncased",
        "roberta-base",
        "google-t5/t5-small",
    ],
    "llama32": [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
    ],
    "deepseek-r1-distill-qwen": [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ],
    "gemma2": [
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
    ],
    "open-source-recommended": [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "microsoft/Phi-4-mini-instruct",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "Qwen/Qwen2.5-Coder-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    ],
    "open-source-all": [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "microsoft/Phi-4-mini-instruct",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
        "Qwen/Qwen2.5-Coder-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ],
}
HF_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.safetensors.index.json",
]
