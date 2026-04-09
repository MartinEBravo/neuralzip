from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = ROOT / "analysis"
PLOTS_DIR = ROOT / "imagenes" / "plots"
RESULTS_CSV = ROOT / "compression_results.csv"
PIPELINE_SUMMARY_CSV = ANALYSIS_DIR / "pipeline_summary.csv"
COMPRESSION_METHODS = ("ZipNN", "NeuralZip", "NeuralZipFull")
DEFAULT_QUANT_BITS = (8, 12)

MODEL_LABELS = {
    # ------- GPT -------
    "gpt2": "GPT-2",
    "sshleifer/tiny-gpt2": "GPT-2 Nano",
    # ----- DeepSeek -----
    # "deepseek-ai/DeepSeek-V3.2": "DeepSeek V3.2",
    # "deepseek-ai/DeepSeek-V3.2-Speciale": "DeepSeek V3.2 Speciale",
    # "deepseek-ai/DeepSeek-R1": "DeepSeek R1",
    # "deepseek-ai/DeepSeek-R1-Zero": "DeepSeek R1 Zero",
    # ------- Gemma ------
    "google/gemma-4-E2B": "Gemma 4 E2B",
    "google/gemma-4-E2B-it": "Gemma 4 E2B IT",
    "google/gemma-4-E4B": "Gemma 4 E4B",
    "google/gemma-4-E4B-it": "Gemma 4 E4B IT",
    # "google/gemma-4-26B-A4B": "Gemma 4 26B A4B",
    # "google/gemma-4-26B-A4B-it": "Gemma 4 26B A4B IT",
    # "google/gemma-4-31B": "Gemma 4 31B",
    # "google/gemma-4-31B-it": "Gemma 4 31B IT",
    # ------- Llama ------
    "meta-llama/Llama-3.2-1B": "Llama 3.2 1B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 1B Instruct",
    "meta-llama/Llama-3.2-3B": "Llama 3.2 3B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 3B Instruct",
    "meta-llama/Llama-3.2-11B-Vision": "Llama 3.2 11B Vision",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "Llama 3.2 11B Vision Instruct",
    # "meta-llama/Llama-3.2-90B-Vision": "Llama 3.2 90B Vision",
    # "meta-llama/Llama-3.2-90B-Vision-Instruct": "Llama 3.2 90B Vision Instruct",
    # --------- Microsoft ----------
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
    # ------ Qwen ------
    "Qwen/Qwen3.5-0.8B": "Qwen 3.5 0.8B",
    "Qwen/Qwen3.5-2B": "Qwen 3.5 2B",
    "Qwen/Qwen3.5-4B": "Qwen 3.5 4B",
    "Qwen/Qwen3.5-9B": "Qwen 3.5 9B",
    # "Qwen/Qwen3.5-27B": "Qwen 3.5 27B",
    # "Qwen/Qwen3.5-35B-A3B": "Qwen 3.5 35B A3B",
    # "Qwen/Qwen3.5-122B-A10B": "Qwen 3.5 122B A10B",
    # "Qwen/Qwen3.5-397B-A17B": "Qwen 3.5 397B A17B",
}

MODEL_PRESETS: dict[str, list[str]] = {
    "baseline": [
        "gpt2",
        "sshleifer/tiny-gpt2",
    ],
    "gemma": [
        "google/gemma-4-E2B",
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B",
        "google/gemma-4-E4B-it",
    ],
    "llama": [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
    ],
    "qwen": [
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
    ],
    "phi": [
        "microsoft/Phi-4-mini-instruct",
    ],
    "all": [
        "gpt2",
        "sshleifer/tiny-gpt2",
        "google/gemma-4-E2B",
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B",
        "google/gemma-4-E4B-it",
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "microsoft/Phi-4-mini-instruct",
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
    ],
}


def get_model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


HF_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.safetensors.index.json",
]
