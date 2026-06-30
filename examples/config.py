ZIPNN_METHOD = "HUFFMAN"

MODEL_SUITE = [
    {
        "name": "Llama 3.3 70B Instruct",
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "parameters_b": 70.0,
        "default_dtype": "bf16",
        "access": "gated",
        "local_subdir": "meta-llama__Llama-3.3-70B-Instruct",
    },
    {
        "name": "Gemma 4 31B",
        "model_id": "google/gemma-4-31B",
        "parameters_b": 31.0,
        "default_dtype": "bf16",
        "access": "gated-or-token",
        "local_subdir": "google__gemma-4-31B",
    },
    {
        "name": "Phi-4 14B",
        "model_id": "microsoft/phi-4",
        "parameters_b": 14.0,
        "default_dtype": "bf16",
        "access": "open-or-token",
        "local_subdir": "microsoft__phi-4",
    },
    {
        "name": "Qwen3.5 27B",
        "model_id": "Qwen/Qwen3.5-27B",
        "parameters_b": 27.0,
        "default_dtype": "bf16",
        "access": "open-or-token",
        "local_subdir": "Qwen__Qwen3.5-27B",
    },
    {
        "name": "LatamGPT-70B",
        "model_id": "latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0",
        "parameters_b": 70.0,
        "default_dtype": "bf16",
        "access": "gated",
        "local_subdir": "latam-gpt__Llama-3.1-70B-LatamGPT-SFT-1.0"
    }
]

COMPRESSOR_SUITE = [
    {
        "name": "ZipNN local",
        "method": "zipnn-local",
        "verify": False,
        "description": (
            "the local ZipNN-compatible codec applied tensor by tensor with "
            "dtype-aware byte reordering and Huffman compression."
        ),
    },
    {
        "name": "ZipNN src",
        "method": "zipnn-src",
        "zipnn_method": ZIPNN_METHOD,
        "verify": False,
        "description": "the official ZipNN package from the authors.",
    },
    {
        "name": "NeuralZip-Fast",
        "method": "neuralzip-fast",
        "verify": False,
        "description": (
            "the fast NeuralZip codec with Jensen-Shannon "
            "clustered codebooks reused by byte stream."
        ),
    },
    {
        "name": "NeuralZip-Ratio",
        "method": "neuralzip-ratio",
        "verify": False,
        "description": (
            "the codebook-aware exponent-stream ratio experiment selecting "
            "between ZipNN-local Huffman and full-vocabulary n-gram arithmetic."
        ),
    },
    {
        "name": "Global Huffman",
        "method": "global-huffman",
        "verify": False,
        "description": (
            "an exponent-stream baseline that builds one Huffman tree from a "
            "representative chunk and reuses it for the remaining chunks."
        ),
    },
]

MOE_MODEL_SUITE = [
    {
        "name": "ZAYA1-8B",
        "model_id": "Zyphra/ZAYA1-8B",
        "parameters_b": 8.0,
        "default_dtype": "bf16",
        "category": "moe",
        "access": "open",
    },
    {
        "name": "Liquid AI LFM2.5-8B-A1B",
        "model_id": "LiquidAI/LFM2.5-8B-A1B",
        "parameters_b": 8.0,
        "default_dtype": "bf16",
        "category": "moe",
        "access": "open",
    },
    {
        "name": "JetMoE-8B",
        "model_id": "jetmoe/jetmoe-8b",
        "parameters_b": 8.0,
        "default_dtype": "bf16",
        "category": "moe",
        "access": "open",
    },
]

LOSSY_LOSSLESS_MODEL_SUITE = [
    {
        "name": "Gemma 4 31B-4bit",
        "model_id": "unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
        "parameters_b": 31.0,
        "default_dtype": "auto",
        "representation": "4-bit quantized",
        "category": "quantized",
        "access": "open-or-token",
    },
    {
        "name": "Phi-4 14B-4bit",
        "model_id": "unsloth/phi-4-bnb-4bit",
        "parameters_b": 14.0,
        "default_dtype": "auto",
        "representation": "4-bit quantized",
        "category": "quantized",
        "access": "open",
    },
    {
        "name": "Qwen3.5 27B-4bit",
        "model_id": "skkwowee/Qwen3.5-27B-bnb-4bit",
        "parameters_b": 27.0,
        "default_dtype": "auto",
        "representation": "4-bit quantized",
        "category": "quantized",
        "access": "open",
    },
    {
        "name": "LatamGPT-70B-4bit",
        "model_id": "jairo/Llama-3.1-70B-LatamGPT-SFT-1.0-GGUF",
        "parameters_b": 70.0,
        "default_dtype": "auto",
        "representation": "4-bit quantized",
        "category": "quantized",
        "access": "open",
    },
    {
        "name": "Llama-3-8B-Instruct-Distilled",
        "model_id": "JunchengXie/Meta-Llama-3-8B-Instruct-openwebtext-distilled-80k-gpt4",
        "parameters_b": 8.0,
        "default_dtype": "bf16",
        "representation": "distilled",
        "category": "distilled",
        "access": "open-or-token",
    },
    {
        "name": "Gemma-2-9B-Distilled",
        "model_id": "kazuyamaa/gemma-2-9b-opd-Distill-v001",
        "parameters_b": 9.0,
        "default_dtype": "bf16",
        "representation": "distilled",
        "category": "distilled",
        "access": "open-or-token",
    },
    {
        "name": "Qwen2.5-7B-Instruct-Distilled",
        "model_id": "RZ412/Qwen2.5-7B-Instruct-S1K-DeepSeek-Distilled",
        "parameters_b": 7.0,
        "default_dtype": "bf16",
        "representation": "distilled",
        "category": "distilled",
        "access": "open",
    },
]
