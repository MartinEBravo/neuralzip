ZIPNN_METHOD = "HUFFMAN"

MODEL_SUITE = [
    {
        "name": "Qwen2.5-1.5B",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "parameters_b": 1.5,
        "default_dtype": "bf16",
        "access": "open",
    },
    {
        "name": "Qwen2.5-3B",
        "model_id": "Qwen/Qwen2.5-3B",
        "parameters_b": 3.0,
        "default_dtype": "bf16",
        "access": "open",
    },
    {
        "name": "OLMo-2-1B",
        "model_id": "allenai/Olmo-2-0425-1B",
        "parameters_b": 1.0,
        "default_dtype": "bf16",
        "access": "open",
    },
    {
        "name": "OLMo-2-7B",
        "model_id": "allenai/Olmo-2-0325-7B",
        "parameters_b": 7.0,
        "default_dtype": "bf16",
        "access": "open",
    },
    {
        "name": "Mistral-7B",
        "model_id": "mistralai/Mistral-7B-v0.3",
        "parameters_b": 7.0,
        "default_dtype": "bf16",
        "access": "open",
    },
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
        "name": "NeuralZip",
        "method": "neuralzip",
        "verify": False,
        "description": (
            "the unified codec: Jensen-Shannon tensor clusters, shared Huff0 "
            "codebooks, and packed exponent n-grams selected per cluster/stream."
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
        "artifact_selector": {
            "kind": "safetensors-index",
            "index": "model.safetensors.index.json",
        },
    },
    {
        "name": "Phi-4 14B-4bit",
        "model_id": "unsloth/phi-4-bnb-4bit",
        "parameters_b": 14.0,
        "default_dtype": "auto",
        "representation": "4-bit quantized",
        "category": "quantized",
        "access": "open",
        "artifact_selector": {
            "kind": "safetensors-index",
            "index": "model.safetensors.index.json",
        },
    },
    {
        "name": "Qwen3.5 27B-4bit",
        "model_id": "skkwowee/Qwen3.5-27B-bnb-4bit",
        "parameters_b": 27.0,
        "default_dtype": "auto",
        "representation": "4-bit quantized",
        "category": "quantized",
        "access": "open",
        "artifact_selector": {
            "kind": "exact-files",
            "files": ("model.safetensors",),
        },
    },
    {
        "name": "LatamGPT-70B-4bit",
        "model_id": "jairo/Llama-3.1-70B-LatamGPT-SFT-1.0-GGUF",
        "parameters_b": 70.0,
        "default_dtype": "auto",
        "representation": "4-bit quantized",
        "category": "quantized",
        "access": "open",
        "artifact_selector": {
            "kind": "exact-files",
            "files": ("latamgpt-Q4_K_M.gguf",),
        },
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

VISION_DIFFUSION_MODEL_SUITE = [
    {
        "name": "ResNet-18",
        "family": "cnn",
        "backend": "torchvision",
        "constructor": "resnet18",
        "weights": "DEFAULT",
        "default_dtype": "bf16",
    },
    {
        "name": "VGG-16 BN",
        "family": "cnn",
        "backend": "torchvision",
        "constructor": "vgg16_bn",
        "weights": "DEFAULT",
        "default_dtype": "bf16",
    },
    {
        "name": "ConvNeXt-Tiny",
        "family": "cnn",
        "backend": "torchvision",
        "constructor": "convnext_tiny",
        "weights": "DEFAULT",
        "default_dtype": "bf16",
    },
    {
        "name": "ViT-B/16",
        "family": "vision-transformer",
        "backend": "torchvision",
        "constructor": "vit_b_16",
        "weights": "DEFAULT",
        "default_dtype": "bf16",
    },
    {
        "name": "Swin-T",
        "family": "vision-transformer",
        "backend": "torchvision",
        "constructor": "swin_t",
        "weights": "DEFAULT",
        "default_dtype": "bf16",
    },
    {
        "name": "Stable Diffusion v1.5 UNet",
        "family": "diffusion",
        "backend": "diffusers-unet",
        "loader": "UNet2DConditionModel",
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "subfolder": "unet",
        "default_dtype": "bf16",
        "fallback": {
            "name": "DDPM CIFAR-10 UNet (open fallback)",
            "family": "diffusion",
            "backend": "diffusers-unet",
            "loader": "UNet2DModel",
            "model_id": "google/ddpm-cifar10-32",
            "default_dtype": "bf16",
            "fallback_for": "Stable Diffusion v1.5 UNet",
        },
    },
]
