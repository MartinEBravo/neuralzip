"""Canonical model catalog shared by experiment runners."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    suites: tuple[str, ...]
    backend: str
    source: str
    parameters_b: float | None = None
    constructor: str | None = None
    subfolder: str | None = None
    weight_file: str | None = None


_PHENOMENON = ("phenomenon",)
_PHENOMENON_VISION_STORAGE = ("phenomenon", "vision", "model-storing")
_VISION_STORAGE = ("vision", "model-storing")
_MODEL_STORAGE = ("model-storing",)
_MOE = ("moe",)
_DISTILLED = ("distilled",)
_SETUP_TRANSFER = ("setup-transfer",)


def _hf_model(
    key: str,
    label: str,
    family: str,
    suites: tuple[str, ...],
    source: str,
    parameters_b: float | None,
    *,
    subfolder: str | None = None,
    weight_file: str | None = None,
) -> ModelSpec:
    return ModelSpec(
        key,
        label,
        family,
        suites,
        "hf",
        source,
        parameters_b,
        subfolder=subfolder,
        weight_file=weight_file,
    )


def _torchvision_model(
    key: str,
    label: str,
    family: str,
    suites: tuple[str, ...],
    parameters_b: float,
) -> ModelSpec:
    constructor = key.replace("-", "_")
    return ModelSpec(
        key,
        label,
        family,
        suites,
        "torchvision",
        f"torchvision.models.{constructor}",
        parameters_b,
        constructor=constructor,
    )


MODEL_SPECS = (
    # Phenomenon and vision baselines.
    _hf_model("gpt2", "GPT-2", "GPT-2", _PHENOMENON, "gpt2", 0.124),
    _hf_model(
        "bert-base",
        "BERT Base",
        "BERT",
        _PHENOMENON,
        "google-bert/bert-base-uncased",
        0.109,
    ),
    _torchvision_model(
        "resnet18", "ResNet-18", "ResNet", _PHENOMENON_VISION_STORAGE, 0.012
    ),
    _torchvision_model(
        "resnet34", "ResNet-34", "ResNet", _PHENOMENON_VISION_STORAGE, 0.022
    ),
    _torchvision_model("resnet50", "ResNet-50", "ResNet", _VISION_STORAGE, 0.026),
    _torchvision_model("resnet101", "ResNet-101", "ResNet", _VISION_STORAGE, 0.045),
    _torchvision_model("resnet152", "ResNet-152", "ResNet", _VISION_STORAGE, 0.060),
    # Dense language models stored in full.
    _hf_model(
        "qwen25-1p5b",
        "Qwen2.5-1.5B",
        "Qwen2.5",
        _MODEL_STORAGE,
        "Qwen/Qwen2.5-1.5B",
        1.5,
    ),
    _hf_model(
        "qwen25-3b", "Qwen2.5-3B", "Qwen2.5", _MODEL_STORAGE, "Qwen/Qwen2.5-3B", 3.0
    ),
    _hf_model(
        "qwen25-7b", "Qwen2.5-7B", "Qwen2.5", _MODEL_STORAGE, "Qwen/Qwen2.5-7B", 7.0
    ),
    _hf_model(
        "qwen25-14b", "Qwen2.5-14B", "Qwen2.5", _MODEL_STORAGE, "Qwen/Qwen2.5-14B", 14.0
    ),
    _hf_model(
        "olmo2-1b", "OLMo-2-1B", "OLMo 2", _MODEL_STORAGE, "allenai/Olmo-2-0425-1B", 1.0
    ),
    _hf_model(
        "olmo2-7b", "OLMo-2-7B", "OLMo 2", _MODEL_STORAGE, "allenai/OLMo-2-1124-7B", 7.0
    ),
    _hf_model(
        "olmo2-13b",
        "OLMo-2-1124-13B",
        "OLMo 2",
        _MODEL_STORAGE,
        "allenai/OLMo-2-1124-13B",
        13.0,
    ),
    _hf_model(
        "mistral7b",
        "Mistral-7B",
        "Mistral",
        _MODEL_STORAGE,
        "mistralai/Mistral-7B-v0.3",
        7.0,
    ),
    _hf_model(
        "mistral-nemo-12b",
        "Mistral-Nemo-Base-2407",
        "Mistral",
        _MODEL_STORAGE,
        "mistralai/Mistral-Nemo-Base-2407",
        12.0,
    ),
    _hf_model(
        "llama32-1b",
        "Llama-3.2-1B",
        "Llama 3",
        _MODEL_STORAGE,
        "meta-llama/Llama-3.2-1B",
        1.0,
    ),
    _hf_model(
        "llama32-3b",
        "Llama-3.2-3B",
        "Llama 3",
        _MODEL_STORAGE,
        "meta-llama/Llama-3.2-3B",
        3.0,
    ),
    _hf_model(
        "llama31-nemotron-8b",
        "Llama-3.1-Nemotron-Nano-8B-v1",
        "Llama 3 / derivatives",
        _MODEL_STORAGE,
        "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        8.0,
    ),
    _hf_model(
        "gemma4-31b",
        "Gemma 4 31B",
        "Gemma 4",
        _MODEL_STORAGE,
        "google/gemma-4-31B",
        31.0,
    ),
    _hf_model(
        "gemma3-1b",
        "Gemma 3 1B PT",
        "Gemma 3",
        _MODEL_STORAGE,
        "google/gemma-3-1b-pt",
        1.0,
    ),
    _hf_model(
        "gemma3-4b",
        "Gemma 3 4B PT",
        "Gemma 3",
        _MODEL_STORAGE,
        "google/gemma-3-4b-pt",
        4.0,
    ),
    _hf_model(
        "gemma3-12b",
        "Gemma 3 12B PT",
        "Gemma 3",
        _MODEL_STORAGE,
        "google/gemma-3-12b-pt",
        12.0,
    ),
    _hf_model(
        "phi4-14b", "Phi-4 14B", "Phi 4", _MODEL_STORAGE, "microsoft/phi-4", 14.0
    ),
    _hf_model(
        "qwen35-27b",
        "Qwen3.5 27B",
        "Qwen 3.5",
        _MODEL_STORAGE,
        "Qwen/Qwen3.5-27B",
        27.0,
    ),
    _hf_model(
        "qwen35-08b-base",
        "Qwen3.5-0.8B-Base",
        "Qwen 3.5",
        _MODEL_STORAGE,
        "Qwen/Qwen3.5-0.8B-Base",
        0.8,
    ),
    _hf_model(
        "qwen35-2b-base",
        "Qwen3.5-2B-Base",
        "Qwen 3.5",
        _MODEL_STORAGE,
        "Qwen/Qwen3.5-2B-Base",
        2.0,
    ),
    _hf_model(
        "latamgpt-70b",
        "LatamGPT-70B",
        "Llama 3 / LatamGPT",
        _MODEL_STORAGE,
        "latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0",
        70.0,
    ),
    # Additional vision and diffusion models.
    _torchvision_model(
        "convnext-tiny", "ConvNeXt-Tiny", "ConvNeXt", _VISION_STORAGE, 0.029
    ),
    _torchvision_model(
        "convnext-small", "ConvNeXt-Small", "ConvNeXt", _VISION_STORAGE, 0.050
    ),
    _torchvision_model(
        "convnext-base", "ConvNeXt-Base", "ConvNeXt", _VISION_STORAGE, 0.089
    ),
    _torchvision_model(
        "convnext-large", "ConvNeXt-Large", "ConvNeXt", _VISION_STORAGE, 0.198
    ),
    _torchvision_model(
        "efficientnet-b0", "EfficientNet-B0", "EfficientNet", _VISION_STORAGE, 0.005
    ),
    _torchvision_model(
        "efficientnet-b3", "EfficientNet-B3", "EfficientNet", _VISION_STORAGE, 0.012
    ),
    _torchvision_model(
        "efficientnet-b7", "EfficientNet-B7", "EfficientNet", _VISION_STORAGE, 0.066
    ),
    _hf_model(
        "sd15-unet",
        "Stable Diffusion v1.5 UNet",
        "Stable Diffusion",
        _VISION_STORAGE,
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        0.860,
        subfolder="unet",
        weight_file="unet/diffusion_pytorch_model.safetensors",
    ),
    # Additional task families used to evaluate setup transfer.
    _hf_model(
        "whisper-large-v3",
        "Whisper large-v3",
        "Whisper v3",
        _SETUP_TRANSFER,
        "openai/whisper-large-v3",
        1.55,
    ),
    _hf_model(
        "whisper-large-v3-turbo",
        "Whisper large-v3 Turbo",
        "Whisper v3",
        _SETUP_TRANSFER,
        "openai/whisper-large-v3-turbo",
        0.81,
    ),
    _hf_model(
        "siglip2-giant",
        "SigLIP 2 Giant",
        "SigLIP 2",
        _SETUP_TRANSFER,
        "google/siglip2-giant-opt-patch16-384",
        None,
    ),
    _hf_model(
        "siglip2-so400m",
        "SigLIP 2 So400m",
        "SigLIP 2",
        _SETUP_TRANSFER,
        "google/siglip2-so400m-patch16-384",
        None,
    ),
    # Mixture-of-experts models.
    _hf_model("zaya1-8b", "ZAYA1-8B", "ZAYA", _MOE, "Zyphra/ZAYA1-8B", 8.0),
    _hf_model(
        "lfm25-8b-a1b",
        "Liquid AI LFM2.5-8B-A1B",
        "LFM",
        _MOE,
        "LiquidAI/LFM2.5-8B-A1B",
        8.0,
    ),
    _hf_model("jetmoe-8b", "JetMoE-8B", "JetMoE", _MOE, "jetmoe/jetmoe-8b", 8.0),
    _hf_model(
        "olmoe-1b-7b-0125",
        "OLMoE-1B-7B-0125",
        "OLMoE",
        _MOE,
        "allenai/OLMoE-1B-7B-0125",
        7.0,
    ),
    _hf_model(
        "qwen15-moe-a27b",
        "Qwen1.5-MoE-A2.7B",
        "Qwen 1.5 MoE",
        _MOE,
        "Qwen/Qwen1.5-MoE-A2.7B",
        14.3,
    ),
    _hf_model(
        "qwen3-30b-a3b", "Qwen3-30B-A3B", "Qwen 3 MoE", _MOE, "Qwen/Qwen3-30B-A3B", 30.5
    ),
    _hf_model(
        "qwen35-35b-a3b",
        "Qwen3.5-35B-A3B",
        "Qwen 3.5 MoE",
        _MOE,
        "Qwen/Qwen3.5-35B-A3B",
        35.0,
    ),
    _hf_model(
        "deepseek-moe-16b",
        "DeepSeek-MoE-16B-Base",
        "DeepSeek MoE",
        _MOE,
        "deepseek-ai/deepseek-moe-16b-base",
        16.0,
    ),
    _hf_model(
        "mixtral-8x7b",
        "Mixtral-8x7B-v0.1",
        "Mixtral",
        _MOE,
        "mistralai/Mixtral-8x7B-v0.1",
        46.7,
    ),
    _hf_model(
        "gemma4-26b-a4b",
        "Gemma-4-26B-A4B",
        "Gemma 4 MoE",
        _MOE,
        "google/gemma-4-26B-A4B",
        26.0,
    ),
    # Distilled models.
    _hf_model(
        "deepseek-r1-distill-llama-8b",
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek R1 distilled",
        _DISTILLED,
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        8.0,
    ),
    _hf_model(
        "deepseek-r1-distill-qwen-7b",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek R1 distilled",
        _DISTILLED,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        7.0,
    ),
    _hf_model(
        "gemma2-9b-distilled",
        "Gemma-2-9B-Distilled",
        "Gemma 2 distilled",
        _DISTILLED,
        "kazuyamaa/gemma-2-9b-opd-Distill-v001",
        9.0,
    ),
    _hf_model(
        "qwen25-7b-distilled",
        "Qwen2.5-7B-Instruct-Distilled",
        "Qwen 2.5 distilled",
        _DISTILLED,
        "RZ412/Qwen2.5-7B-Instruct-S1K-DeepSeek-Distilled",
        7.0,
    ),
)

SPECS_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def _quantized_model(
    name: str,
    model_id: str,
    parameters_b: float,
    representation: str,
    access: str,
    artifact: str,
    revision: str,
) -> dict[str, object]:
    selector = (
        {"kind": "safetensors-index", "index": "model.safetensors.index.json"}
        if artifact == "index"
        else {"kind": "exact-files", "files": (artifact,)}
    )
    return {
        "name": name,
        "model_id": model_id,
        "parameters_b": parameters_b,
        "default_dtype": "auto",
        "representation": representation,
        "category": "quantized",
        "access": access,
        "revision": revision,
        "artifact_selector": selector,
    }


QUANTIZED_MODEL_SPECS = (
    _quantized_model(
        "Gemma 4 31B-4bit",
        "unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
        31.0,
        "4-bit quantized",
        "open-or-token",
        "index",
        "8e256fc6d63003fc0ca8c91b976e6dcc38433385",
    ),
    _quantized_model(
        "Phi-4 14B-4bit",
        "unsloth/phi-4-bnb-4bit",
        14.0,
        "4-bit quantized",
        "open",
        "index",
        "098ea917f967c8811e6f579cc38872b2bf3d4bf4",
    ),
    _quantized_model(
        "Qwen3.5 27B-4bit",
        "skkwowee/Qwen3.5-27B-bnb-4bit",
        27.0,
        "4-bit quantized",
        "open",
        "model.safetensors",
        "26ba09cf1db425f2c170b3446f42de8891e28400",
    ),
    _quantized_model(
        "Qwen 2.5 7B AWQ",
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        7.0,
        "4-bit AWQ",
        "open",
        "index",
        "b25037543e9394b818fdfca67ab2a00ecc7dd641",
    ),
    _quantized_model(
        "Qwen 2.5 7B GPTQ-Int4",
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        7.0,
        "4-bit GPTQ",
        "open",
        "index",
        "e9c932ac1893a49ae0fc497ad6e1e86e2e39af20",
    ),
    _quantized_model(
        "LatamGPT-70B-4bit",
        "jairo/Llama-3.1-70B-LatamGPT-SFT-1.0-GGUF",
        70.0,
        "4-bit quantized",
        "open",
        "latamgpt-Q4_K_M.gguf",
        "43aaa004bc882624cfed02c98a898516987da773",
    ),
)
