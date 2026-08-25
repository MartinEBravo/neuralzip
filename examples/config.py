ZIPNN_METHOD = "HUFFMAN"

try:
    from .model_catalog import MODEL_SPECS
except ImportError:  # Direct execution from the examples directory.
    from model_catalog import MODEL_SPECS


def _catalog_entry(spec, attributes):
    """Project one canonical model spec to the legacy dictionary API."""
    return {
        "name": spec.label,
        "model_id": spec.source,
        "parameters_b": spec.parameters_b,
        **attributes,
    }


def _legacy_attributes(access, *, category=False, local_subdir=None):
    attributes = {"default_dtype": "bf16"}
    if category:
        attributes["category"] = "model-storing"
    attributes["access"] = access
    if local_subdir is not None:
        attributes["local_subdir"] = local_subdir
    return attributes


_MODEL_SUITE_LEGACY_ATTRIBUTES = {
    key: _legacy_attributes(access, category=category, local_subdir=local_subdir)
    for key, access, category, local_subdir in (
        ("qwen25-15b", "open", False, None),
        ("qwen25-3b", "open", False, None),
        ("qwen25-7b", "open", True, None),
        ("qwen25-14b", "open", True, None),
        ("olmo2-1b", "open", False, None),
        ("olmo2-7b", "open", False, None),
        ("olmo2-13b", "open", True, None),
        ("mistral7b", "open", False, None),
        ("mistral-nemo-12b", "open", True, None),
        ("llama32-1b", "gated", True, None),
        ("llama32-3b", "gated", True, None),
        ("llama31-nemotron-8b", "open", True, None),
        ("llama33-70b", "gated", False, "meta-llama__Llama-3.3-70B-Instruct"),
        ("gemma4-31b", "gated-or-token", False, "google__gemma-4-31B"),
        ("gemma3-1b", "gated-or-token", True, None),
        ("gemma3-4b", "gated-or-token", True, None),
        ("gemma3-12b", "gated-or-token", True, None),
        ("phi4-14b", "open-or-token", False, "microsoft__phi-4"),
        ("qwen35-27b", "open-or-token", False, "Qwen__Qwen3.5-27B"),
        ("qwen35-08b-base", "open", True, None),
        ("qwen35-2b-base", "open", True, None),
        (
            "latamgpt-70b",
            "gated",
            False,
            "latam-gpt__Llama-3.1-70B-LatamGPT-SFT-1.0",
        ),
    )
}


MODEL_SUITE = [
    _catalog_entry(spec, _MODEL_SUITE_LEGACY_ATTRIBUTES[spec.key])
    for spec in MODEL_SPECS
    if "model-storing" in spec.suites and "vision" not in spec.suites
]

_DISTILLED_ACCESS_BY_KEY = {
    "deepseek-r1-distill-llama-8b": "open",
    "deepseek-r1-distill-qwen-7b": "open",
    "gemma2-9b-distilled": "open-or-token",
    "qwen25-7b-distilled": "open",
}


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
        "name": "Qwen 2.5 7B AWQ",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "parameters_b": 7.0,
        "default_dtype": "auto",
        "representation": "4-bit AWQ",
        "category": "quantized",
        "access": "open",
        "artifact_selector": {
            "kind": "safetensors-index",
            "index": "model.safetensors.index.json",
        },
    },
    {
        "name": "Qwen 2.5 7B GPTQ-Int4",
        "model_id": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "parameters_b": 7.0,
        "default_dtype": "auto",
        "representation": "4-bit GPTQ",
        "category": "quantized",
        "access": "open",
        "artifact_selector": {
            "kind": "safetensors-index",
            "index": "model.safetensors.index.json",
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
    *[
        _catalog_entry(
            spec,
            {
                "default_dtype": "bf16",
                "representation": "distilled",
                "category": "distilled",
                "access": _DISTILLED_ACCESS_BY_KEY[spec.key],
            },
        )
        for spec in MODEL_SPECS
        if "distilled" in spec.suites
    ],
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
