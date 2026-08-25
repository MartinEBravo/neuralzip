import hashlib
import json
import unittest
from collections import Counter
from dataclasses import asdict

from examples import model_catalog

EXPECTED_MODEL_KEYS = (
    "gpt2",
    "bert-base",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "qwen25-1p5b",
    "qwen25-3b",
    "qwen25-7b",
    "qwen25-14b",
    "olmo2-1b",
    "olmo2-7b",
    "olmo2-13b",
    "mistral7b",
    "mistral-nemo-12b",
    "llama32-1b",
    "llama32-3b",
    "llama31-nemotron-8b",
    "gemma4-31b",
    "gemma3-1b",
    "gemma3-4b",
    "gemma3-12b",
    "phi4-14b",
    "qwen35-27b",
    "qwen35-08b-base",
    "qwen35-2b-base",
    "latamgpt-70b",
    "convnext-tiny",
    "convnext-small",
    "convnext-base",
    "convnext-large",
    "efficientnet-b0",
    "efficientnet-b3",
    "efficientnet-b7",
    "sd15-unet",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
    "siglip2-giant",
    "siglip2-so400m",
    "zaya1-8b",
    "lfm25-8b-a1b",
    "jetmoe-8b",
    "olmoe-1b-7b-0125",
    "qwen15-moe-a27b",
    "qwen3-30b-a3b",
    "qwen35-35b-a3b",
    "deepseek-moe-16b",
    "mixtral-8x7b",
    "gemma4-26b-a4b",
    "deepseek-r1-distill-llama-8b",
    "deepseek-r1-distill-qwen-7b",
    "gemma2-9b-distilled",
    "qwen25-7b-distilled",
)
EXPECTED_MODEL_SPECS = (
    54,
    "eb37e7b3049cc3be54987d4cc7c9316e7f53246ad59d322ddf883328c88bf2d1",
)


def _digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_preserves_every_spec_and_family_order(self):
        specs = model_catalog.MODEL_SPECS
        self.assertEqual(tuple(spec.key for spec in specs), EXPECTED_MODEL_KEYS)
        self.assertEqual(len(specs), EXPECTED_MODEL_SPECS[0])
        self.assertEqual(
            _digest([asdict(spec) for spec in specs]),
            EXPECTED_MODEL_SPECS[1],
        )
        self.assertEqual(
            Counter(suite for spec in specs for suite in spec.suites),
            {
                "phenomenon": 4,
                "vision": 13,
                "model-storing": 34,
                "moe": 10,
                "distilled": 4,
                "setup-transfer": 4,
            },
        )
        self.assertEqual(tuple(model_catalog.SPECS_BY_KEY), EXPECTED_MODEL_KEYS)
        self.assertTrue(
            all(model_catalog.SPECS_BY_KEY[spec.key] is spec for spec in specs)
        )

    def test_quantized_catalog_has_explicit_artifact_selection(self):
        specs = model_catalog.QUANTIZED_MODEL_SPECS
        self.assertEqual(len(specs), 6)
        self.assertTrue(all(spec["category"] == "quantized" for spec in specs))
        self.assertTrue(all(spec["artifact_selector"] for spec in specs))
        self.assertEqual(
            next(spec for spec in specs if "LatamGPT" in spec["name"])[
                "artifact_selector"
            ],
            {"kind": "exact-files", "files": ("latamgpt-Q4_K_M.gguf",)},
        )


if __name__ == "__main__":
    unittest.main()
