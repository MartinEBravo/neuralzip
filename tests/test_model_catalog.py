import hashlib
import json
import os
import subprocess
import sys
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from examples import model_catalog
import examples.run_general_codec_rerun as general_runner


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_KEYS = tuple(
    """
    gpt2 bert-base resnet18 resnet34 resnet50 resnet101 resnet152
    qwen25-15b qwen25-3b qwen25-7b qwen25-14b olmo2-1b olmo2-7b
    olmo2-13b mistral7b mistral-nemo-12b llama32-1b llama32-3b
    llama31-nemotron-8b llama33-70b gemma4-31b gemma3-1b gemma3-4b
    gemma3-12b phi4-14b qwen35-27b qwen35-08b-base qwen35-2b-base
    latamgpt-70b convnext-tiny convnext-small convnext-base convnext-large
    efficientnet-b0 efficientnet-b3 efficientnet-b7 sd15-unet zaya1-8b
    lfm25-8b-a1b jetmoe-8b olmoe-1b-7b-0125 qwen15-moe-a27b
    qwen3-30b-a3b qwen35-35b-a3b deepseek-moe-16b mixtral-8x7b
    gemma4-26b-a4b deepseek-r1-distill-llama-8b
    deepseek-r1-distill-qwen-7b gemma2-9b-distilled
    qwen25-7b-distilled
    """.split()
)
EXPECTED_FAMILY_ORDER = {
    "ResNet": 0,
    "ConvNeXt": 1,
    "EfficientNet": 2,
    "Qwen2.5": 3,
    "OLMo 2": 4,
    "Mistral": 5,
    "Stable Diffusion": 6,
    "Phi 4": 7,
    "Qwen 3.5": 8,
    "Gemma 3": 9,
    "Gemma 4": 9,
    "Llama 3": 10,
    "Llama 3 / derivatives": 10,
    "Llama 3 / LatamGPT": 11,
}
EXPECTED_MODEL_SPECS = (
    51,
    "2e1c215e1de86d7c100c3fc74dae6b6f782871f6732525c14bcdb73ef0d34ca5",
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
                "model-storing": 35,
                "moe": 10,
                "distilled": 4,
            },
        )
        self.assertEqual(
            model_catalog.MODEL_STORAGE_FAMILY_ORDER,
            EXPECTED_FAMILY_ORDER,
        )
        self.assertEqual(tuple(model_catalog.SPECS_BY_KEY), EXPECTED_MODEL_KEYS)
        self.assertTrue(
            all(
                model_catalog.SPECS_BY_KEY[spec.key] is spec
                for spec in specs
            )
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

    def test_general_runner_reexports_canonical_catalog(self):
        self.assertIs(general_runner.ModelSpec, model_catalog.ModelSpec)
        self.assertIs(general_runner.MODEL_SPECS, model_catalog.MODEL_SPECS)
        self.assertIs(general_runner.SPECS_BY_KEY, model_catalog.SPECS_BY_KEY)
        self.assertIs(
            general_runner.MODEL_STORAGE_FAMILY_ORDER,
            model_catalog.MODEL_STORAGE_FAMILY_ORDER,
        )

    def test_direct_script_import_path_lists_the_same_catalog(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (str(REPOSITORY_ROOT), environment.get("PYTHONPATH")),
            )
        )
        result = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "examples" / "run_general_codec_rerun.py"),
                "--list",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), len(model_catalog.MODEL_SPECS))
        self.assertEqual(
            lines,
            [
                f"{spec.key}\t{spec.label}\t{spec.family}\t"
                f"{','.join(spec.suites)}\t{spec.source}"
                for spec in model_catalog.MODEL_SPECS
            ],
        )


if __name__ == "__main__":
    unittest.main()
