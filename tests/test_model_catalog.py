import hashlib
import json
import os
import subprocess
import sys
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from examples import config, model_catalog
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
EXPECTED_SNAPSHOTS = {
    "MODEL_SPECS": (51, "2e1c215e1de86d7c100c3fc74dae6b6f782871f6732525c14bcdb73ef0d34ca5"),
    "MODEL_SUITE": (22, "25fd794519b7bafefedcf1eab3d8931f8c2e382069b0a7c1fb7f90dbb7fb4930"),
    "LOSSY_LOSSLESS_MODEL_SUITE": (10, "a2fa96dfad5ef681cbf42fe8c27a8eeeb2da8d14a4329f5d13764f9e0dd59733"),
    "VISION_DIFFUSION_MODEL_SUITE": (6, "1bc11c6068c9cafd4c87eaeb7c425e556a335c1dab3f8b70a5ee8d3e587bfeb4"),
}


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
        self.assertEqual(len(specs), EXPECTED_SNAPSHOTS["MODEL_SPECS"][0])
        self.assertEqual(
            _digest([asdict(spec) for spec in specs]),
            EXPECTED_SNAPSHOTS["MODEL_SPECS"][1],
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

    def test_legacy_suite_dicts_are_exact_snapshots(self):
        for name, (expected_count, expected_digest) in EXPECTED_SNAPSHOTS.items():
            if name == "MODEL_SPECS":
                continue
            with self.subTest(suite=name):
                suite = getattr(config, name)
                self.assertEqual(len(suite), expected_count)
                self.assertEqual(_digest(suite), expected_digest)
                self.assertTrue(all(isinstance(entry, dict) for entry in suite))

    def test_catalog_projections_preserve_names_ids_parameters_and_categories(self):
        model_specs = tuple(
            spec
            for spec in model_catalog.MODEL_SPECS
            if "model-storing" in spec.suites and "vision" not in spec.suites
        )
        distilled_specs = tuple(
            spec
            for spec in model_catalog.MODEL_SPECS
            if "distilled" in spec.suites
        )

        for specs, entries in (
            (model_specs, config.MODEL_SUITE),
            (distilled_specs, config.LOSSY_LOSSLESS_MODEL_SUITE[-4:]),
        ):
            self.assertEqual(
                tuple((entry["name"], entry["model_id"], entry["parameters_b"]) for entry in entries),
                tuple((spec.label, spec.source, spec.parameters_b) for spec in specs),
            )

        self.assertEqual(
            tuple(entry.get("category") for entry in config.MODEL_SUITE),
            (
                None,
                None,
                "model-storing",
                "model-storing",
                None,
                None,
                "model-storing",
                None,
                "model-storing",
                "model-storing",
                "model-storing",
                "model-storing",
                None,
                None,
                "model-storing",
                "model-storing",
                "model-storing",
                None,
                None,
                "model-storing",
                "model-storing",
                None,
            ),
        )
        self.assertEqual(
            tuple(entry["category"] for entry in config.LOSSY_LOSSLESS_MODEL_SUITE),
            ("quantized",) * 6 + ("distilled",) * 4,
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
