import unittest

import torch
from torch import nn

from neuralzip.gpu_adapters import (
    AdapterNotFoundError,
    AmbiguousAdapterError,
    AmbiguousPatternError,
    AutoModuleListAdapter,
    DuplicateBindingError,
    GPUAdapterRegistry,
    GPUWeightKind,
    ModulePattern,
    NestedBlockError,
    PatternNotFoundError,
    RegexGPUAdapter,
    UnsupportedWeightModuleError,
    default_gpu_adapter_registry,
)


class _StandardBlock(nn.Module):
    def __init__(self, *, embedding: bool = False):
        super().__init__()
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(8, 8, bias=False),
                "k_proj": nn.Linear(8, 4, bias=False),
            }
        )
        self.mlp = nn.Linear(8, 16, bias=False)
        if embedding:
            self.tokens = nn.Embedding(32, 8)


class _StandardModel(nn.Module):
    def __init__(self, *, embedding: bool = False):
        super().__init__()
        self.layers = nn.ModuleList(
            [_StandardBlock(embedding=embedding), _StandardBlock(embedding=embedding)]
        )


class _ExpertsBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)]
        )


class _NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ExpertsBlock(), _ExpertsBlock()])


class _UnusualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.ModuleDict(
            {
                "stage_a": _StandardBlock(embedding=True),
                "stage_b": _StandardBlock(embedding=True),
            }
        )


class _SharedWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        shared = nn.Linear(4, 4, bias=False)
        self.layers[0].left = shared
        self.layers[0].right = shared


class AutoModuleListAdapterTests(unittest.TestCase):
    def test_matches_and_reproduces_legacy_block_and_linear_discovery(self):
        model = _StandardModel()
        adapter = AutoModuleListAdapter()

        self.assertTrue(adapter.matches(model))
        discovery = adapter.discover(model)

        self.assertEqual(discovery.adapter_name, "module-list")
        self.assertEqual(discovery.block_order, ("layers.0", "layers.1"))
        self.assertEqual(
            tuple(binding.tensor_name for binding in discovery.bindings),
            (
                "layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.k_proj.weight",
                "layers.0.mlp.weight",
                "layers.1.self_attn.q_proj.weight",
                "layers.1.self_attn.k_proj.weight",
                "layers.1.mlp.weight",
            ),
        )
        binding = discovery.bindings[0]
        self.assertIs(binding.block, model.layers[0])
        self.assertIs(binding.module, model.layers[0].self_attn["q_proj"])
        self.assertIs(binding.parameter, binding.module.weight)
        self.assertEqual(binding.module_identity, id(binding.module))
        self.assertEqual(binding.parameter_identity, id(binding.parameter))
        self.assertIs(binding.module_type, nn.Linear)
        self.assertEqual(binding.kind, GPUWeightKind.LINEAR)
        self.assertEqual(binding.shape, (8, 8))
        self.assertEqual(binding.dtype, torch.float32)
        self.assertEqual(binding.numel, 64)

    def test_embeddings_are_opt_in_to_preserve_legacy_behavior(self):
        model = _StandardModel(embedding=True)
        without_embeddings = AutoModuleListAdapter().discover(model)
        with_embeddings = AutoModuleListAdapter(include_embeddings=True).discover(model)

        self.assertFalse(
            any(
                binding.kind is GPUWeightKind.EMBEDDING
                for binding in without_embeddings.bindings
            )
        )
        embeddings = [
            binding
            for binding in with_embeddings.bindings
            if binding.kind is GPUWeightKind.EMBEDDING
        ]
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0].tensor_name, "layers.0.tokens.weight")
        self.assertEqual(embeddings[0].shape, (32, 8))
        self.assertIs(embeddings[0].module_type, nn.Embedding)

    def test_nested_module_lists_are_owned_by_the_outer_block(self):
        model = _NestedModel()
        discovery = AutoModuleListAdapter().discover(model)

        self.assertEqual(discovery.block_order, ("layers.0", "layers.1"))
        self.assertEqual(len(discovery.bindings), 4)
        self.assertEqual(
            {binding.block_name for binding in discovery.bindings},
            {"layers.0", "layers.1"},
        )
        self.assertEqual(discovery.bindings[0].tensor_name, "layers.0.experts.0.weight")

    def test_model_without_module_list_does_not_match(self):
        model = nn.Sequential(nn.Linear(4, 4))
        adapter = AutoModuleListAdapter()

        self.assertFalse(adapter.matches(model))
        with self.assertRaisesRegex(PatternNotFoundError, "No direct ModuleList"):
            adapter.discover(model)

    def test_shared_module_alias_is_reported_as_duplicate_binding(self):
        with self.assertRaisesRegex(DuplicateBindingError, "same module"):
            AutoModuleListAdapter().discover(_SharedWeightModel())


class RegexGPUAdapterTests(unittest.TestCase):
    def make_adapter(self, **overrides):
        arguments = {
            "block_patterns": (r"trunk\.stage_[ab]",),
            "weight_patterns": (
                ModulePattern(r"self_attn\.[qk]_proj", label="attention"),
                ModulePattern(r"mlp", label="feed-forward"),
                ModulePattern(r"tokens", label="tokens"),
            ),
        }
        arguments.update(overrides)
        return RegexGPUAdapter("unusual", **arguments)

    def test_explicit_regex_discovers_non_module_list_architecture(self):
        model = _UnusualModel()
        adapter = self.make_adapter()

        self.assertTrue(adapter.matches(model))
        discovery = adapter.discover(model)

        self.assertEqual(discovery.block_order, ("trunk.stage_a", "trunk.stage_b"))
        self.assertEqual(len(discovery.bindings), 8)
        self.assertEqual(
            [binding.kind for binding in discovery.bindings].count(
                GPUWeightKind.EMBEDDING
            ),
            2,
        )
        self.assertTrue(
            all(
                binding.module_name == binding.block_name
                or binding.module_name.startswith(f"{binding.block_name}.")
                for binding in discovery.bindings
            )
        )

    def test_empty_weight_patterns_select_all_supported_descendants(self):
        discovery = self.make_adapter(weight_patterns=()).discover(_UnusualModel())
        self.assertEqual(len(discovery.bindings), 8)

    def test_embeddings_can_be_excluded_explicitly(self):
        adapter = self.make_adapter(
            include_embeddings=False,
            weight_patterns=(r"self_attn\.[qk]_proj", r"mlp"),
        )
        discovery = adapter.discover(_UnusualModel())
        self.assertEqual(len(discovery.bindings), 6)
        self.assertTrue(
            all(binding.kind is GPUWeightKind.LINEAR for binding in discovery.bindings)
        )

    def test_required_block_pattern_reports_missing_module(self):
        adapter = RegexGPUAdapter(
            "missing",
            block_patterns=(ModulePattern(r"decoder\.layers\.\d+", label="decoder"),),
        )
        self.assertFalse(adapter.matches(_UnusualModel()))
        with self.assertRaisesRegex(PatternNotFoundError, "decoder"):
            adapter.discover(_UnusualModel())

    def test_required_weight_pattern_is_checked_in_every_block(self):
        model = _UnusualModel()
        del model.trunk["stage_b"].mlp
        adapter = RegexGPUAdapter(
            "incomplete",
            block_patterns=(r"trunk\.stage_[ab]",),
            weight_patterns=(ModulePattern(r"mlp", label="feed-forward"),),
        )
        with self.assertRaisesRegex(
            PatternNotFoundError, "trunk.stage_b.*feed-forward"
        ):
            adapter.discover(model)

    def test_optional_patterns_may_be_absent(self):
        adapter = RegexGPUAdapter(
            "optional",
            block_patterns=(r"trunk\.stage_[ab]",),
            weight_patterns=(ModulePattern(r"does_not_exist", required=False),),
        )
        discovery = adapter.discover(_UnusualModel())
        self.assertEqual(discovery.bindings, ())

    def test_overlapping_block_patterns_are_rejected(self):
        adapter = RegexGPUAdapter(
            "ambiguous-block",
            block_patterns=(r"trunk\.stage_.*", r"trunk\.stage_[ab]"),
        )
        with self.assertRaisesRegex(AmbiguousPatternError, "multiple patterns"):
            adapter.discover(_UnusualModel())

    def test_overlapping_weight_patterns_are_rejected(self):
        adapter = RegexGPUAdapter(
            "ambiguous-weight",
            block_patterns=(r"trunk\.stage_[ab]",),
            weight_patterns=(r"self_attn\..*", r"self_attn\.q_proj"),
        )
        with self.assertRaisesRegex(AmbiguousPatternError, "multiple patterns"):
            adapter.discover(_UnusualModel())

    def test_pattern_selecting_unsupported_module_fails_loudly(self):
        model = _UnusualModel()
        model.trunk["stage_a"].activation = nn.ReLU()
        model.trunk["stage_b"].activation = nn.ReLU()
        adapter = RegexGPUAdapter(
            "unsupported",
            block_patterns=(r"trunk\.stage_[ab]",),
            weight_patterns=(r"activation",),
        )
        with self.assertRaisesRegex(UnsupportedWeightModuleError, "ReLU"):
            adapter.discover(model)

    def test_explicit_nested_blocks_are_rejected(self):
        adapter = RegexGPUAdapter(
            "nested",
            block_patterns=(r"layers\.0", r"layers\.0\.experts\.0"),
        )
        with self.assertRaisesRegex(NestedBlockError, "nested"):
            adapter.discover(_NestedModel())

    def test_duplicate_aliases_are_rejected(self):
        adapter = RegexGPUAdapter(
            "alias",
            block_patterns=(r"layers\.0",),
        )
        with self.assertRaises(DuplicateBindingError):
            adapter.discover(_SharedWeightModel())

    def test_invalid_and_duplicate_pattern_definitions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid module regex"):
            ModulePattern("[")
        with self.assertRaisesRegex(ValueError, "labels must be unique"):
            RegexGPUAdapter(
                "duplicate-label",
                block_patterns=(
                    ModulePattern(r"a", label="block"),
                    ModulePattern(r"b", label="block"),
                ),
            )


class GPUAdapterRegistryTests(unittest.TestCase):
    def test_explicit_adapter_wins_over_automatic_adapter(self):
        model = _StandardModel()
        explicit = RegexGPUAdapter(
            "standard-explicit",
            block_patterns=(r"layers\.\d+",),
            priority=50,
        )
        registry = GPUAdapterRegistry((AutoModuleListAdapter(), explicit))

        self.assertIs(registry.select(model), explicit)
        self.assertEqual(registry.discover(model).adapter_name, "standard-explicit")
        self.assertEqual(registry.names, ("standard-explicit", "module-list"))

    def test_selection_is_ambiguous_at_equal_priority_regardless_of_order(self):
        model = _StandardModel()
        first = RegexGPUAdapter("alpha", block_patterns=(r"layers\.\d+",))
        second = RegexGPUAdapter("beta", block_patterns=(r"layers\.\d+",))
        for adapters in ((first, second), (second, first)):
            registry = GPUAdapterRegistry(adapters)
            with self.assertRaisesRegex(AmbiguousAdapterError, "'alpha', 'beta'"):
                registry.select(model)

    def test_named_selection_is_deterministic_and_surfaces_adapter_errors(self):
        adapter = RegexGPUAdapter("wrong-model", block_patterns=(r"not\.present",))
        registry = GPUAdapterRegistry((adapter,))

        self.assertIs(registry.select(_StandardModel(), name="wrong-model"), adapter)
        with self.assertRaises(PatternNotFoundError):
            registry.discover(_StandardModel(), name="wrong-model")

    def test_duplicate_names_unknown_names_and_no_match_are_reported(self):
        adapter = AutoModuleListAdapter()
        registry = GPUAdapterRegistry((adapter,))
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(AutoModuleListAdapter())
        with self.assertRaises(AdapterNotFoundError):
            registry.get("missing")
        with self.assertRaises(AdapterNotFoundError):
            registry.select(nn.Linear(2, 2))

    def test_default_registry_instances_do_not_share_mutation(self):
        first = default_gpu_adapter_registry()
        second = default_gpu_adapter_registry()
        first.unregister("module-list")

        self.assertEqual(first.names, ())
        self.assertEqual(second.names, ("module-list",))


if __name__ == "__main__":
    unittest.main()
