"""Deterministic model adapters for NeuralZip GPU weight discovery.

The GPU runtime originally inferred transformer blocks exclusively from direct
children of :class:`torch.nn.ModuleList`.  This module preserves that behavior
while making discovery explicit and extensible.  Regex adapters cover models
whose block layout cannot be inferred from ``ModuleList`` containers without
teaching the runtime about individual model classes.

Discovery is deliberately side-effect free: adapters return references to
registered modules and parameters, but never move, replace, or detach them.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from torch import nn


class GPUAdapterError(ValueError):
    """Base class for invalid or ambiguous GPU adapter discovery."""


class AdapterNotFoundError(GPUAdapterError):
    """No registered adapter can describe a model."""


class AmbiguousAdapterError(GPUAdapterError):
    """More than one adapter has the same winning priority."""


class PatternNotFoundError(GPUAdapterError):
    """A required block or weight pattern did not match."""


class AmbiguousPatternError(GPUAdapterError):
    """One module matched multiple patterns in the same adapter."""


class NestedBlockError(GPUAdapterError):
    """An explicit adapter selected both a block and one of its descendants."""


class DuplicateBindingError(GPUAdapterError):
    """A module or parameter was selected through more than one path."""


class UnsupportedWeightModuleError(GPUAdapterError):
    """A weight pattern selected a module unsupported by the GPU runtime."""


class GPUWeightKind(str, Enum):
    """Kinds of two-dimensional weights understood by discovery adapters."""

    LINEAR = "linear"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class GPUBlock:
    """A named owner of weights that are materialized together at runtime."""

    name: str
    module: nn.Module = field(repr=False, compare=False)
    module_type: type[nn.Module] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A GPU block name must be a non-empty string.")
        if not isinstance(self.module, nn.Module):
            raise TypeError("A GPU block must reference a torch.nn.Module.")
        object.__setattr__(self, "module_type", type(self.module))

    @property
    def module_identity(self) -> int:
        """Process-local identity used to recognize aliased block paths."""

        return id(self.module)


@dataclass(frozen=True)
class GPUWeightBinding:
    """One discovered two-dimensional weight and its owning GPU block."""

    block_name: str
    block: nn.Module = field(repr=False, compare=False)
    module_name: str
    module: nn.Module = field(repr=False, compare=False)
    tensor_name: str
    parameter: nn.Parameter = field(repr=False, compare=False)
    shape: tuple[int, int]
    kind: GPUWeightKind
    module_type: type[nn.Module] = field(init=False)

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.block_name, self.module_name, self.tensor_name)
        ):
            raise ValueError("Binding names must be non-empty strings.")
        if not isinstance(self.block, nn.Module) or not isinstance(
            self.module, nn.Module
        ):
            raise TypeError("Bindings must reference torch.nn.Module instances.")
        if not isinstance(self.parameter, nn.Parameter):
            raise TypeError("A binding must reference a torch.nn.Parameter.")
        if self.module._parameters.get("weight") is not self.parameter:
            raise ValueError(
                "A binding parameter must be its module's registered weight."
            )
        if self.tensor_name != f"{self.module_name}.weight":
            raise ValueError("tensor_name must identify module_name.weight.")
        if self.module_name != self.block_name and not self.module_name.startswith(
            f"{self.block_name}."
        ):
            raise ValueError("A binding module must be inside its owning block.")

        shape = tuple(self.shape)
        if (
            len(shape) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in shape
            )
            or shape != tuple(self.parameter.shape)
        ):
            raise ValueError(
                "A GPU weight must have its parameter's positive 2-D shape."
            )
        try:
            kind = GPUWeightKind(self.kind)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported GPU weight kind: {self.kind!r}.") from error
        if kind is GPUWeightKind.LINEAR and not isinstance(self.module, nn.Linear):
            raise TypeError("A linear binding must reference torch.nn.Linear.")
        if kind is GPUWeightKind.EMBEDDING and not isinstance(
            self.module, nn.Embedding
        ):
            raise TypeError("An embedding binding must reference torch.nn.Embedding.")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "module_type", type(self.module))

    @property
    def module_identity(self) -> int:
        """Process-local identity of the selected module."""

        return id(self.module)

    @property
    def parameter_identity(self) -> int:
        """Process-local identity of the selected parameter."""

        return id(self.parameter)

    @property
    def dtype(self):
        return self.parameter.dtype

    @property
    def requires_grad(self) -> bool:
        return self.parameter.requires_grad

    @property
    def numel(self) -> int:
        return self.parameter.numel()


@dataclass(frozen=True)
class GPUDiscovery:
    """Validated block and weight references returned by one adapter."""

    adapter_name: str
    blocks: tuple[GPUBlock, ...]
    bindings: tuple[GPUWeightBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.adapter_name, str) or not self.adapter_name:
            raise ValueError("adapter_name must be a non-empty string.")
        blocks = tuple(self.blocks)
        bindings = tuple(self.bindings)
        if any(not isinstance(block, GPUBlock) for block in blocks):
            raise TypeError("blocks must contain GPUBlock values.")
        if any(not isinstance(binding, GPUWeightBinding) for binding in bindings):
            raise TypeError("bindings must contain GPUWeightBinding values.")

        by_name: dict[str, GPUBlock] = {}
        block_identities: dict[int, str] = {}
        for block in blocks:
            if block.name in by_name:
                raise DuplicateBindingError(f"Duplicate GPU block name: {block.name}.")
            previous_name = block_identities.setdefault(
                block.module_identity, block.name
            )
            if previous_name != block.name:
                raise DuplicateBindingError(
                    f"GPU block {block.name!r} aliases block {previous_name!r}."
                )
            by_name[block.name] = block

        tensor_names: set[str] = set()
        module_identities: dict[int, str] = {}
        parameter_identities: dict[int, str] = {}
        for binding in bindings:
            owner = by_name.get(binding.block_name)
            if owner is None or owner.module is not binding.block:
                raise GPUAdapterError(
                    f"Binding {binding.tensor_name!r} has an unknown block owner."
                )
            if binding.tensor_name in tensor_names:
                raise DuplicateBindingError(
                    f"Duplicate GPU tensor binding: {binding.tensor_name}."
                )
            tensor_names.add(binding.tensor_name)
            previous_module = module_identities.setdefault(
                binding.module_identity, binding.tensor_name
            )
            if previous_module != binding.tensor_name:
                raise DuplicateBindingError(
                    f"GPU tensors {previous_module!r} and {binding.tensor_name!r} "
                    "reference the same module."
                )
            previous_parameter = parameter_identities.setdefault(
                binding.parameter_identity, binding.tensor_name
            )
            if previous_parameter != binding.tensor_name:
                raise DuplicateBindingError(
                    f"GPU tensors {previous_parameter!r} and {binding.tensor_name!r} "
                    "reference the same parameter."
                )

        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "bindings", bindings)

    @property
    def block_order(self) -> tuple[str, ...]:
        return tuple(block.name for block in self.blocks)


class GPUModelAdapter(Protocol):
    """Structural interface accepted by :class:`GPUAdapterRegistry`."""

    @property
    def name(self) -> str: ...

    @property
    def priority(self) -> int: ...

    def matches(self, model: nn.Module) -> bool: ...

    def discover(self, model: nn.Module) -> GPUDiscovery: ...


class BaseGPUModelAdapter(ABC):
    """Nominal base class for built-in and third-party model adapters."""

    def __init__(self, name: str, *, priority: int) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("An adapter name must be a non-empty string.")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError("Adapter priority must be an integer.")
        self._name = name
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @abstractmethod
    def matches(self, model: nn.Module) -> bool:
        """Return whether this adapter can describe ``model``."""

    @abstractmethod
    def discover(self, model: nn.Module) -> GPUDiscovery:
        """Return validated blocks and weight bindings without mutation."""


def _require_model(model: nn.Module) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")


def _join_name(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _named_modules_with_aliases(module: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield every registered module path, retaining aliases."""

    return module.named_modules(remove_duplicate=False)


def _kind_for_module(
    module: nn.Module,
    *,
    include_embeddings: bool,
) -> GPUWeightKind | None:
    if isinstance(module, nn.Linear):
        return GPUWeightKind.LINEAR
    if include_embeddings and isinstance(module, nn.Embedding):
        return GPUWeightKind.EMBEDDING
    return None


def _make_binding(
    block: GPUBlock,
    module_name: str,
    module: nn.Module,
    kind: GPUWeightKind,
) -> GPUWeightBinding:
    parameter = module._parameters.get("weight")
    if not isinstance(parameter, nn.Parameter):
        raise UnsupportedWeightModuleError(
            f"Module {module_name!r} has no registered weight parameter."
        )
    return GPUWeightBinding(
        block_name=block.name,
        block=block.module,
        module_name=module_name,
        module=module,
        tensor_name=f"{module_name}.weight",
        parameter=parameter,
        shape=tuple(parameter.shape),
        kind=kind,
    )


def _automatic_blocks(model: nn.Module) -> tuple[GPUBlock, ...]:
    candidates: list[tuple[str, nn.Module]] = []
    for parent_name, parent in _named_modules_with_aliases(model):
        if not isinstance(parent, nn.ModuleList):
            continue
        for child_name, child in parent._modules.items():
            if child is not None:
                candidates.append((_join_name(parent_name, child_name), child))

    # Match the original runtime: a ModuleList inside a selected block (for
    # example, a list of MoE experts) does not establish nested workspaces.
    candidate_names = {name for name, _ in candidates}
    outermost: list[GPUBlock] = []
    for name, module in candidates:
        ancestor = name
        while "." in ancestor:
            ancestor = ancestor.rsplit(".", 1)[0]
            if ancestor in candidate_names:
                break
        else:
            outermost.append(GPUBlock(name=name, module=module))
    return tuple(outermost)


class AutoModuleListAdapter(BaseGPUModelAdapter):
    """Discover candidate blocks among direct children of ``ModuleList``."""

    def __init__(
        self,
        *,
        include_embeddings: bool = False,
        name: str = "module-list",
        priority: int = -100,
    ) -> None:
        super().__init__(name, priority=priority)
        if not isinstance(include_embeddings, bool):
            raise TypeError("include_embeddings must be a boolean.")
        self.include_embeddings = include_embeddings

    def matches(self, model: nn.Module) -> bool:
        _require_model(model)
        return bool(_automatic_blocks(model))

    def discover(self, model: nn.Module) -> GPUDiscovery:
        _require_model(model)
        blocks = _automatic_blocks(model)
        if not blocks:
            raise PatternNotFoundError(
                "No direct ModuleList block children were found in the model."
            )
        bindings: list[GPUWeightBinding] = []
        for block in blocks:
            for relative_name, module in _named_modules_with_aliases(block.module):
                kind = _kind_for_module(
                    module, include_embeddings=self.include_embeddings
                )
                if kind is None:
                    continue
                bindings.append(
                    _make_binding(
                        block,
                        _join_name(block.name, relative_name),
                        module,
                        kind,
                    )
                )
        return GPUDiscovery(self.name, blocks, tuple(bindings))


@dataclass(frozen=True)
class ModulePattern:
    """A named full-match regular expression used by an explicit adapter."""

    pattern: str
    label: str | None = None
    required: bool = True
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ValueError("A module pattern must be a non-empty string.")
        if self.label is not None and (
            not isinstance(self.label, str) or not self.label
        ):
            raise ValueError("A pattern label must be None or a non-empty string.")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean.")
        try:
            compiled = re.compile(self.pattern)
        except re.error as error:
            raise ValueError(
                f"Invalid module regex {self.pattern!r}: {error}."
            ) from error
        object.__setattr__(self, "_compiled", compiled)

    @property
    def display_name(self) -> str:
        return self.label or self.pattern

    def matches(self, module_name: str) -> bool:
        return self._compiled.fullmatch(module_name) is not None


def _coerce_patterns(
    values: Iterable[str | ModulePattern],
    *,
    argument: str,
    allow_empty: bool,
) -> tuple[ModulePattern, ...]:
    try:
        source = tuple(values)
    except TypeError as error:
        raise TypeError(f"{argument} must be an iterable of regex patterns.") from error
    patterns = tuple(
        value if isinstance(value, ModulePattern) else ModulePattern(value)
        for value in source
    )
    if not allow_empty and not patterns:
        raise ValueError(f"{argument} must contain at least one pattern.")
    labels = [pattern.display_name for pattern in patterns]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{argument} pattern labels must be unique.")
    return patterns


def _matching_patterns(
    module_name: str, patterns: tuple[ModulePattern, ...]
) -> tuple[ModulePattern, ...]:
    return tuple(pattern for pattern in patterns if pattern.matches(module_name))


def _raise_ambiguous_pattern(
    *, module_name: str, matches: tuple[ModulePattern, ...], kind: str
) -> None:
    labels = ", ".join(repr(pattern.display_name) for pattern in matches)
    raise AmbiguousPatternError(
        f"{kind.capitalize()} module {module_name!r} matches multiple patterns: {labels}."
    )


def _validate_no_nested_blocks(blocks: tuple[GPUBlock, ...]) -> None:
    names = {block.name for block in blocks}
    for name in sorted(names):
        ancestor = name
        while "." in ancestor:
            ancestor = ancestor.rsplit(".", 1)[0]
            if ancestor in names:
                raise NestedBlockError(
                    f"Explicit blocks {ancestor!r} and {name!r} are nested."
                )


class RegexGPUAdapter(BaseGPUModelAdapter):
    """Explicit architecture adapter based on full-match module regexes.

    Block patterns match absolute names from ``model.named_modules()``.  Weight
    patterns match names relative to each selected block.  Every required
    weight pattern must match at least once *within every block*, which catches
    incomplete or subtly different layers before an archive is created.  With
    no weight patterns, all supported descendants are selected.
    """

    def __init__(
        self,
        name: str,
        *,
        block_patterns: Iterable[str | ModulePattern],
        weight_patterns: Iterable[str | ModulePattern] = (),
        include_embeddings: bool = True,
        priority: int = 100,
    ) -> None:
        super().__init__(name, priority=priority)
        self.block_patterns = _coerce_patterns(
            block_patterns, argument="block_patterns", allow_empty=False
        )
        self.weight_patterns = _coerce_patterns(
            weight_patterns, argument="weight_patterns", allow_empty=True
        )
        if not isinstance(include_embeddings, bool):
            raise TypeError("include_embeddings must be a boolean.")
        self.include_embeddings = include_embeddings

    def _block_matches(
        self, model: nn.Module
    ) -> tuple[tuple[GPUBlock, ModulePattern], ...]:
        selected: list[tuple[GPUBlock, ModulePattern]] = []
        for module_name, module in _named_modules_with_aliases(model):
            if not module_name:
                continue
            matches = _matching_patterns(module_name, self.block_patterns)
            if len(matches) > 1:
                _raise_ambiguous_pattern(
                    module_name=module_name, matches=matches, kind="block"
                )
            if matches:
                selected.append((GPUBlock(module_name, module), matches[0]))
        return tuple(selected)

    def matches(self, model: nn.Module) -> bool:
        _require_model(model)
        selected = self._block_matches(model)
        counts = {id(pattern): 0 for pattern in self.block_patterns}
        for _block, pattern in selected:
            counts[id(pattern)] += 1
        return bool(selected) and all(
            not pattern.required or counts[id(pattern)] > 0
            for pattern in self.block_patterns
        )

    def discover(self, model: nn.Module) -> GPUDiscovery:
        _require_model(model)
        selected = self._block_matches(model)
        counts = {id(pattern): 0 for pattern in self.block_patterns}
        for _block, pattern in selected:
            counts[id(pattern)] += 1
        missing = [
            pattern.display_name
            for pattern in self.block_patterns
            if pattern.required and counts[id(pattern)] == 0
        ]
        if missing:
            raise PatternNotFoundError(
                "Required block patterns matched no modules: "
                + ", ".join(repr(value) for value in missing)
                + "."
            )
        if not selected:
            raise PatternNotFoundError("The adapter selected no GPU blocks.")

        blocks = tuple(block for block, _pattern in selected)
        _validate_no_nested_blocks(blocks)
        bindings: list[GPUWeightBinding] = []
        for block in blocks:
            pattern_counts = {id(pattern): 0 for pattern in self.weight_patterns}
            for relative_name, module in _named_modules_with_aliases(block.module):
                if not relative_name and not isinstance(
                    module, (nn.Linear, nn.Embedding)
                ):
                    continue
                matches = _matching_patterns(relative_name, self.weight_patterns)
                if len(matches) > 1:
                    _raise_ambiguous_pattern(
                        module_name=_join_name(block.name, relative_name),
                        matches=matches,
                        kind="weight",
                    )
                if self.weight_patterns and not matches:
                    continue

                module_name = _join_name(block.name, relative_name)
                kind = _kind_for_module(
                    module, include_embeddings=self.include_embeddings
                )
                if kind is None:
                    if matches:
                        raise UnsupportedWeightModuleError(
                            f"Weight pattern {matches[0].display_name!r} selected "
                            f"unsupported module {module_name!r} ({type(module).__name__})."
                        )
                    continue
                if matches:
                    pattern_counts[id(matches[0])] += 1
                bindings.append(_make_binding(block, module_name, module, kind))

            missing_weights = [
                pattern.display_name
                for pattern in self.weight_patterns
                if pattern.required and pattern_counts[id(pattern)] == 0
            ]
            if missing_weights:
                raise PatternNotFoundError(
                    f"GPU block {block.name!r} is missing required weight patterns: "
                    + ", ".join(repr(value) for value in missing_weights)
                    + "."
                )

        return GPUDiscovery(self.name, blocks, tuple(bindings))


class GPUAdapterRegistry:
    """Priority-based registry whose result is independent of insertion order."""

    def __init__(self, adapters: Iterable[GPUModelAdapter] = ()) -> None:
        self._adapters: dict[str, GPUModelAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    @staticmethod
    def _validate_adapter(adapter: GPUModelAdapter) -> None:
        name = getattr(adapter, "name", None)
        priority = getattr(adapter, "priority", None)
        if not isinstance(name, str) or not name:
            raise TypeError("An adapter must expose a non-empty string name.")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise TypeError("An adapter must expose an integer priority.")
        if not callable(getattr(adapter, "matches", None)) or not callable(
            getattr(adapter, "discover", None)
        ):
            raise TypeError("An adapter must implement matches() and discover().")

    def register(
        self, adapter: GPUModelAdapter, *, replace: bool = False
    ) -> GPUModelAdapter:
        self._validate_adapter(adapter)
        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean.")
        if adapter.name in self._adapters and not replace:
            raise ValueError(f"GPU adapter {adapter.name!r} is already registered.")
        self._adapters[adapter.name] = adapter
        return adapter

    def unregister(self, name: str) -> GPUModelAdapter:
        try:
            return self._adapters.pop(name)
        except KeyError as error:
            raise AdapterNotFoundError(f"Unknown GPU adapter: {name!r}.") from error

    @property
    def adapters(self) -> tuple[GPUModelAdapter, ...]:
        return tuple(
            sorted(
                self._adapters.values(),
                key=lambda adapter: (-adapter.priority, adapter.name),
            )
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(adapter.name for adapter in self.adapters)

    def get(self, name: str) -> GPUModelAdapter:
        try:
            return self._adapters[name]
        except KeyError as error:
            raise AdapterNotFoundError(f"Unknown GPU adapter: {name!r}.") from error

    def select(self, model: nn.Module, *, name: str | None = None) -> GPUModelAdapter:
        _require_model(model)
        if name is not None:
            return self.get(name)

        matching = [adapter for adapter in self.adapters if adapter.matches(model)]
        if not matching:
            raise AdapterNotFoundError("No registered GPU adapter matches the model.")
        winning_priority = matching[0].priority
        winners = [
            adapter for adapter in matching if adapter.priority == winning_priority
        ]
        if len(winners) > 1:
            names = ", ".join(repr(adapter.name) for adapter in winners)
            raise AmbiguousAdapterError(
                f"Multiple GPU adapters match at priority {winning_priority}: {names}."
            )
        return winners[0]

    def discover(self, model: nn.Module, *, name: str | None = None) -> GPUDiscovery:
        return self.select(model, name=name).discover(model)


__all__ = [
    "AdapterNotFoundError",
    "AmbiguousAdapterError",
    "AmbiguousPatternError",
    "AutoModuleListAdapter",
    "BaseGPUModelAdapter",
    "DuplicateBindingError",
    "GPUAdapterError",
    "GPUAdapterRegistry",
    "GPUBlock",
    "GPUDiscovery",
    "GPUModelAdapter",
    "GPUWeightBinding",
    "GPUWeightKind",
    "ModulePattern",
    "NestedBlockError",
    "PatternNotFoundError",
    "RegexGPUAdapter",
    "UnsupportedWeightModuleError",
]
