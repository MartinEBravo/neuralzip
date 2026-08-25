"""Pure policies for ratio-aware and time-budgeted codec selection.

The functions in this module deliberately do not measure wall time themselves.
Callers must benchmark candidates on the same representative input and pass
cost/time estimators.  Keeping measurement outside the policy makes selection
deterministic in tests.

``ratio_tolerance`` is a relative compressed-size tolerance.  For example,
``0.02`` permits a candidate whose estimated compressed bytes are at most 2%
larger than the smallest candidate.  It is not a percentage-point tolerance.
"""

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

DEFAULT_KIND_PRIORITY = (
    "shared_huff0",
    "packed_huff0",
    "local_huff0",
    "raw",
)


def _finite_nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return number


def _kind(mode: Mapping[str, Any]) -> str:
    return str(mode.get("kind", ""))


def select_fastest_within_ratio(
    candidates: Iterable[Mapping[str, Any]],
    *,
    estimated_bytes: Callable[[Mapping[str, Any]], float],
    estimated_seconds: Callable[[Mapping[str, Any]], float],
    ratio_tolerance: float = 0.0,
) -> Mapping[str, Any]:
    """Select the fastest mode whose size is close enough to the best ratio.

    The smallest estimated byte count defines the ratio baseline.  Candidates
    within ``best_bytes * (1 + ratio_tolerance)`` are eligible, and the fastest
    eligible candidate wins.  Remaining ties use byte count and then
    a fixed mode priority so the result is stable across input ordering.

    This policy lets a shared representation replace
    ``local_huff0`` when it is faster and its ratio regression is acceptable.
    """

    tolerance = _finite_nonnegative(ratio_tolerance, "ratio_tolerance")
    modes = tuple(candidates)
    if not modes:
        raise ValueError("candidates must not be empty.")

    priority = {kind: index for index, kind in enumerate(DEFAULT_KIND_PRIORITY)}
    scored = []
    for mode in modes:
        byte_count = _finite_nonnegative(
            estimated_bytes(mode),
            f"estimated bytes for {_kind(mode)!r}",
        )
        seconds = _finite_nonnegative(
            estimated_seconds(mode),
            f"estimated seconds for {_kind(mode)!r}",
        )
        scored.append((mode, byte_count, seconds))

    ratio_best = min(
        scored,
        key=lambda item: (
            item[1],
            priority.get(_kind(item[0]), len(priority)),
            _kind(item[0]),
        ),
    )
    best_bytes = ratio_best[1]
    byte_limit = best_bytes * (1.0 + tolerance)
    eligible = [item for item in scored if item[1] <= byte_limit]
    selected = min(
        eligible,
        key=lambda item: (
            item[2],
            item[1],
            priority.get(_kind(item[0]), len(priority)),
            _kind(item[0]),
        ),
    )

    return selected[0]


def select_packed_time_budget(
    evaluations: Iterable[Mapping[str, Any]],
    *,
    packed_time_budget_ratio: float,
) -> frozenset[Any]:
    """Restore the historical global packed-time-budget policy.

    Each evaluation must contain ``key`` and ``base_seconds``.  Its optional
    ``opportunity`` mapping contains ``saved_bytes`` and ``extra_seconds``.
    Useful opportunities are ordered by estimated bytes saved per extra second
    and admitted greedily while their cumulative extra time fits within
    ``sum(base_seconds) * packed_time_budget_ratio``.

    A zero budget disables packed selection entirely, matching the historical
    codec even for opportunities reporting zero extra seconds.
    """

    budget_ratio = _finite_nonnegative(
        packed_time_budget_ratio,
        "packed_time_budget_ratio",
    )
    rows = tuple(evaluations)
    base_seconds = sum(
        _finite_nonnegative(row.get("base_seconds", 0.0), "base_seconds")
        for row in rows
    )
    budget_seconds = base_seconds * budget_ratio

    opportunities = []
    for row in rows:
        opportunity = row.get("opportunity")
        if opportunity is None:
            continue
        saved_bytes = _finite_nonnegative(
            opportunity.get("saved_bytes", 0.0),
            "packed saved_bytes",
        )
        raw_extra_seconds = float(opportunity.get("extra_seconds", 0.0))
        if not math.isfinite(raw_extra_seconds):
            raise ValueError("packed extra_seconds must be finite.")
        extra_seconds = max(0.0, raw_extra_seconds)
        if saved_bytes <= 0:
            continue
        savings_per_second = (
            math.inf if extra_seconds == 0 else saved_bytes / extra_seconds
        )
        opportunities.append(
            (
                row.get("key"),
                saved_bytes,
                extra_seconds,
                savings_per_second,
            )
        )

    opportunities.sort(key=lambda item: (-item[3], repr(item[0])))
    selected_keys = set()
    used_seconds = 0.0
    if budget_ratio > 0:
        for key, _, extra_seconds, _ in opportunities:
            if extra_seconds > 0 and used_seconds + extra_seconds > budget_seconds:
                continue
            selected_keys.add(key)
            used_seconds += extra_seconds
    return frozenset(selected_keys)
