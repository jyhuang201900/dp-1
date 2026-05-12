"""Sanity tests for :mod:`forestseg._constants`.

These assertions lock in invariants that the rest of the codebase
silently relies on — most importantly the *partition* relationship
between :data:`REQUIRED_FUSION_GRID_KEYS` and
:data:`FUSION_POSTPROCESS_PARAM_KEYS` (they jointly cover the full
:data:`REQUIRED_FUSION_FIXED_PARAM_KEYS` set, with no overlap). The
rl-policy fallback path in :func:`forestseg.rl_policy.choose_fusion_by_validation`
merges these two dicts via ``{**a, **b}``; if they ever overlap, one
side would silently shadow the other.
"""

from __future__ import annotations

from forestseg._constants import (
    FUSION_POSTPROCESS_PARAM_KEYS,
    REQUIRED_FUSION_FIXED_PARAM_KEYS,
    REQUIRED_FUSION_GRID_KEYS,
)


def test_grid_and_postprocess_keys_partition_fixed_param_keys() -> None:
    grid = set(REQUIRED_FUSION_GRID_KEYS)
    post = set(FUSION_POSTPROCESS_PARAM_KEYS)
    fixed = set(REQUIRED_FUSION_FIXED_PARAM_KEYS)
    assert grid.isdisjoint(post), f"grid/post keys overlap: {grid & post!r}"
    assert grid | post == fixed, f"grid ∪ post != fixed: {(grid | post) ^ fixed!r}"


def test_required_fusion_grid_keys_are_a_strict_prefix_of_fixed_param_keys() -> None:
    n = len(REQUIRED_FUSION_GRID_KEYS)
    assert REQUIRED_FUSION_FIXED_PARAM_KEYS[:n] == REQUIRED_FUSION_GRID_KEYS


def test_postprocess_keys_appear_in_fixed_param_keys_in_declared_order() -> None:
    """``FUSION_POSTPROCESS_PARAM_KEYS`` preserves the declaration order
    from :data:`REQUIRED_FUSION_FIXED_PARAM_KEYS` — this keeps the dict
    iteration order deterministic for callers that materialize it via
    ``{k: ... for k in FUSION_POSTPROCESS_PARAM_KEYS}``.
    """
    fixed = list(REQUIRED_FUSION_FIXED_PARAM_KEYS)
    post = list(FUSION_POSTPROCESS_PARAM_KEYS)
    fixed_indices = [fixed.index(k) for k in post]
    assert fixed_indices == sorted(fixed_indices)
