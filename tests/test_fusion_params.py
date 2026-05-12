"""Unit tests for :meth:`forestseg.fusion.FusionParams.from_mapping`.

The factory consolidates four previously-duplicated construction
sites (``cli._load_selected_params``, ``fusion.run_stage0``, two
branches of ``rl_policy.choose_fusion_by_validation``); these tests
lock in its observable contract: per-key fallback order, numeric
coercion, and field-typing.
"""

from __future__ import annotations

import pytest

from forestseg.fusion import FusionParams


def test_from_mapping_with_full_mapping_round_trips() -> None:
    mapping = {
        "lambda_spec": 0.1,
        "lambda_tex": 0.2,
        "threshold": 0.6,
        "min_area_m2": 150.0,
        "morph_kernel": 5,
        "shadow_penalty": 0.3,
    }
    params = FusionParams.from_mapping(mapping)
    assert params == FusionParams(
        lambda_spec=0.1,
        lambda_tex=0.2,
        threshold=0.6,
        min_area_m2=150.0,
        morph_kernel=5,
        shadow_penalty=0.3,
    )


def test_from_mapping_with_empty_mapping_uses_built_in_defaults() -> None:
    params = FusionParams.from_mapping({})
    assert params == FusionParams(
        lambda_spec=0.2,
        lambda_tex=0.2,
        threshold=0.5,
        min_area_m2=200.0,
        morph_kernel=3,
        shadow_penalty=0.5,
    )


def test_from_mapping_with_none_uses_built_in_defaults() -> None:
    params = FusionParams.from_mapping(None)
    assert params.lambda_spec == pytest.approx(0.2)
    assert params.morph_kernel == 3


def test_from_mapping_prefers_mapping_over_defaults() -> None:
    """When both ``mapping`` and ``defaults`` carry a key, ``mapping`` wins."""
    params = FusionParams.from_mapping(
        {"lambda_spec": 0.4, "threshold": 0.7},
        defaults={"lambda_spec": 0.9, "threshold": 0.1, "morph_kernel": 5},
    )
    assert params.lambda_spec == pytest.approx(0.4)
    assert params.threshold == pytest.approx(0.7)
    # `morph_kernel` only present in `defaults` — must come from there.
    assert params.morph_kernel == 5


def test_from_mapping_coerces_numeric_strings() -> None:
    params = FusionParams.from_mapping(
        {
            "lambda_spec": "0.15",
            "lambda_tex": "0.25",
            "threshold": "0.55",
            "min_area_m2": "180",
            "morph_kernel": "4",
            "shadow_penalty": "0.4",
        }
    )
    assert params.lambda_spec == pytest.approx(0.15)
    assert params.morph_kernel == 4
    assert isinstance(params.morph_kernel, int)


def test_from_mapping_rejects_non_numeric_value() -> None:
    with pytest.raises(ValueError):
        FusionParams.from_mapping({"threshold": "not-a-number"})
