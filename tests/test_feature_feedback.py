"""Unit tests for :mod:`forestseg._feature_feedback`.

Locks in the loader's defensive contract: when the feedback artefact
is missing the transforms list is all-``None``; when present it must
be valid JSON, an object, and carry finite non-negative numeric
fields — otherwise the loader raises rather than silently
substituting defaults.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from forestseg.core.constants import wf
from forestseg.features.transforms import _feature_feedback_transforms


def _write_feedback(work: Path, payload: object) -> None:
    target = Path(wf(str(work), "feature_feedback"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_returns_all_none_when_feedback_missing(tmp_path: Path) -> None:
    transforms = _feature_feedback_transforms(str(tmp_path))
    assert transforms == [None, None, None, None]


def test_returns_spec_and_tex_transforms_for_valid_feedback(tmp_path: Path) -> None:
    _write_feedback(
        tmp_path,
        {
            "lambda_spec": 0.4,
            "lambda_tex": 0.1,
            "threshold": 0.6,
            "reward": 1.5,
        },
    )
    transforms = _feature_feedback_transforms(str(tmp_path))
    assert transforms[0] is None
    assert transforms[3] is None
    spec = transforms[1]
    tex = transforms[2]
    assert spec is not None and tex is not None
    assert spec["threshold"] == pytest.approx(0.6)
    assert spec["scale"] == pytest.approx(0.4 / 0.2)
    assert tex["scale"] == pytest.approx(0.1 / 0.2)
    # 0 <= reward_gain <= 1, so offset stays within [0, 0.05].
    assert 0.0 <= spec["offset"] <= 0.05


def test_rejects_malformed_json(tmp_path: Path) -> None:
    target = Path(wf(str(tmp_path), "feature_feedback"))
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="feature_feedback"):
        _feature_feedback_transforms(str(tmp_path))


def test_rejects_non_object_payload(tmp_path: Path) -> None:
    _write_feedback(tmp_path, [1, 2, 3])
    with pytest.raises(ValueError, match="feature_feedback"):
        _feature_feedback_transforms(str(tmp_path))


def test_rejects_non_finite_threshold(tmp_path: Path) -> None:
    target = Path(wf(str(tmp_path), "feature_feedback"))
    # NaN can't be expressed in standard JSON; write it raw.
    target.write_text('{"threshold": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="feature_feedback.threshold"):
        _feature_feedback_transforms(str(tmp_path))


def test_rejects_negative_lambda(tmp_path: Path) -> None:
    _write_feedback(tmp_path, {"lambda_spec": -0.1})
    with pytest.raises(ValueError, match="feature_feedback.lambda_spec"):
        _feature_feedback_transforms(str(tmp_path))


def test_rejects_bool_disguised_as_number(tmp_path: Path) -> None:
    _write_feedback(tmp_path, {"threshold": True})
    with pytest.raises(ValueError, match="feature_feedback.threshold"):
        _feature_feedback_transforms(str(tmp_path))


def test_finite_values_round_trip_through_json(tmp_path: Path) -> None:
    """Defensive check that ``json.dumps`` -> :func:`_feature_feedback_transforms`
    preserves finiteness — guards against accidental ``NaN`` slipping in
    via writer-side bugs.
    """
    _write_feedback(tmp_path, {"reward": 0.0})
    transforms = _feature_feedback_transforms(str(tmp_path))
    spec = transforms[1]
    assert spec is not None
    for key in ("scale", "offset", "gamma", "threshold", "below_scale", "above_scale"):
        assert math.isfinite(spec[key]), f"non-finite {key}: {spec[key]!r}"
