"""Translate the per-round ``feature_feedback.json`` into spec/tex transforms.

The feedback file is written at the end of each closed-loop round and
captures the chosen fusion parameters plus the realized reward. This
module turns those numbers into the spec/tex feature-stack transform
descriptors that the next round's
:func:`forestseg.features.build_feature_stack` consumes.

The loader is defensive: if the on-disk JSON exists but is corrupt
(invalid JSON, wrong top-level type, missing key, non-finite value)
we raise a :class:`ValueError` rather than silently substituting
defaults — silent fallback would mask a corrupted artefact and lead
to a confusing "training looks fine but feedback never kicks in"
failure mode.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from ._constants import wf

__all__ = ["_feature_feedback_transforms"]


def _coerce_feedback_float(value: Any, key: str, *, allow_negative: bool = False) -> float:
    """Return ``value`` coerced to a finite float with the expected sign.

    Raises :class:`ValueError` for ``bool`` (which would silently coerce
    to 0/1 under ``float``), unparseable types, and ``NaN`` / ``inf``.
    When ``allow_negative`` is false, negative values are also rejected
    — every consumer of this module currently expects non-negative
    scaling factors.
    """
    if isinstance(value, bool):
        raise ValueError(f"feature_feedback.{key} must be numeric, not bool: {value!r}")
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature_feedback.{key} is not coercible to float: {value!r}") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"feature_feedback.{key} is not finite: {value!r}")
    if not allow_negative and coerced < 0.0:
        raise ValueError(f"feature_feedback.{key} must be non-negative: {coerced}")
    return coerced


def _feature_feedback_transforms(work: str) -> list[dict[str, Any] | None]:
    """Return the four per-band transform descriptors for the next round.

    The returned list is ordered ``[input, spec, tex, dl]`` (matching
    :func:`forestseg.features.build_feature_stack`). When no feedback
    file exists yet, every entry is ``None`` (i.e. identity). A
    syntactically broken or semantically inconsistent feedback file
    raises :class:`ValueError`.
    """
    feedback_path = wf(work, "feature_feedback")
    if not os.path.exists(feedback_path):
        return [None, None, None, None]
    try:
        with open(feedback_path, encoding="utf-8") as f:
            feedback = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"feature_feedback is not valid JSON: {feedback_path}") from exc
    if not isinstance(feedback, dict):
        raise ValueError(f"feature_feedback payload must be an object: {feedback_path}")

    spec_threshold = _coerce_feedback_float(feedback.get("threshold", 0.5), "threshold")
    lambda_spec = _coerce_feedback_float(feedback.get("lambda_spec", 0.2), "lambda_spec")
    lambda_tex = _coerce_feedback_float(feedback.get("lambda_tex", 0.2), "lambda_tex")
    reward = _coerce_feedback_float(feedback.get("reward", 0.0), "reward")
    reward_gain = min(max(reward / 3.0, 0.0), 1.0)
    return [
        None,
        {
            "scale": max(lambda_spec, 1e-3) / 0.2,
            "offset": 0.05 * reward_gain,
            "gamma": max(0.7, 1.1 - 0.2 * reward_gain),
            "threshold": spec_threshold,
            "below_scale": 0.5,
            "above_scale": 1.0 + 0.15 * reward_gain,
        },
        {
            "scale": max(lambda_tex, 1e-3) / 0.2,
            "offset": 0.05 * reward_gain,
            "gamma": max(0.7, 1.1 - 0.2 * reward_gain),
            "threshold": spec_threshold,
            "below_scale": 0.5,
            "above_scale": 1.0 + 0.15 * reward_gain,
        },
        None,
    ]
