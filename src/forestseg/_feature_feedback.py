"""Translate the per-round ``feature_feedback.json`` into spec/tex transforms.

The feedback file is written at the end of each closed-loop round and
captures the chosen fusion parameters plus the realized reward. This
module turns those numbers into the spec/tex feature-stack transform
descriptors that the next round's
:func:`forestseg.features.build_feature_stack` consumes.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ._constants import wf

__all__ = ["_feature_feedback_transforms"]


def _feature_feedback_transforms(work: str) -> list[dict[str, Any] | None]:
    """Return the four per-band transform descriptors for the next round.

    The returned list is ordered ``[input, spec, tex, dl]`` (matching
    :func:`forestseg.features.build_feature_stack`). When no feedback
    file exists yet, every entry is ``None`` (i.e. identity).
    """
    feedback_path = wf(work, "feature_feedback")
    if not os.path.exists(feedback_path):
        return [None, None, None, None]
    with open(feedback_path, encoding="utf-8") as f:
        feedback = json.load(f)
    spec_threshold = float(feedback.get("threshold", 0.5))
    lambda_spec = float(feedback.get("lambda_spec", 0.2))
    lambda_tex = float(feedback.get("lambda_tex", 0.2))
    reward = float(feedback.get("reward", 0.0))
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
