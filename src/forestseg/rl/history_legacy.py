"""Best-effort normalization of legacy ``rl_history.json`` entries.

Older versions of the closed-loop driver wrote entries with looser
schemas — missing ``params`` keys, mis-typed ``selection_metric``,
absent ``reward`` mirrors, etc. :func:`_normalize_legacy_rl_history_entry`
is a *lenient* pre-processor that fills in safe defaults so the downstream
strict validator in :mod:`forestseg._rl_history` has a uniform shape to
work with.

This module is intentionally separate from the strict-validation
module so the two responsibilities can evolve independently: legacy
shapes are extended here when older payloads need new fallbacks,
while the strict validator only ever gets stricter. Both modules
share the same selection-metric whitelist via re-import.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS",
    "_normalize_legacy_rl_history_entry",
]

LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS = {
    "lambda_spec": 0.2,
    "lambda_tex": 0.2,
    "threshold": 0.5,
    "min_area_m2": 200.0,
    "morph_kernel": 3,
    "shadow_penalty": 0.5,
}


def _coerce_finite_non_negative(value: Any, fallback: float = 0.0) -> float:
    """Coerce ``value`` to a finite non-negative float, or fall back."""
    if isinstance(value, bool):
        return fallback
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(coerced):
        return fallback
    return max(coerced, 0.0)


def _normalize_legacy_rl_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Fill in safe defaults for older rl_history entry shapes.

    Normalizes (in order): ``selection_metric`` (lowercased, downgraded
    to ``"reward"`` if unknown or if the named metric is absent),
    ``validation_metrics`` (ensures dict + a finite non-negative
    ``reward`` mirror), ``params`` (merged onto
    :data:`LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS`), and the top-level
    ``score`` / ``reward`` mirrors.

    Does *not* enforce strict typing or schema constraints — that is
    the job of :func:`forestseg._rl_history._validate_rl_history_entry`,
    which calls this function first.
    """
    # Import lazily to avoid a circular dependency at module load time.
    from .history import VALID_RL_SELECTION_METRICS

    normalized_entry = dict(entry)

    raw_selection_metric = normalized_entry.get("selection_metric", "reward")
    if isinstance(raw_selection_metric, str):
        raw_selection_metric = raw_selection_metric.strip().lower()
    if raw_selection_metric not in VALID_RL_SELECTION_METRICS:
        raw_selection_metric = "reward"
    selection_metric = str(raw_selection_metric)

    validation_metrics = normalized_entry.get("validation_metrics")
    if isinstance(validation_metrics, dict):
        normalized_validation_metrics = dict(validation_metrics)
    else:
        normalized_validation_metrics = {}

    reward = _coerce_finite_non_negative(
        normalized_validation_metrics.get("reward", normalized_entry.get("reward", normalized_entry.get("score", 0.0)))
    )
    normalized_validation_metrics["reward"] = reward

    selected_metric_source = normalized_validation_metrics.get(selection_metric)
    downgraded_to_reward = False
    if selected_metric_source is None and selection_metric != "reward":
        selection_metric = "reward"
        downgraded_to_reward = True
        selected_metric_source = normalized_validation_metrics.get("reward")
    if selected_metric_source is None:
        selected_metric_source = reward
    if isinstance(selected_metric_source, bool):
        selected_metric_source = reward
    try:
        selected_metric_value = float(selected_metric_source)
    except (TypeError, ValueError):
        selected_metric_value = reward
    if not np.isfinite(selected_metric_value):
        selected_metric_value = reward
    normalized_validation_metrics[selection_metric] = selected_metric_value

    params = normalized_entry.get("params")
    if isinstance(params, dict):
        merged_params = {**LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS, **params}
    else:
        merged_params = dict(LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS)

    normalized_entry["selection_metric"] = selection_metric
    normalized_entry["validation_metrics"] = normalized_validation_metrics
    normalized_entry["params"] = merged_params
    if downgraded_to_reward or "score" not in normalized_entry:
        normalized_entry["score"] = selected_metric_value
    if "reward" not in normalized_entry:
        normalized_entry["reward"] = reward
    return normalized_entry
