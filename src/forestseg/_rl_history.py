"""Strict validation + IO for the on-disk ``rl_history.json`` payload.

This module owns the canonical ``selection_metric`` whitelists and
the load / validate / append / rewrite helpers consumed by
:mod:`forestseg.cli`. Lenient pre-processing of older entry shapes
lives in :mod:`forestseg._rl_history_legacy` and is applied as the
first step of validation; the two modules together implement the
"normalize-then-validate" pipeline.

Names remain re-exported from :mod:`forestseg.cli` for backward
compatibility with the upstream test suite.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from ._io import atomic_write_json, read_json
from ._rl_history_legacy import (
    LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS as LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS,
    _normalize_legacy_rl_history_entry as _normalize_legacy_rl_history_entry,
)
from ._validators import (
    _validate_choice,
    _validate_non_negative_float,
    _validate_positive_float,
    _validate_positive_int,
    _validate_unit_interval,
)

__all__ = [
    "LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS",
    "VALID_RL_SELECTION_METRICS",
    "VALID_TRAIN_SELECTION_METRICS",
    "_load_rl_history",
    "_normalize_legacy_rl_history_entry",
    "_rewrite_latest_rl_history_entry",
    "_validate_rl_history_entry",
]

VALID_TRAIN_SELECTION_METRICS = {"accuracy", "precision", "recall", "f1", "iou"}
VALID_RL_SELECTION_METRICS = {"reward", *VALID_TRAIN_SELECTION_METRICS}


def _entry_invalid(history_path: str, reason: str, index: int | None = None) -> ValueError:
    """Build the canonical ``Invalid rl_history entries`` error.

    The ``Invalid rl_history entries: <history_path>`` prefix is
    preserved verbatim so callers and tests can keep matching it;
    ``reason`` is appended after a colon for human debuggability,
    and ``index`` (1-based) is included when the offending entry's
    position is known.
    """
    suffix_parts: list[str] = []
    if index is not None:
        suffix_parts.append(f"entry #{index}")
    suffix_parts.append(reason)
    return ValueError(f"Invalid rl_history entries: {history_path}: {' — '.join(suffix_parts)}")


def _validate_rl_history_entry(entry: Any, history_path: str, index: int | None = None) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise _entry_invalid(history_path, "entry must be an object", index=index)
    if not entry:
        raise _entry_invalid(history_path, "entry must not be empty", index=index)
    normalized_entry = _normalize_legacy_rl_history_entry(entry)
    normalized_entry["selection_metric"] = _validate_choice(
        normalized_entry.get("selection_metric"), "rl_history.selection_metric", VALID_RL_SELECTION_METRICS
    )
    params = normalized_entry.get("params")
    if not isinstance(params, dict):
        raise _entry_invalid(history_path, "params must be an object", index=index)
    normalized_entry["params"] = {
        "lambda_spec": _validate_unit_interval(params.get("lambda_spec"), "rl_history.params.lambda_spec"),
        "lambda_tex": _validate_unit_interval(params.get("lambda_tex"), "rl_history.params.lambda_tex"),
        "threshold": _validate_unit_interval(params.get("threshold"), "rl_history.params.threshold"),
        "min_area_m2": _validate_positive_float(params.get("min_area_m2"), "rl_history.params.min_area_m2"),
        "morph_kernel": _validate_positive_int(params.get("morph_kernel"), "rl_history.params.morph_kernel"),
        "shadow_penalty": _validate_unit_interval(params.get("shadow_penalty"), "rl_history.params.shadow_penalty"),
    }
    validation_metrics = normalized_entry.get("validation_metrics")
    if not isinstance(validation_metrics, dict):
        raise _entry_invalid(history_path, "validation_metrics must be an object", index=index)
    normalized_validation_metrics = dict(validation_metrics)
    reward_value = _validate_non_negative_float(
        normalized_validation_metrics.get("reward"), "rl_history.validation_metrics.reward"
    )
    normalized_validation_metrics["reward"] = reward_value
    top_level_reward = normalized_entry.get("reward")
    if top_level_reward is not None:
        normalized_top_level_reward = _validate_non_negative_float(top_level_reward, "rl_history.reward")
        if not np.isclose(normalized_top_level_reward, reward_value):
            raise _entry_invalid(
                history_path,
                "top-level reward does not match validation_metrics.reward",
                index=index,
            )
        normalized_entry["reward"] = normalized_top_level_reward
    selected_metric = normalized_entry["selection_metric"]
    if selected_metric not in normalized_validation_metrics:
        raise _entry_invalid(
            history_path,
            f"selection_metric={selected_metric!r} not present in validation_metrics",
            index=index,
        )
    selected_metric_value = normalized_validation_metrics.get(selected_metric)
    if isinstance(selected_metric_value, bool) or not isinstance(selected_metric_value, int | float | str):
        raise _entry_invalid(
            history_path,
            f"validation_metrics[{selected_metric!r}] must be numeric",
            index=index,
        )
    try:
        parsed_selected_metric = float(selected_metric_value)
    except (TypeError, ValueError):
        raise _entry_invalid(
            history_path,
            f"validation_metrics[{selected_metric!r}] is not coercible to float",
            index=index,
        ) from None
    if not np.isfinite(parsed_selected_metric):
        raise _entry_invalid(
            history_path,
            f"validation_metrics[{selected_metric!r}] is not finite",
            index=index,
        )
    normalized_validation_metrics[selected_metric] = parsed_selected_metric
    normalized_entry["validation_metrics"] = normalized_validation_metrics
    round_value = normalized_entry.get("round")
    if round_value is not None:
        normalized_entry["round"] = _validate_positive_int(round_value, "rl_history.round")
    score_value = normalized_entry.get("score")
    if score_value is not None:
        if isinstance(score_value, bool):
            raise _entry_invalid(history_path, "score must be numeric, not bool", index=index)
        try:
            parsed_score = float(score_value)
        except (TypeError, ValueError):
            raise _entry_invalid(history_path, "score is not coercible to float", index=index) from None
        if not np.isfinite(parsed_score):
            raise _entry_invalid(history_path, "score is not finite", index=index)
        if not np.isclose(parsed_score, parsed_selected_metric):
            raise _entry_invalid(
                history_path,
                f"score does not match validation_metrics[{selected_metric!r}]",
                index=index,
            )
        normalized_entry["score"] = parsed_score
    if selected_metric == "reward":
        if (
            score_value is not None
            and top_level_reward is not None
            and not np.isclose(normalized_entry["score"], normalized_entry["reward"])
        ):
            raise _entry_invalid(
                history_path,
                "score and reward disagree for selection_metric='reward'",
                index=index,
            )
        if score_value is not None and not np.isclose(normalized_entry["score"], reward_value):
            raise _entry_invalid(
                history_path,
                "score does not match validation_metrics.reward for selection_metric='reward'",
                index=index,
            )
    train_metrics = normalized_entry.get("train_metrics")
    if train_metrics is not None and not isinstance(train_metrics, dict):
        raise _entry_invalid(history_path, "train_metrics must be an object when present", index=index)
    return normalized_entry


def _load_rl_history(history_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(history_path):
        return []
    try:
        loaded = read_json(history_path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid rl_history file: {history_path}") from exc
    if not isinstance(loaded, list):
        raise ValueError(f"Invalid rl_history payload: {history_path}")
    return [_validate_rl_history_entry(entry, history_path, index=idx) for idx, entry in enumerate(loaded, start=1)]


def _rewrite_latest_rl_history_entry(history_path: str, round_no: int, selection_metric: str, score: float) -> None:
    history = _load_rl_history(history_path)
    if not history:
        raise ValueError(f"Invalid rl_history payload: {history_path}")
    latest_entry = dict(history[-1])
    latest_entry["round"] = round_no
    latest_entry["selection_metric"] = selection_metric
    latest_entry["score"] = score
    history[-1] = _validate_rl_history_entry(latest_entry, history_path, index=len(history))
    atomic_write_json(history_path, history)
