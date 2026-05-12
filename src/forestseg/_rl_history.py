"""RL-history persistence helpers used by :mod:`forestseg.cli`.

This module owns:

- the legacy fusion-parameter defaults used to fill in missing fields
  on entries written by older `dp` versions,
- the canonical ``selection_metric`` whitelist,
- normalization / validation / load / append / rewrite helpers for the
  on-disk ``rl_history.json`` payload that the closed-loop driver
  produces.

The helpers were extracted from :mod:`forestseg.cli` to keep the CLI
module focused on argparse plumbing and command handlers. The names
remain re-exported from :mod:`forestseg.cli` for backward compatibility
with the upstream test suite.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

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
    "_history_entry",
    "_load_rl_history",
    "_normalize_legacy_rl_history_entry",
    "_rewrite_latest_rl_history_entry",
    "_validate_rl_history_entry",
]

VALID_TRAIN_SELECTION_METRICS = {"accuracy", "precision", "recall", "f1", "iou"}
VALID_RL_SELECTION_METRICS = {"reward", *VALID_TRAIN_SELECTION_METRICS}

LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS = {
    "lambda_spec": 0.2,
    "lambda_tex": 0.2,
    "threshold": 0.5,
    "min_area_m2": 200.0,
    "morph_kernel": 3,
    "shadow_penalty": 0.5,
}


def _normalize_legacy_rl_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
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

    reward_source = normalized_validation_metrics.get(
        "reward", normalized_entry.get("reward", normalized_entry.get("score", 0.0))
    )
    if isinstance(reward_source, bool):
        reward_source = 0.0
    try:
        reward = float(reward_source)
    except (TypeError, ValueError):
        reward = 0.0
    if not np.isfinite(reward):
        reward = 0.0
    reward = max(reward, 0.0)
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


def _validate_rl_history_entry(entry: Any, history_path: str, index: int | None = None) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid rl_history entries: {history_path}")
    if not entry:
        raise ValueError(f"Invalid rl_history entries: {history_path}")
    normalized_entry = _normalize_legacy_rl_history_entry(entry)
    normalized_entry["selection_metric"] = _validate_choice(
        normalized_entry.get("selection_metric"), "rl_history.selection_metric", VALID_RL_SELECTION_METRICS
    )
    params = normalized_entry.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Invalid rl_history entries: {history_path}")
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
        raise ValueError(f"Invalid rl_history entries: {history_path}")
    normalized_validation_metrics = dict(validation_metrics)
    reward_value = _validate_non_negative_float(
        normalized_validation_metrics.get("reward"), "rl_history.validation_metrics.reward"
    )
    normalized_validation_metrics["reward"] = reward_value
    top_level_reward = normalized_entry.get("reward")
    if top_level_reward is not None:
        normalized_top_level_reward = _validate_non_negative_float(top_level_reward, "rl_history.reward")
        if not np.isclose(normalized_top_level_reward, reward_value):
            raise ValueError(f"Invalid rl_history entries: {history_path}")
        normalized_entry["reward"] = normalized_top_level_reward
    selected_metric = normalized_entry["selection_metric"]
    if selected_metric not in normalized_validation_metrics:
        raise ValueError(f"Invalid rl_history entries: {history_path}")
    selected_metric_value = normalized_validation_metrics.get(selected_metric)
    if isinstance(selected_metric_value, bool):
        raise ValueError(f"Invalid rl_history entries: {history_path}")
    try:
        parsed_selected_metric = float(selected_metric_value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid rl_history entries: {history_path}") from None
    if not np.isfinite(parsed_selected_metric):
        raise ValueError(f"Invalid rl_history entries: {history_path}")
    normalized_validation_metrics[selected_metric] = parsed_selected_metric
    normalized_entry["validation_metrics"] = normalized_validation_metrics
    round_value = normalized_entry.get("round")
    if round_value is not None:
        normalized_entry["round"] = _validate_positive_int(round_value, "rl_history.round")
    score_value = normalized_entry.get("score")
    if score_value is not None:
        if isinstance(score_value, bool):
            raise ValueError(f"Invalid rl_history entries: {history_path}")
        try:
            parsed_score = float(score_value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid rl_history entries: {history_path}") from None
        if not np.isfinite(parsed_score):
            raise ValueError(f"Invalid rl_history entries: {history_path}")
        if not np.isclose(parsed_score, parsed_selected_metric):
            raise ValueError(f"Invalid rl_history entries: {history_path}")
        normalized_entry["score"] = parsed_score
    if selected_metric == "reward":
        if (
            score_value is not None
            and top_level_reward is not None
            and not np.isclose(normalized_entry["score"], normalized_entry["reward"])
        ):
            raise ValueError(f"Invalid rl_history entries: {history_path}")
        if score_value is not None and not np.isclose(normalized_entry["score"], reward_value):
            raise ValueError(f"Invalid rl_history entries: {history_path}")
    train_metrics = normalized_entry.get("train_metrics")
    if train_metrics is not None and not isinstance(train_metrics, dict):
        raise ValueError(f"Invalid rl_history entries: {history_path}")
    return normalized_entry


def _load_rl_history(history_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(history_path):
        return []
    try:
        with open(history_path, encoding="utf-8") as f:
            loaded = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid rl_history file: {history_path}") from exc
    if not isinstance(loaded, list):
        raise ValueError(f"Invalid rl_history payload: {history_path}")
    return [_validate_rl_history_entry(entry, history_path, index=idx) for idx, entry in enumerate(loaded, start=1)]


def _history_entry(
    round_no: int,
    selection_metric: str,
    score: float,
    reward: float,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    artifacts: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "schema_version": 2,
        "round": round_no,
        "selection_metric": selection_metric,
        "score": score,
        "reward": reward,
        "stage_used": None,
        "params": None,
        "checkpoint_path": None,
        "trained": None,
        "feedback_path": None,
        "feature_meta": None,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "artifacts": artifacts,
    }
    if extra:
        entry.update(extra)
    return entry


def _rewrite_latest_rl_history_entry(history_path: str, round_no: int, selection_metric: str, score: float) -> None:
    history = _load_rl_history(history_path)
    if not history:
        raise ValueError(f"Invalid rl_history payload: {history_path}")
    latest_entry = dict(history[-1])
    latest_entry["round"] = round_no
    latest_entry["selection_metric"] = selection_metric
    latest_entry["score"] = score
    history[-1] = _validate_rl_history_entry(latest_entry, history_path, index=len(history))
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
