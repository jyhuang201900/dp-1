from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from itertools import product
from typing import Any

import numpy as np

from ..core.constants import (
    FUSION_POSTPROCESS_PARAM_KEYS,
    REQUIRED_FUSION_FIXED_PARAM_KEYS,
    REQUIRED_FUSION_GRID_KEYS,
)
from ..export_pkg.metrics import binary_metrics
from ..labels.points import sample_raster_at_points
from .core import FusionParams, fuse_probabilities, run_stage0, run_stage1_grid

_REQUIRED_GRID_KEYS = REQUIRED_FUSION_FIXED_PARAM_KEYS

_POSTPROCESS_GRID_KEYS = FUSION_POSTPROCESS_PARAM_KEYS

_FUSION_GRID_KEYS = REQUIRED_FUSION_GRID_KEYS


def _validate_stage(stage: int) -> int:
    if stage not in {0, 1}:
        raise ValueError(f"stage must be 0 or 1, got {stage}")
    return stage


def _validate_grid_param_keys(grid_params: dict[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        values = grid_params.get(key)
        if values is None:
            raise ValueError(f"grid_params.{key} is required")
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"grid_params.{key} must be a non-empty sequence")
        if len(values) == 0:
            raise ValueError(f"grid_params.{key} cannot be empty")


def _validate_present_grid_param_keys(grid_params: dict[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        if key not in grid_params:
            continue
        values = grid_params[key]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"grid_params.{key} must be a non-empty sequence")
        if len(values) == 0:
            raise ValueError(f"grid_params.{key} cannot be empty")


def _validate_grid_params(grid_params: dict[str, Any]) -> None:
    _validate_grid_param_keys(grid_params, _REQUIRED_GRID_KEYS)


def _resolve_grid_or_fixed_param(fixed_params: dict[str, Any], grid_params: dict[str, Any], key: str) -> Any:
    if key in fixed_params:
        return fixed_params[key]
    grid_values = grid_params.get(key)
    if grid_values is None:
        raise ValueError(f"grid_params.{key} is required")
    if isinstance(grid_values, (str, bytes)) or not isinstance(grid_values, Sequence):
        raise ValueError(f"grid_params.{key} must be a non-empty sequence")
    if len(grid_values) == 0:
        raise ValueError(f"grid_params.{key} cannot be empty")
    return grid_values[0]


def choose_fusion(
    stage: int,
    prob_dl: np.ndarray,
    prob_spec: np.ndarray,
    prob_tex: np.ndarray,
    shadow_score: np.ndarray,
    fixed_params: dict,
    grid_params: dict,
) -> tuple[np.ndarray, dict, float | None, str]:
    """
    Returns: fused_prob, params_dict, score, stage_used
    """
    resolved_stage = _validate_stage(stage)
    if resolved_stage == 0:
        fused, p = run_stage0(prob_dl, prob_spec, prob_tex, shadow_score, fixed_params)
        return fused, asdict(p), None, "stage0"

    fused, p, score = run_stage1_grid(prob_dl, prob_spec, prob_tex, shadow_score, grid_params)
    return fused, asdict(p), float(score), "stage1"


def choose_fusion_by_validation(
    stage: int,
    prob_dl_path: str,
    prob_spec_path: str,
    prob_tex_path: str,
    val_points: list[dict[str, Any]],
    fixed_params: dict,
    grid_params: dict,
) -> tuple[FusionParams, dict[str, Any], str]:
    resolved_stage = _validate_stage(stage)
    y_true = np.asarray([int(p["label"]) for p in val_points], dtype=np.uint8)
    dl_vals = np.asarray(sample_raster_at_points(prob_dl_path, val_points), dtype=np.float32)
    spec_vals = np.asarray(sample_raster_at_points(prob_spec_path, val_points), dtype=np.float32)
    tex_vals = np.asarray(sample_raster_at_points(prob_tex_path, val_points), dtype=np.float32)

    valid = np.isfinite(dl_vals) & np.isfinite(spec_vals) & np.isfinite(tex_vals)
    valid &= (dl_vals >= 0.0) & (dl_vals <= 1.0)
    valid &= (spec_vals >= 0.0) & (spec_vals <= 1.0)
    valid &= (tex_vals >= 0.0) & (tex_vals <= 1.0)
    if resolved_stage > 0:
        _validate_present_grid_param_keys(grid_params, _FUSION_GRID_KEYS)
        if np.any(valid):
            _validate_grid_param_keys(grid_params, _FUSION_GRID_KEYS)
    resolved_postprocess = {
        key: _resolve_grid_or_fixed_param(fixed_params, grid_params, key) for key in _POSTPROCESS_GRID_KEYS
    }
    resolved_fallback_fusion = {
        key: _resolve_grid_or_fixed_param(fixed_params, grid_params, key) for key in _FUSION_GRID_KEYS
    }
    if not np.any(valid):
        p = FusionParams.from_mapping(
            {**resolved_fallback_fusion, **resolved_postprocess},
        )
        fallback_metrics: dict[str, Any] = {
            "threshold": float(p.threshold),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0,
            "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
            "count": 0,
            "reward": 0.0,
            "valid_points_used": 0,
        }
        return p, fallback_metrics, "validation_fallback_fixed"

    y_true = y_true[valid]
    dl_vals = dl_vals[valid]
    spec_vals = spec_vals[valid]
    tex_vals = tex_vals[valid]

    if resolved_stage == 0:
        candidate = FusionParams.from_mapping(fixed_params)
        fused_prob = fuse_probabilities(dl_vals, spec_vals, tex_vals, candidate)
        metrics = binary_metrics(y_true, fused_prob, threshold=float(candidate.threshold))
        metrics["reward"] = float(metrics["accuracy"]) + float(metrics["f1"]) + float(metrics["iou"])
        metrics["valid_points_used"] = len(y_true)
        return candidate, metrics, "validation_stage0"

    best_reward = -1e18
    best_candidate: FusionParams | None = None
    best_metrics: dict[str, Any] | None = None
    precomputed_valid: np.ndarray = np.ones(len(dl_vals), dtype=bool)
    for ls, lt, th in product(
        grid_params["lambda_spec"],
        grid_params["lambda_tex"],
        grid_params["threshold"],
    ):
        candidate = FusionParams.from_mapping(
            {"lambda_spec": ls, "lambda_tex": lt, "threshold": th, **resolved_postprocess},
        )
        fused_prob = fuse_probabilities(dl_vals, spec_vals, tex_vals, candidate, precomputed_valid=precomputed_valid)
        metrics = binary_metrics(y_true, fused_prob, threshold=float(candidate.threshold))
        reward = float(metrics["accuracy"]) + float(metrics["f1"]) + float(metrics["iou"])
        if reward > best_reward:
            best_reward = reward
            best_candidate = candidate
            best_metrics = metrics

    assert best_candidate is not None and best_metrics is not None
    best_metrics["reward"] = float(best_reward)
    best_metrics["valid_points_used"] = len(y_true)
    return best_candidate, best_metrics, f"validation_stage{stage}"
