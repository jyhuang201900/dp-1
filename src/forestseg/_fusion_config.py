"""Validation of the ``fusion`` section of the pipeline config.

This module covers:

- :func:`_resolve_fusion_stage`: pull the active fusion stage (``0`` or
  ``1``) from CLI override or YAML config.
- :func:`_validate_fusion_config`: top-level helper that returns the
  ``(stage, fixed_params, grid_params, grid_keys)`` tuple consumed by
  the RL-fusion and closed-loop drivers.
- :func:`_validate_fusion_param_value`: per-key value coercion that
  knows the allowed numeric range of each fusion parameter.
- :func:`_validate_fusion_params`: validate a fully-fixed parameter
  dict.
- :func:`_validate_fusion_param_candidates`: validate a grid-search
  parameter dict whose values are candidate lists.
"""

from __future__ import annotations

from typing import Any

from ._constants import REQUIRED_FUSION_FIXED_PARAM_KEYS, REQUIRED_FUSION_GRID_KEYS
from ._validators import (
    _validate_positive_float,
    _validate_positive_int,
    _validate_stage_int,
    _validate_unit_interval,
)

__all__ = [
    "_resolve_fusion_stage",
    "_validate_fusion_config",
    "_validate_fusion_param_candidates",
    "_validate_fusion_param_value",
    "_validate_fusion_params",
]


def _resolve_fusion_stage(cfg: dict[str, Any], stage_override: int | None = None) -> int:
    raw_value = stage_override if stage_override is not None else cfg.get("fusion", {}).get("stage", 1)
    return _validate_stage_int(raw_value, "fusion.stage")


def _validate_fusion_param_value(value: Any, label: str, key: str) -> Any:
    if key in {"lambda_spec", "lambda_tex", "threshold", "shadow_penalty"}:
        return _validate_unit_interval(value, label)
    if key == "min_area_m2":
        return _validate_positive_float(value, label)
    if key == "morph_kernel":
        return _validate_positive_int(value, label)
    raise ValueError(f"未知 fusion 参数：{label}。")


def _validate_fusion_params(params: dict[str, Any], *, label_prefix: str) -> dict[str, Any]:
    if not isinstance(params, dict) or not params:
        raise ValueError(f"{label_prefix} 不能为空，且必须为参数字典。")
    missing_keys = [key for key in REQUIRED_FUSION_FIXED_PARAM_KEYS if key not in params]
    if missing_keys:
        raise ValueError(f"{label_prefix} 缺少必需参数：{', '.join(missing_keys)}。")
    return {
        key: _validate_fusion_param_value(params.get(key), f"{label_prefix}.{key}", key)
        for key in REQUIRED_FUSION_FIXED_PARAM_KEYS
    }


def _validate_fusion_param_candidates(grid_params: dict[str, Any], *, label_prefix: str) -> dict[str, list[Any]]:
    normalized: dict[str, list[Any]] = {}
    for key, values in grid_params.items():
        if key not in REQUIRED_FUSION_FIXED_PARAM_KEYS:
            raise ValueError(f"未知 fusion 参数：{label_prefix}.{key}。")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{label_prefix}.{key} 参数候选不能为空列表。")
        normalized[key] = [
            _validate_fusion_param_value(candidate, f"{label_prefix}.{key}[{idx}]", key)
            for idx, candidate in enumerate(values)
        ]
    return normalized


def _validate_fusion_config(
    cfg: dict[str, Any], *, stage_override: int | None = None
) -> tuple[int, dict[str, Any], dict[str, list[Any]], list[str]]:
    fcfg = cfg.get("fusion", {})
    fusion_stage = _resolve_fusion_stage(cfg, stage_override=stage_override)
    fixed_params = fcfg.get("fixed_params", {})
    grid_params = fcfg.get("grid", {})
    if fusion_stage == 0:
        fixed_params = _validate_fusion_params(fixed_params, label_prefix="fusion.fixed_params")
        grid_keys = sorted(grid_params.keys()) if isinstance(grid_params, dict) else []
        return fusion_stage, fixed_params, {}, grid_keys
    if not isinstance(grid_params, dict) or not grid_params:
        raise ValueError("fusion.grid 不能为空，且必须为参数网格字典。")
    missing_grid_keys = [key for key in REQUIRED_FUSION_GRID_KEYS if key not in grid_params]
    if missing_grid_keys:
        raise ValueError(f"fusion.grid 缺少必需参数：{', '.join(missing_grid_keys)}。")
    grid_params = _validate_fusion_param_candidates(grid_params, label_prefix="fusion.grid")
    fixed_params = _validate_fusion_params(fixed_params, label_prefix="fusion.fixed_params")
    return fusion_stage, fixed_params, grid_params, sorted(grid_params.keys())
