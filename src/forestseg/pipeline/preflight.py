"""Section-by-section validators backing ``cmd_preflight_check``.

The closed-loop summary embeds a ``preflight`` block with three
top-level keys: ``checks`` (resolved paths), ``warnings`` (non-fatal
notes), and ``config`` (echo of the validated, fully-typed sub-config
that downstream stages will use). To keep the entrypoint focused on
that report shape, this module owns the per-section validation:
``LabelsPreflight`` / ``DLPreflight`` / ``RLLoopPreflight`` /
``InputPreflight``.

Each section is a frozen dataclass that captures the already-coerced
values; the entrypoint composes them into the final ``config`` block.
This module performs *no* IO — path existence is the entrypoint's
job, since the failing-path strings are part of the public error
contract that tests assert on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.validators import (
    _validate_bool,
    _validate_choice,
    _validate_non_negative_float,
    _validate_non_negative_int,
    _validate_positive_float,
    _validate_positive_int,
    _validate_ratio,
    _validate_unit_interval,
)
from ..rl.bandit_config import _resolve_bandit_config
from ..rl.history import VALID_RL_SELECTION_METRICS, VALID_TRAIN_SELECTION_METRICS

__all__ = [
    "DLPreflight",
    "InputPreflight",
    "LabelsPreflight",
    "RLLoopPreflight",
    "resolve_dl_preflight",
    "resolve_input_preflight",
    "resolve_labels_preflight",
    "resolve_rl_loop_preflight",
]


@dataclass(frozen=True)
class LabelsPreflight:
    """Validated ``labels`` sub-config + report-ready ``config`` view."""

    layer: Any
    class_field: str
    positive_value: str
    negative_value: str
    split_ratio: float
    grid_size: float
    split_seed: int

    def as_report(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "class_field": self.class_field,
            "positive_value": self.positive_value,
            "negative_value": self.negative_value,
            "split_ratio": self.split_ratio,
            "grid_size": self.grid_size,
            "split_seed": self.split_seed,
        }


@dataclass(frozen=True)
class DLPreflight:
    """Validated ``dl`` sub-config."""

    tile_size: int
    stride: int
    batch_size: int
    epochs: int
    mc_dropout_passes: int
    sample_tile_size: int
    max_patches: int
    aux_warmup_epochs: int
    val_threshold: float
    selection_metric: str

    def as_report(self) -> dict[str, Any]:
        return {
            "tile_size": self.tile_size,
            "stride": self.stride,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "mc_dropout_passes": self.mc_dropout_passes,
            "sample_tile_size": self.sample_tile_size,
            "max_patches": self.max_patches,
            "aux_warmup_epochs": self.aux_warmup_epochs,
            "val_threshold": self.val_threshold,
            "selection_metric": self.selection_metric,
        }


@dataclass(frozen=True)
class RLLoopPreflight:
    """Validated ``rl_loop`` sub-config (incl. bandit)."""

    rounds: int
    patience: int
    min_delta: float
    selection_metric: str
    bandit: dict[str, Any]

    def as_report(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "selection_metric": self.selection_metric,
            "bandit": self.bandit,
        }


@dataclass(frozen=True)
class InputPreflight:
    """Validated ``input`` sub-config (scene resolution flags)."""

    prefer_v1: bool
    fallback_quick: bool


def resolve_labels_preflight(lcfg: dict[str, Any]) -> LabelsPreflight:
    """Validate scalar fields under ``labels`` (paths are checked separately)."""
    class_field = str(lcfg.get("class_field", "class")).strip()
    positive_value = str(lcfg.get("positive_value", "1")).strip()
    negative_value = str(lcfg.get("negative_value", "0")).strip()
    if not class_field:
        raise ValueError("labels.class_field 不能为空。")
    if positive_value.lower() == negative_value.lower():
        raise ValueError("labels.positive_value 和 labels.negative_value 不能相同。")
    return LabelsPreflight(
        layer=lcfg.get("layer"),
        class_field=class_field,
        positive_value=positive_value,
        negative_value=negative_value,
        split_ratio=_validate_ratio(lcfg.get("split_ratio", 0.7), "labels.split_ratio"),
        grid_size=_validate_positive_float(lcfg.get("grid_size", 1000.0), "labels.grid_size"),
        split_seed=_validate_non_negative_int(lcfg.get("split_seed", 42), "labels.split_seed"),
    )


def resolve_dl_preflight(dcfg: dict[str, Any]) -> DLPreflight:
    """Validate ``dl`` scalars and enforce ``stride <= tile_size``."""
    tile_size = _validate_positive_int(dcfg.get("tile_size", 512), "dl.tile_size")
    stride = _validate_positive_int(dcfg.get("stride", 384), "dl.stride")
    if stride > tile_size:
        raise ValueError(f"dl.stride 不能大于 dl.tile_size，当前为 {stride} > {tile_size}。")
    return DLPreflight(
        tile_size=tile_size,
        stride=stride,
        batch_size=_validate_positive_int(dcfg.get("batch_size", 8), "dl.batch_size"),
        epochs=_validate_positive_int(dcfg.get("epochs", 8), "dl.epochs"),
        mc_dropout_passes=_validate_positive_int(dcfg.get("mc_dropout_passes", 4), "dl.mc_dropout_passes"),
        sample_tile_size=_validate_positive_int(
            dcfg.get("sample_tile_size", dcfg.get("tile_size", 512)),
            "dl.sample_tile_size",
        ),
        max_patches=_validate_positive_int(dcfg.get("max_patches", 2500), "dl.max_patches"),
        aux_warmup_epochs=_validate_non_negative_int(dcfg.get("aux_warmup_epochs", 0), "dl.aux_warmup_epochs"),
        val_threshold=_validate_unit_interval(dcfg.get("val_threshold", 0.5), "dl.val_threshold"),
        selection_metric=_validate_choice(
            dcfg.get("selection_metric", "f1"),
            "dl.selection_metric",
            VALID_TRAIN_SELECTION_METRICS,
        ),
    )


def resolve_rl_loop_preflight(loop_cfg: dict[str, Any]) -> RLLoopPreflight:
    """Validate ``rl_loop`` scalars and the nested ``bandit`` sub-config."""
    return RLLoopPreflight(
        rounds=_validate_positive_int(loop_cfg.get("rounds", 2), "rl_loop.rounds"),
        patience=_validate_positive_int(loop_cfg.get("patience", 2), "rl_loop.patience"),
        min_delta=_validate_non_negative_float(loop_cfg.get("min_delta", 0.001), "rl_loop.min_delta"),
        selection_metric=_validate_choice(
            loop_cfg.get("selection_metric", "reward"),
            "rl_loop.selection_metric",
            VALID_RL_SELECTION_METRICS,
        ),
        bandit=_resolve_bandit_config(loop_cfg),
    )


def resolve_input_preflight(inp_cfg: dict[str, Any]) -> InputPreflight:
    """Validate the ``input`` boolean flags used by scene resolution."""
    return InputPreflight(
        prefer_v1=_validate_bool(inp_cfg.get("prefer_v1", True), "input.prefer_v1"),
        fallback_quick=_validate_bool(inp_cfg.get("fallback_quick", True), "input.fallback_quick"),
    )
