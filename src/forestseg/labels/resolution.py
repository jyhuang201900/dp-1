"""Validation and shared setup for the ``labels`` config section.

Supports both the modern ``positive_path`` + ``negative_path`` shape and
the legacy single-``path`` shape. The preflight checks here are
intentionally minimal — full content validation (CRS, geometry, label
domain) lives in :mod:`forestseg.labels`.

In addition to the schema-level helpers (``_resolve_label_paths`` /
``_resolve_label_mode`` / ``_require_labels_for_preflight``), this
module owns the typed :class:`LabelSpec` and :class:`LabelGeometry`
bundles plus :func:`resolve_label_spec`. The matching read / validate
dispatchers intentionally stay in :mod:`forestseg.cli` so that
test-time monkeypatching of ``forestseg.cli.validate_label_points`` /
``forestseg.cli.read_label_points`` (and their dual-file siblings)
continues to redirect the real call sites — those bindings are looked
up against the ``forestseg.cli`` module globals at call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.paths import _require_existing_path
from ..core.validators import (
    _validate_non_negative_int,
    _validate_positive_float,
    _validate_ratio,
)

__all__ = [
    "LabelGeometry",
    "LabelSpec",
    "_require_labels_for_preflight",
    "_resolve_label_mode",
    "_resolve_label_paths",
    "resolve_label_spec",
]


@dataclass(frozen=True)
class LabelGeometry:
    """Raster-derived georeferencing for the snap grid used by label IO."""

    crs: Any
    origin_x: float
    origin_y: float


@dataclass(frozen=True)
class LabelSpec:
    """Validated ``labels`` config bundle shared by prepare / check commands."""

    mode: str
    positive_path: str | None
    negative_path: str | None
    single_path: str | None
    class_field: str
    positive_value: str
    negative_value: str
    grid_size: float
    train_ratio: float
    split_seed: int
    layer: str | None


def _resolve_label_paths(lcfg: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Return ``(positive_path, negative_path, single_path)`` from ``lcfg``.

    Each element is ``None`` if the corresponding key is missing or
    blank.
    """
    positive_path = str(lcfg.get("positive_path") or "").strip() or None
    negative_path = str(lcfg.get("negative_path") or "").strip() or None
    single_path = str(lcfg.get("path") or "").strip() or None
    return positive_path, negative_path, single_path


def _resolve_label_mode(lcfg: dict[str, Any]) -> tuple[str | None, str | None, str | None, str]:
    """Determine which label-path schema is in use.

    Returns ``(positive_path, negative_path, single_path, mode)`` where
    ``mode`` is one of ``"dual"`` (positive + negative) or ``"legacy"``
    (single ``path``). Raises :class:`ValueError` for partial /
    inconsistent configurations.
    """
    positive_path, negative_path, single_path = _resolve_label_paths(lcfg)
    if positive_path and negative_path:
        return positive_path, negative_path, single_path, "dual"
    if positive_path or negative_path:
        raise ValueError("labels.positive_path 与 labels.negative_path 必须同时配置，不能只配置一个。")
    if single_path:
        return positive_path, negative_path, single_path, "legacy"
    raise ValueError("缺少 labels.positive_path + labels.negative_path（或兼容字段 labels.path）配置。")


def _require_labels_for_preflight(lcfg: dict[str, Any]) -> dict[str, str]:
    """Resolve the labels schema and assert each referenced file exists.

    Returns a small ``{"labels_*_path": str}`` mapping suitable for
    embedding into the preflight summary.
    """
    positive_path, negative_path, single_path, mode = _resolve_label_mode(lcfg)
    if mode == "dual":
        return {
            "labels_positive_path": _require_existing_path(str(positive_path), "labels.positive_path"),
            "labels_negative_path": _require_existing_path(str(negative_path), "labels.negative_path"),
        }
    return {
        "labels_path": _require_existing_path(str(single_path), "labels.path"),
    }


def resolve_label_spec(lcfg: dict[str, Any]) -> LabelSpec:
    """Validate ``lcfg`` and return a typed :class:`LabelSpec`.

    Raises ``ValueError`` for any out-of-range / wrong-type values, or
    for partial dual-mode configuration.
    """
    positive_path, negative_path, single_path, mode = _resolve_label_mode(lcfg)
    return LabelSpec(
        mode=mode,
        positive_path=positive_path,
        negative_path=negative_path,
        single_path=single_path,
        class_field=str(lcfg.get("class_field", "class")),
        positive_value=str(lcfg.get("positive_value", "1")),
        negative_value=str(lcfg.get("negative_value", "0")),
        grid_size=_validate_positive_float(lcfg.get("grid_size", 1000.0), "labels.grid_size"),
        train_ratio=_validate_ratio(lcfg.get("split_ratio", 0.7), "labels.split_ratio"),
        split_seed=_validate_non_negative_int(lcfg.get("split_seed", 42), "labels.split_seed"),
        layer=lcfg.get("layer"),
    )
