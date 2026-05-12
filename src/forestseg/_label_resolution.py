"""Validation of the ``labels`` section of the pipeline config.

Supports both the modern ``positive_path`` + ``negative_path`` shape
and the legacy single-``path`` shape. The preflight checks here are
intentionally minimal — full content validation (CRS, geometry, label
domain) lives in :mod:`forestseg.labels`.
"""

from __future__ import annotations

from typing import Any

from ._paths import _require_existing_path

__all__ = [
    "_require_labels_for_preflight",
    "_resolve_label_mode",
    "_resolve_label_paths",
]


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
