"""Small, pure scalar validators used across :mod:`forestseg.cli`.

These helpers exist to keep ``cli.py`` short and focused on command
handlers. They raise :class:`ValueError` with a localized message on
invalid input and otherwise return the coerced value.

Every function here is import-safe (no side effects) and has no
dependency on :mod:`forestseg.cli`, so they can also be imported from
other internal modules such as :mod:`forestseg._rl_history` without
risking a circular import.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_ratio(value: Any, label: str) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} 必须在 0 和 1 之间，当前为 {value}。") from None
    if not np.isfinite(ratio):
        raise ValueError(f"{label} 必须为有限数，当前为 {ratio}。")
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(f"{label} 必须在 0 和 1 之间，当前为 {ratio}。")
    return ratio


def _validate_positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须为正数，当前为 {value}。")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} 必须为正数，当前为 {value}。") from None
    if not np.isfinite(parsed):
        raise ValueError(f"{label} 必须为有限数，当前为 {parsed}。")
    if parsed <= 0.0:
        raise ValueError(f"{label} 必须大于 0，当前为 {parsed}。")
    return parsed


def _validate_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须为正整数，当前为 {value}。")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{label} 必须为正整数，当前为 {value}。")
        parsed = int(value)
    else:
        raw = str(value).strip()
        if not raw:
            raise ValueError(f"{label} 必须为正整数，当前为空。")
        if raw.startswith(("+", "-")):
            digits = raw[1:]
        else:
            digits = raw
        if not digits.isdigit():
            raise ValueError(f"{label} 必须为正整数，当前为 {value}。")
        parsed = int(raw)
    if parsed <= 0:
        raise ValueError(f"{label} 必须大于 0，当前为 {parsed}。")
    return parsed


def _validate_choice(value: Any, label: str, valid_values: set[str]) -> str:
    choice = str(value).strip().lower()
    if choice not in valid_values:
        allowed = ", ".join(sorted(valid_values))
        raise ValueError(f"{label} 必须为以下之一：{allowed}。当前为 {value}。")
    return choice


def _validate_unit_interval(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须在 0 和 1 之间，当前为 {value}。")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} 必须在 0 和 1 之间，当前为 {value}。") from None
    if not np.isfinite(parsed):
        raise ValueError(f"{label} 必须为有限数，当前为 {parsed}。")
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{label} 必须在 0 和 1 之间，当前为 {parsed}。")
    return parsed


def _validate_non_negative_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须为非负数，当前为 {value}。")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} 必须为非负数，当前为 {value}。") from None
    if not np.isfinite(parsed):
        raise ValueError(f"{label} 必须为有限数，当前为 {parsed}。")
    if parsed < 0.0:
        raise ValueError(f"{label} 必须大于等于 0，当前为 {parsed}。")
    return parsed


def _validate_non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须为非负整数，当前为 {value}。")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{label} 必须为非负整数，当前为 {value}。")
        parsed = int(value)
    else:
        raw = str(value).strip()
        if not raw:
            raise ValueError(f"{label} 必须为非负整数，当前为空。")
        if raw.startswith(("+", "-")):
            digits = raw[1:]
        else:
            digits = raw
        if not digits.isdigit():
            raise ValueError(f"{label} 必须为非负整数，当前为 {value}。")
        parsed = int(raw)
    if parsed < 0:
        raise ValueError(f"{label} 必须大于等于 0，当前为 {parsed}。")
    return parsed


def _validate_stage_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须为整数，当前为 {value}。")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError(f"{label} 必须为整数，当前为 {value}。")
        parsed = int(value)
    else:
        raw = str(value).strip()
        if not raw:
            raise ValueError(f"{label} 必须为整数，当前为空。")
        if raw.startswith(("+", "-")):
            digits = raw[1:]
        else:
            digits = raw
        if not digits.isdigit():
            raise ValueError(f"{label} 必须为整数，当前为 {value}。")
        parsed = int(raw)
    if parsed not in {0, 1}:
        raise ValueError(f"{label} 仅支持 0 或 1，当前为 {parsed}。")
    return parsed


def _validate_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{label} 必须为布尔值，当前为 {value}。")
