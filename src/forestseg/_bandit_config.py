"""Validation of the ``rl_loop.bandit`` section of the pipeline config."""

from __future__ import annotations

from typing import Any

from ._validators import _validate_bool, _validate_positive_int, _validate_unit_interval

__all__ = ["_resolve_bandit_config"]


def _resolve_bandit_config(loop_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized ``rl_loop.bandit`` config dict.

    Defaults are applied for any missing keys; the result always
    contains all of ``enabled``, ``epsilon``, ``min_epsilon``,
    ``epsilon_decay``, ``alpha``, ``seed``. Raises :class:`ValueError`
    on invalid value types or if ``min_epsilon > epsilon``.
    """
    bandit_cfg = loop_cfg.get("bandit", {})
    if bandit_cfg is None:
        bandit_cfg = {}
    if not isinstance(bandit_cfg, dict):
        raise ValueError("rl_loop.bandit 必须为对象配置。")
    enabled = _validate_bool(bandit_cfg.get("enabled", True), "rl_loop.bandit.enabled")
    epsilon = _validate_unit_interval(bandit_cfg.get("epsilon", 0.2), "rl_loop.bandit.epsilon")
    min_epsilon = _validate_unit_interval(bandit_cfg.get("min_epsilon", 0.05), "rl_loop.bandit.min_epsilon")
    epsilon_decay = _validate_unit_interval(bandit_cfg.get("epsilon_decay", 0.95), "rl_loop.bandit.epsilon_decay")
    alpha = _validate_unit_interval(bandit_cfg.get("alpha", 0.3), "rl_loop.bandit.alpha")
    seed = _validate_positive_int(bandit_cfg.get("seed", 42), "rl_loop.bandit.seed")
    if min_epsilon > epsilon:
        raise ValueError("rl_loop.bandit.min_epsilon 不能大于 rl_loop.bandit.epsilon。")
    return {
        "enabled": enabled,
        "epsilon": epsilon,
        "min_epsilon": min_epsilon,
        "epsilon_decay": epsilon_decay,
        "alpha": alpha,
        "seed": seed,
    }
