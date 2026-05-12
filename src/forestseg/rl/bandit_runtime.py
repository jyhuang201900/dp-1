"""Per-call bandit orchestration glue for the RL-fusion CLI driver.

Three pieces of the bandit pipeline live in three intentionally
separate modules:

* :mod:`forestseg._bandit_config` — validates the ``rl_loop.bandit``
  section and returns a typed dict the driver consumes.
* :mod:`forestseg.bandit` — pure primitives for the action space,
  state load/init, action selection, online updates, and persistence.
* :mod:`forestseg._bandit_runtime` (this module) — the small stateful
  driver that :func:`forestseg.cli.cmd_run_rl_fusion` uses for a
  single fusion round: it tracks ``bandit_meta`` (the report block
  embedded into the fusion payload), narrows ``grid_params`` to the
  selected action's parameters, and applies the reward update when
  the validation score comes back.

This module is intentionally side-effect-free at import time and
holds no module-level state — every call constructs a fresh
:class:`BanditRuntime`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .bandit import (
    build_bandit_action_space,
    load_or_init_bandit_state,
    save_bandit_state,
    select_bandit_action,
    update_bandit_state,
)

__all__ = ["BanditRuntime"]


@dataclass
class BanditRuntime:
    """Single-call bandit driver for one ``run-rl-fusion`` invocation.

    Construction merely seeds ``meta`` and decides whether the bandit
    is active (stage-1 only + opt-in via config). The two real
    operations are :meth:`select_action` (called before fusion is
    evaluated) and :meth:`record_reward` (called after).
    """

    bandit_config: dict[str, Any]
    fusion_stage: int
    work: str

    enabled: bool = field(init=False)
    state_path: str = field(init=False)
    meta: dict[str, Any] = field(init=False)
    _state: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _action_id: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.enabled = bool(self.bandit_config["enabled"]) and self.fusion_stage == 1
        self.state_path = os.path.join(self.work, "bandit_state.json")
        self.meta = {
            "enabled": self.enabled,
            "action_id": None,
            "explore": None,
            "epsilon_before": None,
            "epsilon_after": None,
        }

    @property
    def policy_type(self) -> str:
        """``"bandit"`` when active, otherwise ``"grid"`` (greedy grid search)."""
        return "bandit" if self.enabled else "grid"

    def select_action(self, grid_params: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Pick an action and return the refined ``grid_params``.

        When the bandit is disabled this is a shallow copy of
        ``grid_params`` — preserving the original greedy-grid
        behaviour. When enabled, returns a copy in which
        ``lambda_spec`` / ``lambda_tex`` / ``threshold`` are narrowed
        to single-element lists carrying the selected action's
        values.
        """
        runtime = dict(grid_params)
        if not self.enabled:
            return runtime
        action_space = build_bandit_action_space(grid_params)
        state = load_or_init_bandit_state(
            state_path=self.state_path,
            actions=action_space,
            epsilon=float(self.bandit_config["epsilon"]),
            min_epsilon=float(self.bandit_config["min_epsilon"]),
            epsilon_decay=float(self.bandit_config["epsilon_decay"]),
        )
        selected_action, explore = select_bandit_action(
            state=state,
            actions=action_space,
            seed=int(self.bandit_config["seed"]),
        )
        self.meta["action_id"] = int(selected_action["id"])
        self.meta["explore"] = bool(explore)
        self.meta["epsilon_before"] = float(state.get("epsilon", self.bandit_config["epsilon"]))
        self._state = state
        self._action_id = int(selected_action["id"])
        runtime["lambda_spec"] = [float(selected_action["lambda_spec"])]
        runtime["lambda_tex"] = [float(selected_action["lambda_tex"])]
        runtime["threshold"] = [float(selected_action["threshold"])]
        return runtime

    def record_reward(self, reward: float) -> None:
        """Update bandit state with the realized reward and persist it.

        No-op when the bandit is disabled or when
        :meth:`select_action` was never invoked.
        """
        if not self.enabled or self._state is None or self._action_id is None:
            return
        updated = update_bandit_state(
            state=self._state,
            action_id=self._action_id,
            reward=reward,
            alpha=float(self.bandit_config["alpha"]),
        )
        save_bandit_state(self.state_path, updated)
        self.meta["epsilon_after"] = float(updated["epsilon"])
