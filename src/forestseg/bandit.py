from __future__ import annotations

import json
import math
import os
import random
from itertools import product
from typing import Any

from ._io import atomic_write_json


def build_bandit_action_space(grid_params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for idx, (lambda_spec, lambda_tex, threshold) in enumerate(
        product(
            grid_params.get("lambda_spec", []),
            grid_params.get("lambda_tex", []),
            grid_params.get("threshold", []),
        )
    ):
        actions.append(
            {
                "id": idx,
                "lambda_spec": float(lambda_spec),
                "lambda_tex": float(lambda_tex),
                "threshold": float(threshold),
            }
        )
    if not actions:
        raise ValueError("bandit action space 不能为空。")
    return actions


def _build_action_space_signature(actions: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [
        {
            "lambda_spec": float(action["lambda_spec"]),
            "lambda_tex": float(action["lambda_tex"]),
            "threshold": float(action["threshold"]),
        }
        for action in actions
    ]


def _default_bandit_values(
    *,
    action_count: int,
    epsilon: float,
    min_epsilon: float,
    epsilon_decay: float,
    action_space_signature: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "steps": 0,
        "epsilon": float(epsilon),
        "min_epsilon": float(min_epsilon),
        "epsilon_decay": float(epsilon_decay),
        "q": [0.0 for _ in range(action_count)],
        "n": [0 for _ in range(action_count)],
        "action_space_signature": action_space_signature,
    }


def _normalize_list(values: Any, *, expected_len: int, cast_fn, fill_value, is_valid_fn=None):
    if not isinstance(values, list):
        return [fill_value for _ in range(expected_len)]
    normalized = []
    for idx in range(expected_len):
        if idx < len(values):
            try:
                parsed = cast_fn(values[idx])
            except (TypeError, ValueError):
                parsed = fill_value
            if is_valid_fn is not None and not is_valid_fn(parsed):
                parsed = fill_value
            normalized.append(parsed)
        else:
            normalized.append(fill_value)
    return normalized


def _clamp_unit_interval(value: float, *, fallback: float) -> float:
    if not math.isfinite(value):
        return fallback
    return min(max(value, 0.0), 1.0)


def load_or_init_bandit_state(
    *,
    state_path: str,
    actions: list[dict[str, Any]],
    epsilon: float,
    min_epsilon: float,
    epsilon_decay: float,
) -> dict[str, Any]:
    action_count = len(actions)
    action_space_signature = _build_action_space_signature(actions)
    defaults = _default_bandit_values(
        action_count=action_count,
        epsilon=epsilon,
        min_epsilon=min_epsilon,
        epsilon_decay=epsilon_decay,
        action_space_signature=action_space_signature,
    )
    if not os.path.exists(state_path):
        return defaults
    try:
        with open(state_path, encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(loaded, dict):
        return defaults

    loaded_signature = loaded.get("action_space_signature")
    if loaded_signature is not None and loaded_signature != action_space_signature:
        return defaults

    normalized_steps_raw = loaded.get("steps", 0)
    try:
        normalized_steps = int(normalized_steps_raw)
    except (TypeError, ValueError):
        normalized_steps = 0
    if normalized_steps < 0:
        normalized_steps = 0

    normalized_epsilon_raw = loaded.get("epsilon", epsilon)
    try:
        normalized_epsilon = float(normalized_epsilon_raw)
    except (TypeError, ValueError):
        normalized_epsilon = float(epsilon)
    normalized_epsilon = _clamp_unit_interval(normalized_epsilon, fallback=float(epsilon))

    normalized = {
        "schema_version": 1,
        "steps": normalized_steps,
        "epsilon": normalized_epsilon,
        "min_epsilon": float(min_epsilon),
        "epsilon_decay": float(epsilon_decay),
        "q": _normalize_list(
            loaded.get("q"),
            expected_len=action_count,
            cast_fn=float,
            fill_value=0.0,
            is_valid_fn=math.isfinite,
        ),
        "n": _normalize_list(
            loaded.get("n"),
            expected_len=action_count,
            cast_fn=int,
            fill_value=0,
            is_valid_fn=lambda value: value >= 0,
        ),
        "action_space_signature": action_space_signature,
    }
    return normalized


def select_bandit_action(
    *,
    state: dict[str, Any],
    actions: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, Any], bool]:
    epsilon = float(state.get("epsilon", 0.0))
    epsilon = min(max(epsilon, 0.0), 1.0)
    rng = random.Random(int(seed) + int(state.get("steps", 0)))
    q_values = [float(v) for v in state.get("q", [])]
    if len(q_values) != len(actions):
        q_values = [0.0 for _ in actions]
        state["q"] = q_values

    explore = rng.random() < epsilon
    if explore:
        chosen_idx = rng.randrange(len(actions))
    else:
        best_q = max(q_values)
        best_indices = [idx for idx, value in enumerate(q_values) if value == best_q]
        chosen_idx = best_indices[rng.randrange(len(best_indices))]
    return actions[chosen_idx], explore


def update_bandit_state(
    *,
    state: dict[str, Any],
    action_id: int,
    reward: float,
    alpha: float,
) -> dict[str, Any]:
    q_values = [float(v) for v in state.get("q", [])]
    n_values = [int(v) for v in state.get("n", [])]
    if action_id < 0 or action_id >= len(q_values) or action_id >= len(n_values):
        raise ValueError(f"bandit.action_id 越界：{action_id}")

    current_q = q_values[action_id]
    updated_q = current_q + float(alpha) * (float(reward) - current_q)
    q_values[action_id] = float(updated_q)
    n_values[action_id] = int(n_values[action_id]) + 1

    steps = int(state.get("steps", 0)) + 1
    epsilon_before = min(max(float(state.get("epsilon", 0.0)), 0.0), 1.0)
    min_epsilon = min(max(float(state.get("min_epsilon", 0.0)), 0.0), 1.0)
    epsilon_decay = float(state.get("epsilon_decay", 1.0))
    epsilon_after = max(min_epsilon, epsilon_before * epsilon_decay)

    updated = {
        "schema_version": 1,
        "steps": steps,
        "epsilon": float(epsilon_after),
        "min_epsilon": float(min_epsilon),
        "epsilon_decay": float(epsilon_decay),
        "q": q_values,
        "n": n_values,
    }
    action_space_signature = state.get("action_space_signature")
    if isinstance(action_space_signature, list):
        updated["action_space_signature"] = action_space_signature
    return updated


def save_bandit_state(state_path: str, state: dict[str, Any]) -> None:
    """Atomically write the bandit ``state`` to ``state_path``.

    The bandit state persists across runs and rounds, so a torn write
    would corrupt the policy for every subsequent round —
    :func:`forestseg._io.atomic_write_json` rules out that failure mode.
    """
    atomic_write_json(state_path, state)
