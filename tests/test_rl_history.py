"""Direct unit tests for ``forestseg.cli._load_rl_history`` /
``_validate_rl_history_entry``.

The integration tests in ``tests/test_cli_preflight.py`` exercise these
helpers transitively through ``cmd_run_rl_fusion``. This module locks
the behaviour in at the unit level so regressions surface with a
shorter failure trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forestseg.pipeline.cli import (
    LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS,
    _load_rl_history,
    _validate_rl_history_entry,
)


def _valid_entry() -> dict[str, object]:
    return {
        "selection_metric": "reward",
        "params": dict(LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS),
        "validation_metrics": {"reward": 0.5},
        "reward": 0.5,
        "score": 0.5,
    }


def test_load_rl_history_returns_empty_when_file_missing(tmp_path: Path) -> None:
    assert _load_rl_history(str(tmp_path / "does-not-exist.json")) == []


def test_load_rl_history_rejects_non_list_payload(tmp_path: Path) -> None:
    path = tmp_path / "rl_history.json"
    path.write_text(json.dumps({"params": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="rl_history"):
        _load_rl_history(str(path))


def test_load_rl_history_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "rl_history.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="rl_history"):
        _load_rl_history(str(path))


def test_load_rl_history_rejects_non_dict_entries(tmp_path: Path) -> None:
    path = tmp_path / "rl_history.json"
    path.write_text(json.dumps(["oops"]), encoding="utf-8")
    with pytest.raises(ValueError, match="rl_history"):
        _load_rl_history(str(path))


def test_load_rl_history_rejects_empty_dict_entries(tmp_path: Path) -> None:
    """An empty entry carries no information and must fail fast.

    Regression test for the case that previously slipped past validation
    via legacy-default normalization.
    """
    path = tmp_path / "rl_history.json"
    path.write_text(json.dumps([{}]), encoding="utf-8")
    with pytest.raises(ValueError, match="rl_history"):
        _load_rl_history(str(path))


def test_load_rl_history_accepts_valid_payload(tmp_path: Path) -> None:
    path = tmp_path / "rl_history.json"
    path.write_text(json.dumps([_valid_entry()]), encoding="utf-8")
    entries = _load_rl_history(str(path))
    assert len(entries) == 1
    assert entries[0]["selection_metric"] == "reward"
    assert entries[0]["validation_metrics"]["reward"] == pytest.approx(0.5)


def test_validate_rl_history_entry_rejects_non_dict() -> None:
    with pytest.raises(ValueError, match="rl_history"):
        _validate_rl_history_entry("not a dict", "history.json")


def test_validate_rl_history_entry_rejects_empty_dict() -> None:
    with pytest.raises(ValueError, match="rl_history"):
        _validate_rl_history_entry({}, "history.json")


def test_validate_rl_history_entry_downgrades_unknown_selection_metric() -> None:
    """Unknown legacy selection_metric is silently coerced to ``reward``.

    This is intentional permissive behaviour for payloads written by
    older `dp` versions that may carry retired metric names.
    """
    entry = _valid_entry()
    entry["selection_metric"] = "not-a-metric"
    normalized = _validate_rl_history_entry(entry, "history.json")
    assert normalized["selection_metric"] == "reward"


def test_validate_rl_history_entry_rejects_negative_reward() -> None:
    entry = _valid_entry()
    entry["validation_metrics"] = {"reward": -0.1}
    entry["reward"] = -0.1
    entry["score"] = -0.1
    with pytest.raises(ValueError, match="rl_history"):
        _validate_rl_history_entry(entry, "history.json")


def test_validate_rl_history_entry_rejects_mismatched_top_level_reward() -> None:
    entry = _valid_entry()
    entry["reward"] = 0.9  # not close to validation_metrics.reward (0.5)
    with pytest.raises(ValueError, match="rl_history"):
        _validate_rl_history_entry(entry, "history.json")


def test_validate_rl_history_entry_normalizes_legacy_payload() -> None:
    """A pre-`selection_metric` legacy entry should normalize to ``reward``."""
    legacy_entry = {
        "params": dict(LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS),
        "validation_metrics": {"reward": 0.25},
        "reward": 0.25,
    }
    normalized = _validate_rl_history_entry(legacy_entry, "history.json")
    assert normalized["selection_metric"] == "reward"
    assert normalized["validation_metrics"]["reward"] == pytest.approx(0.25)
    assert normalized["score"] == pytest.approx(0.25)
