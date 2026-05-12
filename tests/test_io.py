"""Unit tests for :mod:`forestseg._io`.

The atomic writer is used by every critical pipeline artefact
(``rl_history.json``, ``closed_loop_summary.json``,
``fusion_selected.json``); these tests lock in its observable
contract — same end-state as a naive write on success, no on-disk
target left behind on failure — at the unit level.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forestseg._io import atomic_write_json, read_json, write_json


def test_write_json_round_trips_via_read_json(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    payload = {"a": [1, 2, 3], "b": "森林"}
    write_json(str(target), payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert read_json(str(target)) == payload


def test_atomic_write_json_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    payload = [{"id": 1}, {"id": 2, "tag": "森林"}]
    atomic_write_json(str(target), payload)
    assert read_json(str(target)) == payload


def test_atomic_write_json_creates_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "out.json"
    payload = {"ok": True}
    atomic_write_json(str(target), payload)
    assert read_json(str(target)) == payload


def test_atomic_write_json_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    target.write_text(json.dumps({"old": True}), encoding="utf-8")
    atomic_write_json(str(target), {"new": True})
    assert read_json(str(target)) == {"new": True}


def test_atomic_write_json_does_not_leave_temp_files_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "out.json"

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        atomic_write_json(str(target), {"bad": Unserializable()})

    assert not target.exists()
    leftover = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftover == [], f"temp files leaked: {leftover}"
