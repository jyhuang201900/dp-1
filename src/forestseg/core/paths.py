"""Small filesystem + IO helpers shared by :mod:`forestseg.cli` commands.

Everything here is intentionally side-effect-free at import time and
has no dependency on other :mod:`forestseg` internals beyond stdlib
modules, so it can be re-used from other private modules without
risking a circular import.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any

import yaml

__all__ = [
    "_ensure_directory_writable",
    "_log_progress",
    "_require_existing_path",
    "ensure_work",
    "load_cfg",
]


def load_cfg(path: str) -> dict[str, Any]:
    """Load and return the YAML pipeline configuration at ``path``."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_work(cfg: dict[str, Any]) -> str:
    """Create ``cfg["work_dir"]`` (and its ``checkpoints/`` subdir) and return it."""
    work = cfg["work_dir"]
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.join(work, "checkpoints"), exist_ok=True)
    return work


def _log_progress(message: str) -> None:
    """Emit a single ``[progress] ...`` line on stderr (used by command handlers)."""
    print(f"[progress] {message}", file=sys.stderr, flush=True)


def _require_existing_path(path: str, label: str) -> str:
    """Return ``path`` unchanged after asserting it is configured and exists.

    Raises :class:`ValueError` if ``path`` is empty / falsy and
    :class:`FileNotFoundError` if it does not exist on disk.
    """
    if not path:
        raise ValueError(f"缺少 {label} 配置。")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} 不存在：{path}")
    return path


def _ensure_directory_writable(path: str, label: str) -> str:
    """Create ``path`` if missing and assert it is a writable directory.

    Probes write access by creating and deleting a temporary file under
    ``path``. Raises :class:`ValueError` (missing config),
    :class:`NotADirectoryError` (path exists but is not a directory),
    or :class:`PermissionError` (not writable).
    """
    if not path:
        raise ValueError(f"缺少 {label} 配置。")
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{label} 不是目录：{path}")
    probe_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".preflight-write-", suffix=".tmp", delete=False) as probe:
            probe_path = probe.name
    except OSError as exc:
        raise PermissionError(f"{label} 不可写：{path}") from exc
    finally:
        if probe_path and os.path.exists(probe_path):
            try:
                os.remove(probe_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
    return path
