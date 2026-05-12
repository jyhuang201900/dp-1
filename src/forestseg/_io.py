"""Tiny JSON read / write helpers shared by :mod:`forestseg` modules.

These helpers exist to do two things consistently:

1. Eliminate the boilerplate ``with open(...) as f: json.dump/load`` triplets
   that were repeated across the CLI and its private siblings.
2. Provide an *atomic* writer for critical pipeline artefacts
   (``rl_history.json``, ``closed_loop_summary.json``,
   ``fusion_selected.json``). A write that crashes mid-flight would
   otherwise leave a truncated file on disk that downstream stages
   (e.g. strict ``rl_history`` validation) would later reject — and
   because the closed-loop driver re-uses the same path between rounds,
   such a corrupt file blocks every subsequent run.

The atomic writer is intentionally minimal: write to a temp file
alongside ``path`` (on the same filesystem, so :func:`os.replace`
is atomic), :func:`os.fsync` the temp file's data + directory, then
:func:`os.replace` over the target. Callers that want best-effort
non-atomic writes (e.g. metrics for ad-hoc inspection) can keep using
:func:`write_json`.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from typing import Any

__all__ = [
    "atomic_write_json",
    "read_json",
    "write_json",
]


def read_json(path: str) -> Any:
    """Return ``json.load`` of the UTF-8 file at ``path``."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any, *, indent: int = 2) -> str:
    """Write ``obj`` as UTF-8-encoded JSON to ``path``; return ``path``.

    Not atomic — a crash mid-write may leave a truncated file. Prefer
    :func:`atomic_write_json` for artefacts that downstream stages
    re-read.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return path


def atomic_write_json(path: str, obj: Any, *, indent: int = 2) -> str:
    """Atomically write ``obj`` as UTF-8 JSON to ``path``; return ``path``.

    Implementation: write to a uniquely-named temp file in the *same*
    directory (so :func:`os.replace` is atomic on POSIX/NTFS), fsync
    the data, then :func:`os.replace` over the destination. The temp
    file is removed on failure to avoid leaving stray files behind.
    """
    target_dir = os.path.dirname(path) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=target_dir,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
            f.flush()
            # fsync is best-effort: some filesystems (notably network
            # mounts) reject it. The os.replace below is still atomic
            # w.r.t. concurrent readers, so a missing fsync only widens
            # the durability window (not the consistency window).
            with contextlib.suppress(OSError):
                os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_path)
        raise
    return path
