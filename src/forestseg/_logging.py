"""Lightweight structured logging helpers used across :mod:`forestseg`.

The legacy CLI prints free-form ``[stage] message`` strings to stdout. The
helpers here keep that behaviour fully backwards-compatible while also feeding
the standard :mod:`logging` facility, so downstream callers / notebooks can
configure handlers as they like::

    import logging
    from forestseg._logging import configure_logging, get_logger

    configure_logging(level=logging.INFO)
    log = get_logger(__name__)
    log.info("preflight: starting")

Designed to be safe to import from anywhere; importing this module does **not**
mutate global logging state. Call :func:`configure_logging` explicitly.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TextIO

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger scoped under the ``forestseg`` namespace."""
    if name is None:
        return logging.getLogger("forestseg")
    if name == "forestseg" or name.startswith("forestseg."):
        return logging.getLogger(name)
    return logging.getLogger(f"forestseg.{name}")


def configure_logging(
    level: int | str | None = None,
    *,
    stream: TextIO | None = None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``forestseg`` root logger.

    Parameters
    ----------
    level:
        Logging level. May be passed as an int (``logging.INFO``) or a string
        (``"INFO"``). Falls back to the ``FORESTSEG_LOG_LEVEL`` environment
        variable, then to ``logging.INFO``.
    stream:
        Output stream. Defaults to :data:`sys.stderr`.
    fmt / datefmt:
        Formatter strings.
    force:
        If ``True``, remove any pre-existing handlers on the package logger
        before adding the new one.
    """
    if level is None:
        env_level = os.environ.get("FORESTSEG_LOG_LEVEL", "INFO")
        level = env_level.strip().upper() or logging.INFO

    if isinstance(level, str):
        level = logging.getLevelName(level)
        if not isinstance(level, int):
            level = logging.INFO

    logger = logging.getLogger("forestseg")
    logger.setLevel(level)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
