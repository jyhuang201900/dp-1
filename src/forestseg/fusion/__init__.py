"""Probabilistic fusion, configuration, postprocessing, and RL policy.

Re-exports from :mod:`forestseg.fusion.core` so that
``from forestseg.fusion import FusionParams`` keeps working.
"""

from __future__ import annotations

from .core import (
    FusionParams,
    fuse_probabilities,
    run_stage0,
    run_stage1_grid,
)

__all__ = [
    "FusionParams",
    "fuse_probabilities",
    "run_stage0",
    "run_stage1_grid",
]
