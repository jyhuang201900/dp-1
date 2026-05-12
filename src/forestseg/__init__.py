"""
forestseg — KH-4 supervised forest extraction pipeline.

This package provides a modular end-to-end workflow:

    raw raster + labels + config
      -> preprocess
      -> spectral / texture priors
      -> label QA + grid-aware split
      -> feature stack
      -> supervised DL training & sliding-window inference
      -> probabilistic fusion (DL + spectral + texture) with grid / RL search
      -> postprocess (morphology, small-component removal, shadow penalty)
      -> raster + vector export

Sub-packages (canonical locations):

- :mod:`forestseg.core`             — constants, IO helpers, logging, paths, validators
- :mod:`forestseg.io`               — raster I/O utilities and scene resolution
- :mod:`forestseg.labels`           — vector label reading, validation, splitting
- :mod:`forestseg.features`         — spectral / texture priors, feature stacks, feedback
- :mod:`forestseg.dl`               — DeepLabV3+ model, training, inference, sampling
- :mod:`forestseg.fusion`           — probabilistic fusion, postprocessing, RL policy
- :mod:`forestseg.rl`               — contextual bandit, RL environment, history tracking
- :mod:`forestseg.pipeline`         — CLI, preflight, artifacts, closed-loop driver
- :mod:`forestseg.export_pkg`       — raster / vector export helpers, metrics

Backward-compatible stub modules at the old flat paths (e.g.
``forestseg.cli``, ``forestseg.io_raster``) re-export the public API
from the new sub-packages so existing imports keep working.

See :mod:`forestseg.pipeline.cli` (or run ``python -m forestseg --help``)
for the top-level entry point.
"""

from __future__ import annotations

from ._version import __version__

__all__ = ["__version__"]
