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

Public sub-modules:

- :mod:`forestseg.cli`             — argparse-based command-line interface
- :mod:`forestseg.io_raster`       — raster I/O utilities and windowed iteration
- :mod:`forestseg.labels`          — vector label reading, validation, splitting
- :mod:`forestseg.features`        — multi-source feature stack assembly
- :mod:`forestseg.spectral`        — spectral prior derivation
- :mod:`forestseg.texture`         — texture prior derivation
- :mod:`forestseg.model`           — DeepLabV3+ segmentation model construction
- :mod:`forestseg.train`           — supervised training loop
- :mod:`forestseg.infer`           — sliding-window inference with MC dropout
- :mod:`forestseg.sampler`         — point-centred tile sampling
- :mod:`forestseg.fusion`          — DL/spectral/texture probability fusion
- :mod:`forestseg.postprocess`     — morphology + shadow penalty postprocess
- :mod:`forestseg.rl_env`          — RL state representation
- :mod:`forestseg.rl_policy`       — RL-driven parameter selection
- :mod:`forestseg.bandit`          — contextual bandit alternative
- :mod:`forestseg.feedback_loop`   — closed-loop feature feedback
- :mod:`forestseg.scene_runtime`   — scene resolution from ``scene.sh``
- :mod:`forestseg.metrics`         — binary classification metrics
- :mod:`forestseg.export`          — raster / vector export helpers

See :mod:`forestseg.cli` (or run ``python -m forestseg --help``) for the
top-level entry point.
"""

from __future__ import annotations

from ._version import __version__

__all__ = ["__version__"]
