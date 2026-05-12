"""Feature extraction: spectral / texture priors, feature stacks, feedback.

Re-exports from :mod:`forestseg.features.stack` so that
``from forestseg.features import build_feature_stack`` keeps working.
"""

from __future__ import annotations

from .stack import build_feature_stack

__all__ = ["build_feature_stack"]
