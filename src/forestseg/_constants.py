"""Static configuration constants for the :mod:`forestseg` pipeline.

This module owns:

- :data:`WORK_FILES`: the canonical mapping from intermediate-artifact
  keys to their on-disk basenames inside ``cfg["work_dir"]``.
- :data:`REQUIRED_FUSION_GRID_KEYS`: ordered tuple of the grid-search
  parameters that must appear in ``cfg["fusion"]["grid"]`` for stage >=
  1 runs.
- :data:`REQUIRED_FUSION_FIXED_PARAM_KEYS`: ordered tuple of the fixed
  parameters that must appear in ``cfg["fusion"]["fixed_params"]``.
- :func:`wf`: tiny convenience helper that builds the on-disk path for
  a :data:`WORK_FILES` key inside a given working directory.

These names are also re-exported from :mod:`forestseg.cli` for
backward compatibility.
"""

from __future__ import annotations

import os

__all__ = [
    "FUSION_POSTPROCESS_PARAM_KEYS",
    "REQUIRED_FUSION_FIXED_PARAM_KEYS",
    "REQUIRED_FUSION_GRID_KEYS",
    "WORK_FILES",
    "wf",
]

WORK_FILES: dict[str, str] = {
    "input": "input_prepared.tif",
    "prob_spec": "prob_spec.tif",
    "conf_spec": "conf_spec.tif",
    "prob_tex": "prob_tex.tif",
    "tex_complexity": "tex_complexity.tif",
    "conf_tex": "conf_tex.tif",
    "feature_stack": "feature_stack.tif",
    "prob_dl": "prob_dl.tif",
    "unc_dl": "unc_dl.tif",
    "prob_fused": "prob_fused.tif",
    "mask_final": "mask_final.tif",
    "samples_train": "samples_train.json",
    "samples_val": "samples_val.json",
    "metrics_val": "metrics_val.json",
    "rl_history": "rl_history.json",
    "feature_feedback": "feature_feedback.json",
    "closed_loop_summary": "closed_loop_summary.json",
}

REQUIRED_FUSION_GRID_KEYS: tuple[str, ...] = (
    "lambda_spec",
    "lambda_tex",
    "threshold",
)

REQUIRED_FUSION_FIXED_PARAM_KEYS: tuple[str, ...] = (
    "lambda_spec",
    "lambda_tex",
    "threshold",
    "min_area_m2",
    "morph_kernel",
    "shadow_penalty",
)

# Postprocess-only subset of :data:`REQUIRED_FUSION_FIXED_PARAM_KEYS`:
# these keys live in ``fusion.fixed_params`` but participate in the
# morphology / vector-export step rather than the probabilistic fusion
# itself. Together with :data:`REQUIRED_FUSION_GRID_KEYS` they partition
# the full fixed-param key set.
FUSION_POSTPROCESS_PARAM_KEYS: tuple[str, ...] = (
    "min_area_m2",
    "morph_kernel",
    "shadow_penalty",
)


def wf(work_dir: str, key: str) -> str:
    """Return the on-disk path of intermediate artifact ``key``.

    ``key`` must be one of the keys in :data:`WORK_FILES`. A :class:`KeyError`
    is raised otherwise.
    """
    return os.path.join(work_dir, WORK_FILES[key])
