from __future__ import annotations

import numpy as np

from ..io.raster import normalized_valid_mask


def entropy_map(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return (-(p * np.log(p) + (1 - p) * np.log(1 - p))).astype(np.float32)


def build_state(
    prob_dl: np.ndarray,
    prob_spec: np.ndarray,
    prob_tex: np.ndarray,
    tex_complexity: np.ndarray,
    img01: np.ndarray,
) -> dict[str, float]:
    valid = (
        normalized_valid_mask(prob_dl)
        & normalized_valid_mask(prob_spec)
        & normalized_valid_mask(prob_tex)
        & normalized_valid_mask(tex_complexity)
        & normalized_valid_mask(img01)
    )
    if not np.any(valid):
        return {
            "mean_prob_dl": 0.0,
            "std_prob_dl": 0.0,
            "entropy": 0.0,
            "consistency_st": 0.0,
            "consistency_ds": 0.0,
            "consistency_dt": 0.0,
            "texture_complexity": 0.0,
            "shadow_score": 0.0,
        }

    ent = entropy_map(np.clip(prob_dl[valid], 1e-6, 1 - 1e-6))
    consistency_st = 1.0 - np.mean(np.abs(prob_spec[valid] - prob_tex[valid]))
    consistency_ds = 1.0 - np.mean(np.abs(prob_dl[valid] - prob_spec[valid]))
    consistency_dt = 1.0 - np.mean(np.abs(prob_dl[valid] - prob_tex[valid]))

    dark_score = float(np.mean(1.0 - img01[valid]))
    tex_score = float(np.mean(tex_complexity[valid]))

    return {
        "mean_prob_dl": float(np.mean(prob_dl[valid])),
        "std_prob_dl": float(np.std(prob_dl[valid])),
        "entropy": float(np.mean(ent)),
        "consistency_st": float(consistency_st),
        "consistency_ds": float(consistency_ds),
        "consistency_dt": float(consistency_dt),
        "texture_complexity": tex_score,
        "shadow_score": 0.6 * dark_score + 0.4 * (1.0 - tex_score),
    }
