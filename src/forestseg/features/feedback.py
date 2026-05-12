from __future__ import annotations

import numpy as np

from ..io.raster import normalized_valid_mask


def select_hard_examples(
    prob_dl: np.ndarray,
    prob_spec: np.ndarray,
    prob_tex: np.ndarray,
    uncertainty_quantile: float = 0.85,
    consistency_quantile: float = 0.2,
) -> np.ndarray:
    valid = normalized_valid_mask(prob_dl) & normalized_valid_mask(prob_spec) & normalized_valid_mask(prob_tex)
    hard = np.zeros(prob_dl.shape, dtype=np.uint8)
    if not np.any(valid):
        return hard

    uncertainty = 1.0 - (2.0 * np.abs(prob_dl - 0.5))
    consistency = 1.0 - (np.abs(prob_dl - prob_spec) + np.abs(prob_dl - prob_tex)) / 2.0

    u_thr = float(np.quantile(uncertainty[valid], uncertainty_quantile))
    c_thr = float(np.quantile(consistency[valid], consistency_quantile))

    hard[valid] = ((uncertainty[valid] >= u_thr) | (consistency[valid] <= c_thr)).astype(np.uint8)
    return hard


def select_high_conf_pseudolabel(
    prob_dl: np.ndarray,
    prob_spec: np.ndarray,
    prob_tex: np.ndarray,
    high: float = 0.9,
    low: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    valid_pixels = normalized_valid_mask(prob_dl) & normalized_valid_mask(prob_spec) & normalized_valid_mask(prob_tex)
    agree_high = valid_pixels & (prob_dl >= high) & (prob_spec >= high) & (prob_tex >= high)
    agree_low = valid_pixels & (prob_dl <= low) & (prob_spec <= low) & (prob_tex <= low)

    pseudo = np.full(prob_dl.shape, 255, dtype=np.uint8)  # 255 = ignore
    pseudo[agree_high] = 1
    pseudo[agree_low] = 0

    valid = (pseudo != 255).astype(np.uint8)
    return pseudo, valid
