from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import local_binary_pattern

from .io_raster import normalized_valid_mask


def _local_variance(img01: np.ndarray, win: int) -> np.ndarray:
    mean = uniform_filter(img01, size=win, mode="reflect")
    mean2 = uniform_filter(img01 * img01, size=win, mode="reflect")
    return np.clip(mean2 - mean * mean, 0.0, None).astype(np.float32)


def _norm01(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    if not np.any(valid):
        return out
    vals = arr[valid]
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        return out
    clipped = np.clip(arr, lo, hi)
    out[valid] = ((clipped[valid] - lo) / (hi - lo)).astype(np.float32)
    return out


def texture_probability(
    img01: np.ndarray,
    windows: list[int] | tuple[int, ...] = (7, 15, 31),
    lbp_radius: int = 1,
    lbp_points: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    valid = normalized_valid_mask(img01)
    prob = np.zeros_like(img01, dtype=np.float32)
    complexity = np.zeros_like(img01, dtype=np.float32)
    if not np.any(valid):
        return prob, complexity

    fill_value = float(np.median(img01[valid]))
    filled = np.where(valid, img01, fill_value).astype(np.float32)

    vars_multi = []
    for w in windows:
        ww = int(w)
        if ww < 3:
            continue
        if ww % 2 == 0:
            ww += 1
        vars_multi.append(_local_variance(filled, ww))

    if not vars_multi:
        vars_multi = [_local_variance(filled, 7)]

    var_mean = np.mean(np.stack(vars_multi, axis=0), axis=0)
    var_norm = _norm01(var_mean, valid)

    lbp = local_binary_pattern((filled * 255).astype(np.uint8), lbp_points, lbp_radius, method="uniform")
    lbp_norm = _norm01(lbp.astype(np.float32), valid)

    prob[valid] = np.clip(0.7 * var_norm[valid] + 0.3 * lbp_norm[valid], 0.0, 1.0).astype(np.float32)
    complexity[valid] = var_norm[valid]
    return prob, complexity * (2.0 * np.abs(prob - 0.5)).astype(np.float32)


def consistency(pred_prob: np.ndarray, tex_prob: np.ndarray) -> float:
    return float(1.0 - np.mean(np.abs(pred_prob - tex_prob)))
