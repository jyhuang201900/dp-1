from __future__ import annotations

import numpy as np
from skimage.filters import threshold_local, threshold_otsu

from ..io.raster import normalized_valid_mask


def spectral_probability(
    img01: np.ndarray,
    quantile: float = 0.35,
    otsu_weight: float = 0.4,
    quantile_weight: float = 0.3,
    local_weight: float = 0.3,
    local_block_size: int = 51,
) -> tuple[np.ndarray, np.ndarray]:
    # KH-4 单波段经验先验：森林区通常相对偏暗（具体由后续融合校正）
    if local_block_size % 2 == 0:
        local_block_size += 1

    valid = normalized_valid_mask(img01)
    prob = np.zeros_like(img01, dtype=np.float32)
    conf = np.zeros_like(img01, dtype=np.float32)
    if not np.any(valid):
        return prob, conf

    valid_pixels = img01[valid]
    fill_value = float(np.median(valid_pixels))
    filled = np.where(valid, img01, fill_value).astype(np.float32)

    t_otsu = float(threshold_otsu(valid_pixels))
    t_q = float(np.quantile(valid_pixels, quantile))
    t_local = threshold_local(filled, block_size=local_block_size, method="gaussian")

    p_otsu = (filled <= t_otsu).astype(np.float32)
    p_q = (filled <= t_q).astype(np.float32)
    p_local = (filled <= t_local).astype(np.float32)

    total = max(otsu_weight + quantile_weight + local_weight, 1e-6)
    prob_valid = (otsu_weight * p_otsu + quantile_weight * p_q + local_weight * p_local) / total
    prob[valid] = np.clip(prob_valid[valid], 0.0, 1.0).astype(np.float32)
    conf[valid] = (2.0 * np.abs(prob[valid] - 0.5)).astype(np.float32)
    return prob, conf


def consistency(pred_prob: np.ndarray, spec_prob: np.ndarray) -> float:
    return float(1.0 - np.mean(np.abs(pred_prob - spec_prob)))
