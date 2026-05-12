from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects

from .io_raster import normalized_valid_mask


def _aspect_ratio(minr: int, minc: int, maxr: int, maxc: int) -> float:
    h = max(maxr - minr, 1)
    w = max(maxc - minc, 1)
    return float(max(h / w, w / h))


def shadow_connected_penalty(
    mask: np.ndarray,
    img01: np.ndarray,
    tex_complexity: np.ndarray,
    penalty_strength: float = 0.5,
) -> np.ndarray:
    valid = normalized_valid_mask(img01) & normalized_valid_mask(tex_complexity)
    if penalty_strength <= 0 or not np.any(valid):
        return mask & valid

    out = mask & valid
    lbl = label(out, connectivity=2)
    if lbl.max() == 0:
        return out

    q_dark = float(np.quantile(img01[valid], 0.35))
    q_tex = float(np.quantile(tex_complexity[valid], 0.4))

    for r in regionprops(lbl, intensity_image=img01):
        rr, cc = r.coords[:, 0], r.coords[:, 1]
        mean_i = float(np.mean(img01[rr, cc]))
        mean_t = float(np.mean(tex_complexity[rr, cc]))
        aspect = _aspect_ratio(*r.bbox)
        ecc = float(r.eccentricity)

        shadow_like = (mean_i <= q_dark) and (mean_t <= q_tex) and (aspect >= 3.0 or ecc >= 0.95)
        if shadow_like and penalty_strength >= 0.4:
            out[rr, cc] = False

    return out


def postprocess_mask(
    prob_fused: np.ndarray,
    img01: np.ndarray,
    tex_complexity: np.ndarray,
    threshold: float,
    pixel_area_m2: float,
    min_area_m2: float,
    opening_kernel: int,
    closing_kernel: int,
    shadow_penalty: float,
) -> np.ndarray:
    valid = normalized_valid_mask(prob_fused) & normalized_valid_mask(img01) & normalized_valid_mask(tex_complexity)
    mask = (prob_fused >= threshold) & valid

    if opening_kernel > 0:
        mask = binary_opening(mask, disk(int(opening_kernel))) & valid
    if closing_kernel > 0:
        mask = binary_closing(mask, disk(int(closing_kernel))) & valid

    min_pixels = max(int(min_area_m2 / max(pixel_area_m2, 1e-6)), 1)
    mask = remove_small_objects(mask, min_size=min_pixels)
    mask = shadow_connected_penalty(mask, img01, tex_complexity, penalty_strength=shadow_penalty) & valid

    return mask.astype(np.uint8)
