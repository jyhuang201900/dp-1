from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from skimage.measure import label

from .io_raster import normalized_valid_mask

# Defaults applied by :meth:`FusionParams.from_mapping` when neither
# the source mapping nor the caller-supplied ``defaults`` mapping
# provides a value for a key. These reproduce the historical
# stage-0 / legacy-rl-history defaults so existing payloads remain
# round-trippable.
_FUSION_PARAM_DEFAULTS: dict[str, float] = {
    "lambda_spec": 0.2,
    "lambda_tex": 0.2,
    "threshold": 0.5,
    "min_area_m2": 200.0,
    "morph_kernel": 3,
    "shadow_penalty": 0.5,
}


@dataclass
class FusionParams:
    lambda_spec: float
    lambda_tex: float
    threshold: float
    min_area_m2: float
    morph_kernel: int
    shadow_penalty: float

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        defaults: Mapping[str, Any] | None = None,
    ) -> FusionParams:
        """Build a :class:`FusionParams` from a dict-like ``mapping``.

        Missing keys fall back first to ``defaults`` (when provided)
        and then to the module-level :data:`_FUSION_PARAM_DEFAULTS`.
        Numeric coercion is centralised here so callers don't have to
        spell out ``float(...) / int(...)`` per field — keeping the
        construction logic consistent across the CLI, fusion driver
        and RL policy.
        """
        src: Mapping[str, Any] = mapping or {}
        fallback: Mapping[str, Any] = defaults or {}

        def _pick(key: str) -> Any:
            if key in src:
                return src[key]
            if key in fallback:
                return fallback[key]
            return _FUSION_PARAM_DEFAULTS[key]

        return cls(
            lambda_spec=float(_pick("lambda_spec")),
            lambda_tex=float(_pick("lambda_tex")),
            threshold=float(_pick("threshold")),
            min_area_m2=float(_pick("min_area_m2")),
            morph_kernel=int(_pick("morph_kernel")),
            shadow_penalty=float(_pick("shadow_penalty")),
        )


def _weights(lambda_spec: float, lambda_tex: float) -> tuple[float, float, float]:
    ls = float(lambda_spec)
    lt = float(lambda_tex)
    ld = 1.0 - ls - lt
    if ld < 0:
        s = ls + lt
        if s <= 0:
            ls, lt = 0.0, 0.0
            ld = 1.0
        else:
            ls = ls / s * 0.5
            lt = lt / s * 0.5
            ld = 1.0 - ls - lt
    return ld, ls, lt


def fuse_probabilities(prob_dl: np.ndarray, prob_spec: np.ndarray, prob_tex: np.ndarray, p: FusionParams) -> np.ndarray:
    ld, ls, lt = _weights(p.lambda_spec, p.lambda_tex)
    valid = normalized_valid_mask(prob_dl) & normalized_valid_mask(prob_spec) & normalized_valid_mask(prob_tex)
    out = np.zeros_like(prob_dl, dtype=np.float32)
    if not np.any(valid):
        return out
    fused = ld * prob_dl + ls * prob_spec + lt * prob_tex
    out[valid] = np.clip(fused[valid], 0.0, 1.0).astype(np.float32)
    return out


def _fragmentation(mask: np.ndarray, valid: np.ndarray) -> float:
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return 0.0
    lbl = label(mask & valid, connectivity=2)
    n = int(lbl.max())
    return float(n / max(valid_count, 1))


def _objective(
    fused: np.ndarray,
    prob_dl: np.ndarray,
    prob_spec: np.ndarray,
    prob_tex: np.ndarray,
    threshold: float,
    shadow_score: np.ndarray,
    shadow_penalty: float,
) -> float:
    valid = (
        normalized_valid_mask(fused)
        & normalized_valid_mask(prob_dl)
        & normalized_valid_mask(prob_spec)
        & normalized_valid_mask(prob_tex)
        & normalized_valid_mask(shadow_score)
    )
    if not np.any(valid):
        return -1e18

    c1 = 1.0 - float(np.mean(np.abs(fused[valid] - prob_dl[valid])))
    c2 = 1.0 - float(np.mean(np.abs(fused[valid] - prob_spec[valid])))
    c3 = 1.0 - float(np.mean(np.abs(fused[valid] - prob_tex[valid])))

    mask = (fused >= threshold) & valid
    frag = _fragmentation(mask, valid)
    shadow_fp = float(np.mean(mask[valid].astype(np.float32) * shadow_score[valid]))

    return 0.45 * c1 + 0.275 * c2 + 0.275 * c3 - 2.0 * frag - shadow_penalty * shadow_fp


def run_stage0(
    prob_dl: np.ndarray,
    prob_spec: np.ndarray,
    prob_tex: np.ndarray,
    shadow_score: np.ndarray,
    fixed_params: dict,
) -> tuple[np.ndarray, FusionParams]:
    p = FusionParams.from_mapping(fixed_params)
    fused = fuse_probabilities(prob_dl, prob_spec, prob_tex, p)
    return fused, p


def run_stage1_grid(
    prob_dl: np.ndarray,
    prob_spec: np.ndarray,
    prob_tex: np.ndarray,
    shadow_score: np.ndarray,
    grid: dict,
) -> tuple[np.ndarray, FusionParams, float]:
    best_score = -1e18
    best_p: FusionParams | None = None
    best_fused: np.ndarray | None = None

    for ls, lt, th, area, mk, sp in product(
        grid["lambda_spec"],
        grid["lambda_tex"],
        grid["threshold"],
        grid["min_area_m2"],
        grid["morph_kernel"],
        grid["shadow_penalty"],
    ):
        p = FusionParams(
            lambda_spec=float(ls),
            lambda_tex=float(lt),
            threshold=float(th),
            min_area_m2=float(area),
            morph_kernel=int(mk),
            shadow_penalty=float(sp),
        )
        fused = fuse_probabilities(prob_dl, prob_spec, prob_tex, p)
        score = _objective(fused, prob_dl, prob_spec, prob_tex, p.threshold, shadow_score, p.shadow_penalty)
        if score > best_score:
            best_score = score
            best_p = p
            best_fused = fused

    assert best_p is not None and best_fused is not None
    return best_fused, best_p, float(best_score)
