from __future__ import annotations

import os
from typing import Any

import numpy as np
import rasterio

from ..core.io import atomic_write_json
from ..io.raster import FLOAT_NODATA, normalize_clip


def _normalize_optional_raster(input_path: str, output_path: str, pmin: float = 2.0, pmax: float = 98.0) -> str:
    with rasterio.open(input_path) as src:
        arr = src.read(1, masked=True).astype(np.float32)
        vals = arr.compressed()
        if vals.size == 0:
            raise ValueError(f"No valid pixels in optional raster: {input_path}")
        lo, hi = np.percentile(vals, [pmin, pmax])
        out = np.full(arr.shape, FLOAT_NODATA, dtype=np.float32)
        valid = ~np.ma.getmaskarray(arr)
        filled = np.ma.filled(arr, lo).astype(np.float32)
        out[valid] = normalize_clip(filled, float(lo), float(hi))[valid]
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, compress="lzw", tiled=True, BIGTIFF="YES", nodata=FLOAT_NODATA)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out, 1)
    return output_path


def _apply_transform(arr: np.ndarray, transform: dict[str, Any] | None) -> np.ndarray:
    if not transform:
        return arr
    out: np.ndarray = arr.astype(np.float32, copy=True)
    valid = np.isfinite(out) & (out != FLOAT_NODATA)
    if not np.any(valid):
        return out
    scale = float(transform.get("scale", 1.0))
    offset = float(transform.get("offset", 0.0))
    gamma = float(transform.get("gamma", 1.0))
    threshold = transform.get("threshold")
    below_scale = float(transform.get("below_scale", 1.0))
    above_scale = float(transform.get("above_scale", 1.0))
    base = np.clip(out[valid] * scale + offset, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        base = np.power(base, gamma, dtype=np.float32)
    out[valid] = np.clip(base, 0.0, 1.0)
    if threshold is not None:
        th = float(threshold)
        low_mask = valid & (out < th)
        high_mask = valid & (out >= th)
        out[low_mask] = np.clip(out[low_mask] * below_scale, 0.0, 1.0)
        out[high_mask] = np.clip(out[high_mask] * above_scale, 0.0, 1.0)
    return out


def _assert_aligned_raster(ref: rasterio.DatasetReader, src: rasterio.DatasetReader, path: str) -> None:
    if ref.width != src.width or ref.height != src.height:
        raise ValueError(f"特征栅格尺寸不一致：{path}")
    if ref.crs != src.crs:
        raise ValueError(f"特征栅格 CRS 不一致：{path}")
    if ref.transform != src.transform:
        raise ValueError(f"特征栅格仿射变换不一致：{path}")


def build_feature_stack(
    feature_paths: list[str],
    output_path: str,
    feature_names: list[str],
    dem_path: str | None = None,
    dem_output_path: str | None = None,
    feature_transforms: list[dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    paths = list(feature_paths)
    names = list(feature_names)
    transforms = list(feature_transforms or [None] * len(paths))
    if not paths:
        raise ValueError("feature_paths cannot be empty")
    if len(names) != len(paths):
        raise ValueError("feature_names must match feature_paths length")
    if len(transforms) != len(paths):
        raise ValueError("feature_transforms must match feature_paths length")
    if dem_path:
        if not dem_output_path:
            raise ValueError("dem_output_path is required when dem_path is provided")
        with rasterio.open(paths[0]) as ref_src, rasterio.open(dem_path) as dem_src:
            _assert_aligned_raster(ref_src, dem_src, dem_path)
        _normalize_optional_raster(dem_path, dem_output_path)
        paths.append(dem_output_path)
        names.append("dem")
        transforms.append(None)

    sources = [rasterio.open(p) for p in paths]
    try:
        ref = sources[0]
        for path, src in zip(paths[1:], sources[1:], strict=True):
            _assert_aligned_raster(ref, src, path)
        profile = ref.profile.copy()
        profile.update(
            count=len(sources), dtype="float32", compress="lzw", tiled=True, BIGTIFF="YES", nodata=FLOAT_NODATA
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.descriptions = tuple(names)
            for _, window in ref.block_windows(1):
                for idx, (src, transform) in enumerate(zip(sources, transforms, strict=True), start=1):
                    arr = src.read(1, window=window, masked=True).astype(np.float32)
                    out = np.ma.filled(arr, FLOAT_NODATA).astype(np.float32)
                    out = _apply_transform(out, transform)
                    dst.write(out, idx, window=window)
    finally:
        for src in sources:
            src.close()

    meta = {
        "feature_names": names,
        "feature_paths": paths,
        "feature_transforms": transforms,
        "output_path": output_path,
    }
    meta_path = os.path.splitext(output_path)[0] + "_meta.json"
    atomic_write_json(meta_path, meta)
    return meta
