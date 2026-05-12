from __future__ import annotations

from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from .labels import raster_xy_to_rowcol


def _circle_mask(tile_size: int, radius_px: int) -> np.ndarray:
    cy = tile_size // 2
    cx = tile_size // 2
    yy, xx = np.ogrid[:tile_size, :tile_size]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius_px**2).astype(np.float32)


def balanced_sample_points(
    points: list[dict[str, Any]], batch_size: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    pos = [p for p in points if int(p["label"]) == 1]
    neg = [p for p in points if int(p["label"]) == 0]
    if not pos or not neg:
        idx = rng.choice(len(points), size=batch_size, replace=len(points) < batch_size)
        return [points[int(i)] for i in idx]
    npos = batch_size // 2
    nneg = batch_size - npos
    pos_idx = rng.choice(len(pos), size=npos, replace=len(pos) < npos)
    neg_idx = rng.choice(len(neg), size=nneg, replace=len(neg) < nneg)
    batch = [pos[int(i)] for i in pos_idx] + [neg[int(i)] for i in neg_idx]
    rng.shuffle(batch)
    return batch


def read_feature_patch(ds: rasterio.DatasetReader, x: float, y: float, tile_size: int) -> tuple[np.ndarray, np.ndarray]:
    row, col = raster_xy_to_rowcol(ds, x, y)
    half = tile_size // 2
    win = Window(col_off=col - half, row_off=row - half, width=tile_size, height=tile_size)
    arr = ds.read(window=win, boundless=True, fill_value=0).astype(np.float32)
    if arr.shape[1] != tile_size or arr.shape[2] != tile_size:
        out = np.zeros((arr.shape[0], tile_size, tile_size), dtype=np.float32)
        out[:, : arr.shape[1], : arr.shape[2]] = arr
        arr = out
    center_mask = _circle_mask(tile_size, max(1, tile_size // 16))
    return arr, center_mask


def make_training_batch(
    ds: rasterio.DatasetReader,
    points: list[dict[str, Any]],
    batch_size: int,
    tile_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = balanced_sample_points(points, batch_size=batch_size, rng=rng)
    x = np.zeros((batch_size, ds.count, tile_size, tile_size), dtype=np.float32)
    y = np.zeros((batch_size, 1, tile_size, tile_size), dtype=np.float32)
    m = np.zeros((batch_size, 1, tile_size, tile_size), dtype=np.float32)
    for i, pt in enumerate(batch):
        feat, mask = read_feature_patch(ds, float(pt["x"]), float(pt["y"]), tile_size=tile_size)
        x[i] = feat
        y[i, 0] = float(pt["label"]) * mask
        m[i, 0] = mask
    return x, y, m


def evaluate_points(
    ds: rasterio.DatasetReader,
    model: Any,
    device: Any,
    points: list[dict[str, Any]],
    tile_size: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    y_true = np.asarray([int(p["label"]) for p in points], dtype=np.uint8)
    y_prob = np.zeros(len(points), dtype=np.float32)
    for start in range(0, len(points), batch_size):
        chunk = points[start : start + batch_size]
        feats = []
        for pt in chunk:
            feat, _ = read_feature_patch(ds, float(pt["x"]), float(pt["y"]), tile_size=tile_size)
            feats.append(feat[None, ...])
        x = torch.from_numpy(np.concatenate(feats, axis=0)).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).detach().cpu().numpy()[:, 0, :, :]
        center = tile_size // 2
        radius = max(1, tile_size // 16)
        yy, xx = np.ogrid[:tile_size, :tile_size]
        mask = (yy - center) ** 2 + (xx - center) ** 2 <= radius**2
        y_prob[start : start + len(chunk)] = prob[:, mask].mean(axis=1).astype(np.float32)
    return y_true, y_prob
