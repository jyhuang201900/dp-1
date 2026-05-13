from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

FLOAT_NODATA = -9999.0
UINT8_NODATA = 255


def nodata_for_dtype(dtype: str) -> float | int:
    return UINT8_NODATA if str(dtype).lower() == "uint8" else FLOAT_NODATA


def normalized_valid_mask(arr: np.ndarray, nodata: float | int | None = None) -> np.ndarray:
    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    valid &= arr >= 0.0
    valid &= arr <= 1.0
    return valid


@dataclass
class RasterData:
    arr: np.ndarray
    profile: dict


@dataclass
class CoreHaloWindow:
    core: Window
    halo: Window
    core_row_offset_in_halo: int
    core_col_offset_in_halo: int


def read_band1(path: str) -> RasterData:
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
    return RasterData(arr=arr, profile=profile)


def read_downsampled_band1(path: str, max_dim: int = 2048) -> RasterData:
    with rasterio.open(path) as ds:
        scale = max(ds.height / max_dim, ds.width / max_dim, 1.0)
        out_h = max(1, round(ds.height / scale))
        out_w = max(1, round(ds.width / scale))
        arr = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.average, masked=True)
        fill_value = float(ds.nodata) if ds.nodata is not None else 0.0
        profile = ds.profile.copy()
    return RasterData(arr=np.ma.filled(arr, fill_value).astype(np.float32), profile=profile)


def write_band1(path: str, arr: np.ndarray, profile: dict, dtype: str = "float32") -> None:
    out = profile.copy()
    out.update(count=1, dtype=dtype, compress="lzw", tiled=True, BIGTIFF="YES", nodata=nodata_for_dtype(dtype))
    with rasterio.open(path, "w", **out) as ds:
        ds.write(arr.astype(dtype), 1)


def create_empty_raster(path: str, profile: dict, dtype: str = "float32", nodata: float | int | None = None) -> None:
    out = profile.copy()
    out.update(count=1, dtype=dtype, compress="lzw", tiled=True, BIGTIFF="YES")
    out.update(nodata=nodata_for_dtype(dtype) if nodata is None else nodata)
    with rasterio.open(path, "w", **out):
        pass


def iter_windows(height: int, width: int, tile_size: int, stride: int) -> Iterator[Window]:
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    y_last = max(height - tile_size, 0)
    x_last = max(width - tile_size, 0)
    if not ys or ys[-1] != y_last:
        ys.append(y_last)
    if not xs or xs[-1] != x_last:
        xs.append(x_last)

    for y in ys:
        for x in xs:
            h = min(tile_size, height - y)
            w = min(tile_size, width - x)
            yield Window(col_off=x, row_off=y, width=w, height=h)


def iter_core_halo_windows(height: int, width: int, core_size: int, halo: int) -> Iterator[CoreHaloWindow]:
    for core in iter_windows(height, width, core_size, core_size):
        x = int(core.col_off)
        y = int(core.row_off)
        w = int(core.width)
        h = int(core.height)

        halo_x = x - halo
        halo_y = y - halo
        halo_w = w + 2 * halo
        halo_h = h + 2 * halo

        yield CoreHaloWindow(
            core=core,
            halo=Window(col_off=halo_x, row_off=halo_y, width=halo_w, height=halo_h),
            core_row_offset_in_halo=halo,
            core_col_offset_in_halo=halo,
        )


def iter_inference_windows(
    height: int, width: int, tile_size: int, stride: int
) -> Iterator[tuple[Window, Window, tuple[int, int]]]:
    overlap = max(tile_size - stride, 0)
    trim_lo = overlap // 2
    trim_hi = overlap - trim_lo

    for tile in iter_windows(height, width, tile_size, stride):
        x = int(tile.col_off)
        y = int(tile.row_off)
        w = int(tile.width)
        h = int(tile.height)

        left = 0 if x == 0 else min(trim_lo, w)
        top = 0 if y == 0 else min(trim_lo, h)
        right = 0 if x + w >= width else min(trim_hi, w)
        bottom = 0 if y + h >= height else min(trim_hi, h)

        core_x = x + left
        core_y = y + top
        core_w = max(w - left - right, 0)
        core_h = max(h - top - bottom, 0)
        if core_w == 0 or core_h == 0:
            continue

        core = Window(col_off=core_x, row_off=core_y, width=core_w, height=core_h)
        yield tile, core, (top, left)


def gaussian_weight(h: int, w: int, sigma_ratio: float = 0.125) -> np.ndarray:
    yy = np.linspace(-1, 1, h, dtype=np.float32)
    xx = np.linspace(-1, 1, w, dtype=np.float32)
    gy = np.exp(-(yy * yy) / (2 * sigma_ratio * sigma_ratio))
    gx = np.exp(-(xx * xx) / (2 * sigma_ratio * sigma_ratio))
    weight = np.outer(gy, gx)
    weight = weight / (np.max(weight) + 1e-6)
    return weight.astype(np.float32)


def normalize_clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    scale = np.float32(1.0 / (hi - lo))
    return np.clip((arr - lo) * scale, 0.0, 1.0).astype(np.float32)


def compute_percentiles_uint8(path: str, pmin: float = 2.0, pmax: float = 98.0) -> tuple[float, float]:
    hist: np.ndarray = np.zeros(256, dtype=np.int64)
    with rasterio.open(path) as ds:
        for _, window in ds.block_windows(1):
            arr = ds.read(1, window=window, masked=True)
            vals = arr.compressed()
            if vals.size == 0:
                continue
            hist += np.bincount(vals.astype(np.uint8), minlength=256)

    cdf = np.cumsum(hist)
    total = int(cdf[-1])
    if total == 0:
        return 0.0, 255.0
    lo_idx = int(np.searchsorted(cdf, total * (pmin / 100.0)))
    hi_idx = int(np.searchsorted(cdf, total * (pmax / 100.0)))
    return float(lo_idx), float(max(hi_idx, lo_idx + 1))


def normalize_raster_to_file(input_path: str, output_path: str, pmin: float = 2.0, pmax: float = 98.0) -> dict:
    lo, hi = compute_percentiles_uint8(input_path, pmin=pmin, pmax=pmax)
    valid_pixels = 0
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, compress="lzw", tiled=True, BIGTIFF="YES", nodata=FLOAT_NODATA)
        with rasterio.open(output_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window, masked=True).astype(np.float32)
                valid = ~np.ma.getmaskarray(arr)
                out = np.full(arr.shape, FLOAT_NODATA, dtype=np.float32)
                if np.any(valid):
                    filled = np.ma.filled(arr, lo).astype(np.float32)
                    norm = normalize_clip(filled, lo, hi)
                    out[valid] = norm[valid]
                    valid_pixels += int(np.count_nonzero(valid))
                dst.write(out, 1, window=window)
    return {"lo": lo, "hi": hi, "valid_pixels": valid_pixels}


def process_single_input_to_outputs(
    input_path: str,
    output_specs: Sequence[tuple[str, str]],
    core_size: int,
    halo: int,
    processor: Callable[[np.ndarray], Sequence[np.ndarray]],
) -> None:
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        src_nodata = float(src.nodata) if src.nodata is not None else FLOAT_NODATA
        outputs = []
        for out_path, dtype in output_specs:
            out_profile = profile.copy()
            out_profile.update(
                count=1,
                dtype=dtype,
                compress="lzw",
                tiled=True,
                BIGTIFF="YES",
                nodata=nodata_for_dtype(dtype),
            )
            outputs.append(rasterio.open(out_path, "w", **out_profile))

        try:
            for item in iter_core_halo_windows(src.height, src.width, core_size=core_size, halo=halo):
                arr = src.read(1, window=item.halo, boundless=True, fill_value=src_nodata, masked=True).astype(
                    np.float32
                )
                valid = ~np.ma.getmaskarray(arr)
                arr_filled = np.ma.filled(arr, src_nodata).astype(np.float32)
                result_arrays = processor(arr_filled)
                for out_ds, res in zip(outputs, result_arrays, strict=True):
                    nodata = out_ds.nodata if out_ds.nodata is not None else nodata_for_dtype(out_ds.dtypes[0])
                    res_out = np.array(res, copy=True)
                    res_out[~valid] = nodata
                    y0 = item.core_row_offset_in_halo
                    x0 = item.core_col_offset_in_halo
                    y1 = y0 + int(item.core.height)
                    x1 = x0 + int(item.core.width)
                    out_ds.write(res_out[y0:y1, x0:x1].astype(out_ds.dtypes[0]), 1, window=item.core)
        finally:
            for ds in outputs:
                ds.close()


def process_aligned_inputs_to_output(
    input_paths: Sequence[str],
    output_path: str,
    dtype: str,
    core_size: int,
    halo: int,
    processor: Callable[..., np.ndarray],
) -> None:
    sources = [rasterio.open(path) for path in input_paths]
    try:
        ref = sources[0]
        profile = ref.profile.copy()
        profile.update(count=1, dtype=dtype, compress="lzw", tiled=True, BIGTIFF="YES", nodata=nodata_for_dtype(dtype))
        with rasterio.open(output_path, "w", **profile) as dst:
            for item in iter_core_halo_windows(ref.height, ref.width, core_size=core_size, halo=halo):
                arrays = []
                valid = None
                for src in sources:
                    src_nodata = float(src.nodata) if src.nodata is not None else FLOAT_NODATA
                    arr = src.read(1, window=item.halo, boundless=True, fill_value=src_nodata, masked=True).astype(
                        np.float32
                    )
                    arr_valid = ~np.ma.getmaskarray(arr)
                    valid = arr_valid if valid is None else (valid & arr_valid)
                    arrays.append(np.ma.filled(arr, src_nodata).astype(np.float32))
                res = np.array(processor(*arrays), copy=True)
                nodata = dst.nodata if dst.nodata is not None else nodata_for_dtype(dtype)
                if valid is not None:
                    res[~valid] = nodata
                y0 = item.core_row_offset_in_halo
                x0 = item.core_col_offset_in_halo
                y1 = y0 + int(item.core.height)
                x1 = x0 + int(item.core.width)
                dst.write(res[y0:y1, x0:x1].astype(dtype), 1, window=item.core)
    finally:
        for src in sources:
            src.close()


def pixel_area_m2(profile: dict) -> float:
    transform = profile.get("transform")
    if transform is None:
        return 1.0
    return float(abs(transform.a * transform.e))
