from __future__ import annotations

import json
import os

import fiona
import rasterio
from rasterio.features import shapes
from rasterio.windows import transform as window_transform
from shapely.geometry import mapping, shape


def copy_raster(src_path: str, dst_path: str) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(compress="lzw", tiled=True, BIGTIFF="YES")
        with rasterio.open(dst_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                arr = src.read(1, window=window)
                dst.write(arr, 1, window=window)
    return dst_path


def export_generated_rasters(
    output_dir: str,
    cropped_raster: str,
    prob_spec_path: str,
    prob_tex_path: str,
    prob_dl_path: str,
    prob_fused_path: str,
    mask_final_path: str,
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    p_spec = os.path.join(output_dir, f"{cropped_raster}_forest_prob_spec.tif")
    p_tex = os.path.join(output_dir, f"{cropped_raster}_forest_prob_tex.tif")
    p_dl = os.path.join(output_dir, f"{cropped_raster}_forest_prob_dl.tif")
    p_fused = os.path.join(output_dir, f"{cropped_raster}_forest_prob_fused.tif")
    p_mask = os.path.join(output_dir, f"{cropped_raster}_forest_mask_final.tif")

    copy_raster(prob_spec_path, p_spec)
    copy_raster(prob_tex_path, p_tex)
    copy_raster(prob_dl_path, p_dl)
    copy_raster(prob_fused_path, p_fused)
    copy_raster(mask_final_path, p_mask)

    return {
        "prob_spec": p_spec,
        "prob_tex": p_tex,
        "prob_dl": p_dl,
        "prob_fused": p_fused,
        "mask_final": p_mask,
    }


def export_vector_streaming(mask_tif: str, gpkg_path: str) -> str:
    os.makedirs(os.path.dirname(gpkg_path), exist_ok=True)
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)

    with rasterio.open(mask_tif) as ds:
        crs_wkt = ds.crs.to_wkt() if ds.crs else None
        schema = {"geometry": "Polygon", "properties": {"class": "str"}}

        with fiona.open(
            gpkg_path,
            mode="w",
            driver="GPKG",
            crs_wkt=crs_wkt,
            schema=schema,
            encoding="utf-8",
            layer="forest",
        ) as sink:
            for _, window in ds.block_windows(1):
                arr = ds.read(1, window=window)
                mask = arr == 1
                if not mask.any():
                    continue
                tx = window_transform(window, ds.transform)
                for geom, val in shapes(arr.astype("uint8"), mask=mask, transform=tx):
                    if int(val) != 1:
                        continue
                    sink.write({"geometry": mapping(shape(geom)), "properties": {"class": "forest"}})

    return gpkg_path


def export_selected_params(output_dir: str, cropped_raster: str, params: dict, score: float | None = None) -> str:
    path = os.path.join(output_dir, f"{cropped_raster}_fusion_params.json")
    payload = {"params": params}
    if score is not None:
        payload["score"] = float(score)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
