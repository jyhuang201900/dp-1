from __future__ import annotations

import os

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from .io_raster import FLOAT_NODATA, create_empty_raster, iter_inference_windows, normalized_valid_mask
from .model import build_model, enable_mc_dropout, to_device


def load_checkpoint_model(checkpoint_path: str, encoder_name: str = "resnet34", in_channels: int = 1):
    if not os.path.exists(checkpoint_path):
        return None, None

    model = build_model(encoder_name=encoder_name, in_channels=in_channels, classes=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model, device = to_device(model)
    model.eval()
    return model, device


def infer_to_files(
    input_path: str,
    checkpoint_path: str,
    encoder_name: str,
    tile_size: int,
    stride: int,
    batch_size: int,
    prob_out_path: str,
    unc_out_path: str,
    mc_dropout_passes: int = 4,
) -> None:
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        src_nodata = float(src.nodata) if src.nodata is not None else FLOAT_NODATA
        create_empty_raster(prob_out_path, profile, dtype="float32", nodata=FLOAT_NODATA)
        create_empty_raster(unc_out_path, profile, dtype="float32", nodata=FLOAT_NODATA)
        model, device = load_checkpoint_model(checkpoint_path, encoder_name=encoder_name, in_channels=src.count)

        if model is None:
            with rasterio.open(prob_out_path, "r+") as prob_ds, rasterio.open(unc_out_path, "r+") as unc_ds:
                for _, window in src.block_windows(1):
                    arr = src.read(1, window=window, masked=True).astype(np.float32)
                    valid = ~np.ma.getmaskarray(arr)
                    filled = np.ma.filled(arr, src_nodata).astype(np.float32)
                    base = np.full(filled.shape, FLOAT_NODATA, dtype=np.float32)
                    unc = np.full(filled.shape, FLOAT_NODATA, dtype=np.float32)
                    if np.any(valid):
                        base[valid] = np.clip(1.0 - filled[valid], 0.0, 1.0).astype(np.float32)
                        unc[valid] = 0.25
                    prob_ds.write(base, 1, window=window)
                    unc_ds.write(unc, 1, window=window)
            return

        windows = list(iter_inference_windows(src.height, src.width, tile_size=tile_size, stride=stride))
        with rasterio.open(prob_out_path, "r+") as prob_ds, rasterio.open(unc_out_path, "r+") as unc_ds:
            for i in tqdm(range(0, len(windows), batch_size), desc="infer"):
                batch = windows[i : i + batch_size]
                tensors = []
                metas = []
                for tile_win, core_win, (top, left) in batch:
                    arr = src.read(window=tile_win, boundless=True, fill_value=src_nodata, masked=True).astype(
                        np.float32
                    )
                    bands, hh, ww = arr.shape
                    filled = np.ma.filled(arr, src_nodata).astype(np.float32)
                    valid = ~np.ma.getmaskarray(arr[0])
                    tile = np.zeros((bands, tile_size, tile_size), dtype=np.float32)
                    tile_valid: np.ndarray = np.zeros((tile_size, tile_size), dtype=bool)
                    tile[:, :hh, :ww] = np.where(np.ma.getmaskarray(arr), 0.0, filled)
                    tile_valid[:hh, :ww] = valid
                    tensors.append(tile[None, ...])
                    metas.append((core_win, top, left, tile_valid))
                x_t = torch.from_numpy(np.concatenate(tensors, axis=0)).to(device)
                pass_probs = []
                with torch.no_grad():
                    model.eval()
                    for pidx in range(max(int(mc_dropout_passes), 1)):
                        if pidx > 0:
                            enable_mc_dropout(model)
                        logits = model(x_t)
                        prob = torch.sigmoid(logits).detach().cpu().numpy()[:, 0, :, :]
                        pass_probs.append(prob)
                probs = np.stack(pass_probs, axis=0)
                mean_prob = np.mean(probs, axis=0)
                var_prob = np.var(probs, axis=0)
                for (core_win, top, left, tile_valid), mprob, vprob in zip(metas, mean_prob, var_prob, strict=True):
                    core_h = int(core_win.height)
                    core_w = int(core_win.width)
                    valid_core = tile_valid[top : top + core_h, left : left + core_w] & normalized_valid_mask(
                        mprob[top : top + core_h, left : left + core_w]
                    )
                    core_slice: np.ndarray = np.full((core_h, core_w), FLOAT_NODATA, dtype=np.float32)
                    unc_slice: np.ndarray = np.full((core_h, core_w), FLOAT_NODATA, dtype=np.float32)
                    if np.any(valid_core):
                        core_prob = mprob[top : top + core_h, left : left + core_w]
                        core_unc = vprob[top : top + core_h, left : left + core_w]
                        core_slice[valid_core] = core_prob[valid_core].astype(np.float32)
                        unc_slice[valid_core] = core_unc[valid_core].astype(np.float32)
                    prob_ds.write(core_slice, 1, window=core_win)
                    unc_ds.write(unc_slice, 1, window=core_win)
