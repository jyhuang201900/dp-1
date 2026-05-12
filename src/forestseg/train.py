from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .coupling_tensor import warmup_scale
from .metrics import binary_metrics
from .model import DiceLoss, build_model, to_device
from .sampler import evaluate_points, make_training_batch


def _resolve_selection_metric(cfg: dict[str, Any], metrics: dict[str, Any]) -> tuple[str, float]:
    selection_metric = str(cfg.get("selection_metric", "f1")).strip().lower()
    if selection_metric not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise ValueError(f"未知 dl.selection_metric：{selection_metric}。可用指标：{available}。")
    return selection_metric, float(metrics[selection_metric])


def _validate_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必须为正整数，当前为 {value}。")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{label} 必须为正整数，当前为 {value}。")
        parsed = int(value)
    else:
        raw = str(value).strip()
        if not raw:
            raise ValueError(f"{label} 必须为正整数，当前为空。")
        if raw.startswith(("+", "-")):
            digits = raw[1:]
        else:
            digits = raw
        if not digits.isdigit():
            raise ValueError(f"{label} 必须为正整数，当前为 {value}。")
        parsed = int(raw)
    if parsed <= 0:
        raise ValueError(f"{label} 必须大于 0，当前为 {parsed}。")
    return parsed


@dataclass
class TrainResult:
    checkpoint_path: str
    best_score: float
    metrics: dict[str, Any]


def train_supervised_model(
    feature_stack_path: str,
    train_points: list[dict[str, Any]],
    val_points: list[dict[str, Any]],
    cfg: dict,
    checkpoint_path: str,
) -> TrainResult:
    sample_tile_size = _validate_positive_int(
        cfg.get("sample_tile_size", cfg.get("tile_size", 512)), "sample_tile_size"
    )
    batch_size = _validate_positive_int(cfg.get("batch_size", 8), "batch_size")
    epochs = _validate_positive_int(cfg.get("epochs", 8), "epochs")
    max_patches = _validate_positive_int(cfg.get("max_patches", 2500), "max_patches")
    lr = float(cfg.get("learning_rate", 3e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    aux_warmup = (
        _validate_positive_int(cfg.get("aux_warmup_epochs", 0), "aux_warmup_epochs")
        if cfg.get("aux_warmup_epochs", 0) != 0
        else 0
    )

    with rasterio.open(feature_stack_path) as ds:
        in_channels = int(ds.count)
        model = build_model(
            encoder_name=cfg.get("encoder_name", "resnet34"),
            in_channels=in_channels,
            classes=1,
        )
        model, device = to_device(model)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        dice = DiceLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        steps_per_epoch = max(1, max_patches // max(batch_size, 1))
        best_score = float("-inf")
        best_metrics: dict[str, Any] = {}
        history: list[dict[str, Any]] = []

        for ep in range(epochs):
            model.train()
            ep_losses = []
            warm = warmup_scale(ep, aux_warmup)
            for _ in tqdm(range(steps_per_epoch), desc=f"train epoch {ep + 1}/{epochs}"):
                x_np, y_np, m_np = make_training_batch(
                    ds=ds,
                    points=train_points,
                    batch_size=batch_size,
                    tile_size=sample_tile_size,
                    rng=rng,
                )
                x = torch.from_numpy(x_np).to(device)
                y = torch.from_numpy(y_np).to(device)
                m = torch.from_numpy(m_np).to(device)
                logits = model(x)
                bce_raw = bce(logits, y)
                bce_loss = torch.sum(bce_raw * m) / (torch.sum(m) + 1e-6)
                dice_loss = dice(logits, y, mask=m)
                loss = bce_loss + (0.5 + 0.5 * warm) * dice_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ep_losses.append(float(loss.item()))

            model.eval()
            val_true, val_prob = evaluate_points(
                ds=ds,
                model=model,
                device=device,
                points=val_points,
                tile_size=sample_tile_size,
                batch_size=batch_size,
            )
            metrics = binary_metrics(val_true, val_prob, threshold=float(cfg.get("val_threshold", 0.5)))
            metrics["epoch"] = ep + 1
            metrics["train_loss"] = float(np.mean(ep_losses)) if ep_losses else float("inf")
            selection_metric, score_value = _resolve_selection_metric(cfg, metrics)
            metrics["selection_metric"] = selection_metric
            history_entry = dict(metrics)
            history.append(history_entry)
            if score_value > best_score:
                best_score = score_value
                best_metrics = dict(metrics)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "best_score": best_score,
                        "config": cfg,
                        "in_channels": in_channels,
                        "metrics": dict(best_metrics),
                    },
                    checkpoint_path,
                )

    result_metrics = dict(best_metrics)
    result_metrics["history"] = [dict(entry) for entry in history]
    result_metrics["selection_metric"] = result_metrics.get("selection_metric", selection_metric)
    result_metrics["best_score"] = best_score
    return TrainResult(checkpoint_path=checkpoint_path, best_score=best_score, metrics=result_metrics)
