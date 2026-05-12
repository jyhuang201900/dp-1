from __future__ import annotations

from typing import Any

import numpy as np

from ..core.io import atomic_write_json


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = y_true.astype(np.uint8)
    y_pred = (y_prob >= threshold).astype(np.uint8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "count": len(y_true),
    }


def save_metrics(path: str, metrics: dict[str, Any]) -> str:
    """Atomically write ``metrics`` to ``path`` as JSON; return ``path``.

    The metrics artefact (``metrics_val.json``) is consumed by both the RL
    fusion driver and the per-round snapshot, so a torn write would block
    later rounds — :func:`forestseg._io.atomic_write_json` guarantees
    readers see either the previous contents or the full new payload.
    """
    return atomic_write_json(path, metrics)
