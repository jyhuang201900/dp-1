"""Round-level artifact bookkeeping for the closed-loop RL fusion driver.

This module owns the *pure* pieces of the artifact layout: deriving
the per-round directory and exception/classification helpers used by
the closed-loop summary.

The on-disk copy / snapshot / atomic-restore helpers themselves
intentionally live in :mod:`forestseg.cli` so the test suite's
``monkeypatch.setattr("forestseg.cli._require_json_copy", ...)`` style
hooks continue to reach the actual call sites (the closed-loop driver
calls those helpers via :mod:`forestseg.cli`'s module globals).
"""

from __future__ import annotations

import os

__all__ = [
    "RestoreOutputsError",
    "_optional_restore_outcome",
    "_round_artifact_paths",
]


def _round_artifact_paths(work: str, round_index: int) -> dict[str, str]:
    """Return the per-round artifact paths under ``work/rounds/round_NN``."""
    round_dir = os.path.join(work, "rounds", f"round_{round_index:02d}")
    os.makedirs(round_dir, exist_ok=True)
    fusion_selected_path = os.path.join(round_dir, "fusion_selected.json")
    return {
        "dir": round_dir,
        "feature_meta": os.path.join(round_dir, "feature_stack_meta.json"),
        "train_metrics": os.path.join(round_dir, "train_metrics.json"),
        "metrics_val": os.path.join(round_dir, "metrics_val.json"),
        "rl_payload": fusion_selected_path,
        "fusion_selected": fusion_selected_path,
        "feature_feedback": os.path.join(round_dir, "feature_feedback.json"),
        "prob_dl": os.path.join(round_dir, "prob_dl.tif"),
        "unc_dl": os.path.join(round_dir, "unc_dl.tif"),
        "prob_fused": os.path.join(round_dir, "prob_fused.tif"),
    }


class RestoreOutputsError(Exception):
    """Raised by ``_restore_outputs_atomically`` when a restore fails.

    Carries the list of labels that *were* successfully restored before
    the failure, plus the label of the artifact that failed.
    """

    def __init__(self, message: str, *, restored_labels: list[str], failed_label: str | None):
        super().__init__(message)
        self.restored_labels = list(restored_labels)
        self.failed_label = failed_label


def _optional_restore_outcome(
    optional_labels: list[str], restored_labels: list[str], failed_labels: list[str]
) -> dict[str, list[str]]:
    """Classify optional artifacts into restored / failed / skipped buckets."""
    restored_optional_artifacts = [label for label in optional_labels if label in restored_labels]
    failed_optional_artifacts = [label for label in optional_labels if label in failed_labels]
    skipped_optional_artifacts = [
        label for label in optional_labels if label not in restored_labels and label not in failed_labels
    ]
    return {
        "restored_optional_artifacts": restored_optional_artifacts,
        "failed_optional_artifacts": failed_optional_artifacts,
        "skipped_optional_artifacts": skipped_optional_artifacts,
    }
