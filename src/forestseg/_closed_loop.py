"""Pure helpers backing :func:`forestseg.cli.cmd_run_closed_loop`.

The closed-loop driver itself stays in :mod:`forestseg.cli` because it
drives several other ``cmd_*`` handlers and uses the on-disk
snapshot / restore helpers that the test suite monkeypatches there.
What lives in this module is the side-effect-free skeleton around
that driver:

- :func:`reward_and_score` — computes ``(reward, score)`` from a
  ``validation_metrics`` dict for a given selection metric.
- :func:`build_artifact_paths_block` — builds the constant
  ``artifacts`` view referenced by both success and failure summaries.
- :func:`build_summary` — assembles the canonical schema-v2 summary
  dict for both ``status="ok"`` and ``status="failed"`` runs.

Closed-loop config validation is intentionally **not** consolidated
here: the driver validates ``rl_loop.rounds`` / ``patience`` /
``min_delta`` / ``selection_metric`` one at a time so that a failure
on a later field still surfaces the already-validated earlier fields
in the failure summary (which the test suite checks).
"""

from __future__ import annotations

import os
from typing import Any

from ._rl_history import _history_entry

__all__ = [
    "build_artifact_paths_block",
    "build_best_round_restore_plan",
    "build_round_artifacts_map",
    "build_round_history_entries",
    "build_summary",
    "reward_and_score",
]


def build_round_artifacts_map(round_paths: dict[str, str]) -> dict[str, str]:
    """Return the per-round ``artifacts`` mapping embedded in history entries.

    Pulled from ``round_paths`` (the dict returned by
    :func:`forestseg._artifacts._round_artifact_paths`); kept here so the
    history-entry layout has a single source of truth.
    """
    return {
        "feature_meta": round_paths["feature_meta"],
        "train_metrics": round_paths["train_metrics"],
        "metrics_val": round_paths["metrics_val"],
        "rl_payload": round_paths["rl_payload"],
        "fusion_selected": round_paths["fusion_selected"],
        "feature_feedback": round_paths["feature_feedback"],
        "prob_dl": round_paths["prob_dl"],
        "unc_dl": round_paths["unc_dl"],
        "prob_fused": round_paths["prob_fused"],
    }


def build_best_round_restore_plan(
    work: str,
    best_artifacts: dict[str, Any],
    *,
    wf: Any,
    path_exists: Any = os.path.exists,
) -> tuple[list[tuple[str, str, str, str]], list[str]]:
    """Plan the atomic restore of the best-round artifacts.

    Returns ``(restore_entries, optional_restore_labels)``.

    ``restore_entries`` is the ``(src, dst, label, kind)`` quadruple
    list consumed by :func:`forestseg.cli._restore_outputs_atomically`.
    ``optional_restore_labels`` carries the labels that may legitimately
    be missing (currently just ``"unc_dl"``).

    ``path_exists`` is parameterised so callers can swap it for a test
    fake; it defaults to :func:`os.path.exists`.
    """
    best_fusion_selected = best_artifacts.get("fusion_selected") or best_artifacts.get("rl_payload", "")
    restore_entries: list[tuple[str, str, str, str]] = [
        (
            best_fusion_selected,
            os.path.join(work, "fusion_selected.json"),
            "best round fusion_selected",
            "json",
        ),
        (
            best_artifacts.get("feature_feedback", ""),
            wf(work, "feature_feedback"),
            "best round feature_feedback",
            "json",
        ),
        (best_artifacts.get("prob_dl", ""), wf(work, "prob_dl"), "best round prob_dl", "file"),
        (best_artifacts.get("prob_fused", ""), wf(work, "prob_fused"), "best round prob_fused", "file"),
        (best_artifacts.get("metrics_val", ""), wf(work, "metrics_val"), "best round metrics_val", "json"),
    ]
    best_unc_dl = best_artifacts.get("unc_dl")
    if best_unc_dl and path_exists(best_unc_dl):
        restore_entries.append((best_unc_dl, wf(work, "unc_dl"), "unc_dl", "file"))
        optional_restore_labels = ["unc_dl"]
    else:
        optional_restore_labels = ["unc_dl"]
    return restore_entries, optional_restore_labels


def build_round_history_entries(
    *,
    round_no: int,
    selection_metric: str,
    score: float,
    reward: float,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    artifacts: dict[str, str],
    rl_out: dict[str, Any],
    train_out: dict[str, Any],
    feedback_path: str,
    feature_meta: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the pair of history entries persisted for a single round.

    Returns ``(loop_entry, metrics_entry)``. They share the base
    ``extra`` block (stage, params, checkpoint, feedback). The
    ``loop_entry`` additionally carries nested ``train`` and ``rl``
    sub-blocks that downstream consumers rely on for richer drill-down;
    the ``metrics_entry`` (written into ``metrics_round_history.json``)
    omits them to keep that artifact compact.
    """
    base_extra: dict[str, Any] = {
        "stage_used": rl_out.get("stage_used"),
        "params": rl_out.get("params"),
        "checkpoint_path": train_out.get("checkpoint_path"),
        "trained": train_out.get("trained"),
        "feedback_path": feedback_path,
        "feature_meta": feature_meta,
    }
    loop_entry = _history_entry(
        round_no=round_no,
        selection_metric=selection_metric,
        score=score,
        reward=reward,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        artifacts=artifacts,
        extra={
            **base_extra,
            "train": {
                "checkpoint_path": train_out.get("checkpoint_path"),
                "trained": train_out.get("trained"),
            },
            "rl": {
                "stage_used": rl_out.get("stage_used"),
                "params": rl_out.get("params"),
            },
        },
    )
    metrics_entry = _history_entry(
        round_no=round_no,
        selection_metric=selection_metric,
        score=score,
        reward=reward,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        artifacts=artifacts,
        extra=dict(base_extra),
    )
    return loop_entry, metrics_entry


def reward_and_score(
    validation_metrics: dict[str, Any],
    selection_metric: str,
) -> tuple[float, float]:
    """Return ``(reward, score)`` for a single round.

    ``reward`` is always pulled from ``validation_metrics["reward"]``.
    ``score`` is the same as ``reward`` when the selection metric is
    ``"reward"``; otherwise it's the float-coerced value of
    ``validation_metrics[selection_metric]``.
    """
    reward = float(validation_metrics["reward"])
    score = reward if selection_metric == "reward" else float(validation_metrics[selection_metric])
    return reward, score


def build_artifact_paths_block(
    work: str,
    *,
    metrics_history_path: str,
    summary_path: str,
    wf: Any,
) -> dict[str, str]:
    """Return the ``artifacts`` view referenced by closed-loop summaries.

    ``wf`` is the work-file path helper from :mod:`forestseg._constants`;
    it is passed in (rather than imported) to keep this module free of
    a circular dependency on ``_constants`` while still letting callers
    redirect it for testing.
    """
    return {
        "fusion_selected": os.path.join(work, "fusion_selected.json"),
        "feature_feedback": wf(work, "feature_feedback"),
        "prob_dl": wf(work, "prob_dl"),
        "unc_dl": wf(work, "unc_dl"),
        "prob_fused": wf(work, "prob_fused"),
        "metrics_val": wf(work, "metrics_val"),
        "rl_history": wf(work, "rl_history"),
        "metrics_round_history": metrics_history_path,
        "closed_loop_summary": summary_path,
    }


def build_summary(
    *,
    status: str,
    preflight: dict[str, Any] | None,
    rounds_requested: int | None,
    rounds_completed: int,
    patience: int | None,
    min_delta: float | None,
    selection_metric: str | None,
    stopped_early: bool,
    best_round: int | None,
    best_score: float | None,
    best_entry: dict[str, Any] | None,
    restore_outcome: dict[str, list[str]],
    history: list[dict[str, Any]],
    artifacts: dict[str, str],
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical schema-v2 closed-loop summary dict.

    ``status`` is typically ``"ok"`` or ``"failed"``. The four
    rl_loop fields (``rounds_requested`` / ``patience`` /
    ``min_delta`` / ``selection_metric``) are accepted individually
    rather than as a typed bundle so that a partial failure summary
    can carry the values that were already validated before the
    raising one was hit. When ``status="failed"``, ``failure`` should
    carry the structured failure detail (``stage`` / ``round`` /
    ``error_type`` / ``message``).
    """
    summary: dict[str, Any] = {
        "schema_version": 2,
        "status": status,
        "preflight": preflight,
        "loop": {
            "rounds_requested": rounds_requested,
            "rounds_completed": rounds_completed,
            "selection_metric": selection_metric,
            "patience": patience,
            "min_delta": min_delta,
            "stopped_early": stopped_early,
        },
        "best": {
            "round": best_round if best_entry else None,
            "score": best_score if best_entry else None,
            "reward": best_entry.get("reward") if best_entry else None,
            "selection_metric": selection_metric,
            "entry": best_entry,
            "restore_outcome": restore_outcome,
        },
        "history": history,
        "artifacts": artifacts,
    }
    if failure is not None:
        summary["failure"] = failure
    return summary
