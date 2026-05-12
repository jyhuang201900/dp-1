"""Argparse + command-handler surface of the :mod:`forestseg` CLI.

The actual building blocks live in small private siblings:

- :mod:`forestseg._constants` — ``WORK_FILES``, fusion key tuples, ``wf``
- :mod:`forestseg._paths` — config load, ``work_dir`` setup, path probes
- :mod:`forestseg._validators` — small scalar validators
- :mod:`forestseg._fusion_config` — ``fusion`` section validation
- :mod:`forestseg._bandit_config` — ``rl_loop.bandit`` section validation
- :mod:`forestseg._label_resolution` — ``labels`` section validation
- :mod:`forestseg._artifacts` — round artifact paths + atomic restore
- :mod:`forestseg._feature_feedback` — feature-feedback transforms
- :mod:`forestseg._rl_history` — ``rl_history.json`` IO + validation

Every name moved into a private module is re-exported from this module
via explicit ``import X as X`` re-exports so the existing public
attribute surface (``forestseg.cli.X``) and the test suite imports stay
unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from typing import Any

import numpy as np

from ._artifacts import (
    RestoreOutputsError as RestoreOutputsError,
    _optional_restore_outcome,
    _round_artifact_paths,
)
from ._bandit_config import _resolve_bandit_config as _resolve_bandit_config
from ._bandit_runtime import BanditRuntime
from ._closed_loop import (
    ClosedLoopState as _ClosedLoopState,
    build_artifact_paths_block as _build_closed_loop_artifacts,
    build_best_round_restore_plan as _build_best_round_restore_plan,
    build_round_artifacts_map as _build_round_artifacts_map,
    build_round_history_entries as _build_round_history_entries,
    reward_and_score as _closed_loop_reward_and_score,
)
from ._constants import (
    REQUIRED_FUSION_FIXED_PARAM_KEYS as REQUIRED_FUSION_FIXED_PARAM_KEYS,
    REQUIRED_FUSION_GRID_KEYS as REQUIRED_FUSION_GRID_KEYS,
    WORK_FILES as WORK_FILES,
    wf,
)
from ._feature_feedback import _feature_feedback_transforms as _feature_feedback_transforms
from ._fusion_config import (
    _resolve_fusion_stage as _resolve_fusion_stage,
    _validate_fusion_config,
    _validate_fusion_param_candidates as _validate_fusion_param_candidates,
    _validate_fusion_param_value as _validate_fusion_param_value,
    _validate_fusion_params as _validate_fusion_params,
)
from ._label_resolution import (
    LabelGeometry,
    LabelSpec,
    _require_labels_for_preflight,
    _resolve_label_mode as _resolve_label_mode,
    _resolve_label_paths as _resolve_label_paths,
    resolve_label_spec,
)
from ._paths import (
    _ensure_directory_writable as _ensure_directory_writable,
    _log_progress,
    _require_existing_path,
    ensure_work,
    load_cfg,
)
from ._rl_history import (
    LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS as LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS,
    VALID_RL_SELECTION_METRICS,
    VALID_TRAIN_SELECTION_METRICS,
    _load_rl_history,
    _normalize_legacy_rl_history_entry as _normalize_legacy_rl_history_entry,
    _rewrite_latest_rl_history_entry,
    _validate_rl_history_entry as _validate_rl_history_entry,
)
from ._validators import (
    _validate_bool,
    _validate_choice,
    _validate_non_negative_float,
    _validate_non_negative_int,
    _validate_positive_float,
    _validate_positive_int,
    _validate_ratio,
    _validate_stage_int as _validate_stage_int,
    _validate_unit_interval,
)
from .export import export_generated_rasters, export_selected_params, export_vector_streaming
from .features import build_feature_stack
from .fusion import FusionParams, fuse_probabilities
from .infer import infer_to_files
from .io_raster import (
    normalize_raster_to_file,
    pixel_area_m2,
    process_aligned_inputs_to_output,
    process_single_input_to_outputs,
    read_band1,
    read_downsampled_band1,
)
from .labels import (
    LabelPoint,
    export_points_preview,
    load_points_json,
    read_label_points,
    read_label_points_from_two_files,
    save_points_json,
    split_points_by_grid,
    validate_label_points,
    validate_label_points_from_two_files,
)
from .metrics import save_metrics
from .postprocess import postprocess_mask
from .rl_env import build_state
from .rl_policy import choose_fusion_by_validation
from .scene_runtime import resolve_scene_input
from .spectral import spectral_probability
from .texture import texture_probability
from .train import train_supervised_model


def _copy_json_if_exists(src: str, dst: str) -> None:
    """Copy ``src`` JSON to ``dst`` (pretty-printed) when ``src`` exists."""
    if not os.path.exists(src):
        return
    with open(src, encoding="utf-8") as f:
        obj = json.load(f)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _require_json_copy(src: str, dst: str, label: str) -> None:
    """Copy ``src`` JSON to ``dst`` (pretty-printed); error if ``src`` missing."""
    if not os.path.exists(src):
        raise ValueError(f"Missing {label}: {src}")
    with open(src, encoding="utf-8") as f:
        obj = json.load(f)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _snapshot_required_json(src: str, dst: str, label: str) -> None:
    """Snapshot wrapper around :func:`_require_json_copy`."""
    _require_json_copy(src, dst, label)


def _copy_file_if_exists(src: str, dst: str) -> None:
    """Byte-copy ``src`` to ``dst`` when ``src`` exists."""
    if not os.path.exists(src):
        return
    shutil.copyfile(src, dst)


def _require_file_copy(src: str, dst: str, label: str) -> None:
    """Byte-copy ``src`` to ``dst``; error if ``src`` missing."""
    if not os.path.exists(src):
        raise ValueError(f"Missing {label}: {src}")
    shutil.copyfile(src, dst)


def _snapshot_required_file(src: str, dst: str, label: str) -> None:
    """Snapshot wrapper around :func:`_require_file_copy`."""
    _require_file_copy(src, dst, label)


def _restore_outputs_atomically(entries: list[tuple[str, str, str, str]]) -> list[str]:
    """Atomically restore a batch of artifacts via stage + backup + ``os.replace``.

    ``entries`` is a list of ``(src, dst, label, kind)`` tuples where
    ``kind`` is ``"json"`` (use :func:`_require_json_copy`) or anything
    else (use :func:`_require_file_copy`).

    Lives in :mod:`forestseg.cli` rather than :mod:`forestseg._artifacts`
    so the test suite's ``monkeypatch.setattr("forestseg.cli._require_*", ...)``
    hooks reach the actual call sites — the helpers are looked up in
    this module's globals at call time.
    """
    if not entries:
        return []
    work_dir = os.path.dirname(entries[0][1]) or None
    stage_dir = tempfile.mkdtemp(prefix="restore_best_round_", dir=work_dir)
    backup_dir = tempfile.mkdtemp(prefix="restore_best_round_backup_", dir=work_dir)
    staged_paths: list[tuple[str, str, str]] = []
    replaced_paths: list[tuple[str, str, bool]] = []
    restored_labels: list[str] = []
    try:
        try:
            for index, (src, dst, label, kind) in enumerate(entries):
                staged_path = os.path.join(stage_dir, f"{index:02d}_{os.path.basename(dst)}")
                if kind == "json":
                    _require_json_copy(src, staged_path, label)
                else:
                    _require_file_copy(src, staged_path, label)
                staged_paths.append((staged_path, dst, label))
        except Exception as exc:
            raise RestoreOutputsError(str(exc), restored_labels=restored_labels, failed_label=label) from exc
        try:
            for staged_path, dst, label in staged_paths:
                existed = os.path.exists(dst)
                backup_path = os.path.join(backup_dir, os.path.basename(dst))
                if existed:
                    shutil.copyfile(dst, backup_path)
                replaced_paths.append((backup_path, dst, existed))
                os.replace(staged_path, dst)
                restored_labels.append(label)
        except Exception as exc:
            for backup_path, dst, existed in reversed(replaced_paths):
                if existed and os.path.exists(backup_path):
                    shutil.copyfile(backup_path, dst)
                elif not existed and os.path.exists(dst):
                    os.remove(dst)
            raise RestoreOutputsError(str(exc), restored_labels=restored_labels, failed_label=label) from exc
        return restored_labels
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        shutil.rmtree(backup_dir, ignore_errors=True)


def _with_stage_override(cfg: dict[str, Any], stage_override: int | None) -> dict[str, Any]:
    """Return ``cfg`` (unchanged) or a shallow copy with ``fusion.stage`` set.

    Centralises the ``--stage`` override pattern shared by
    :func:`cmd_preflight_check`, :func:`cmd_run_closed_loop`,
    :func:`cmd_run_all`, and :func:`main`.
    """
    if stage_override is None:
        return cfg
    return {**cfg, "fusion": {**cfg.get("fusion", {}), "stage": stage_override}}


def cmd_preflight_check(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate config + on-disk preconditions before any pipeline stage runs.

    Reads ``labels`` / ``dl`` / ``dem`` / ``rl_loop`` / ``fusion`` sections,
    checks every required path exists, probes work-dir writability, and
    returns a ``preflight`` block embedded in the closed-loop summary.
    """
    work = ensure_work(cfg)
    lcfg = cfg.get("labels", {})
    dcfg = cfg.get("dl", {})
    dem_cfg = cfg.get("dem", {})
    loop_cfg = cfg.get("rl_loop", {})
    inp_cfg = cfg.get("input", {})
    checkpoint_path = str(dcfg.get("checkpoint_path") or os.path.join(work, "checkpoints", "best.pt"))
    checkpoint_dir = os.path.dirname(checkpoint_path) or work
    checks = {
        "scene_sh": _require_existing_path(str(cfg.get("scene_sh") or ""), "scene_sh"),
        "work_dir": work,
        "checkpoint_dir": checkpoint_dir,
    }
    checks.update(_require_labels_for_preflight(lcfg))
    checks["checkpoint_dir"] = _ensure_directory_writable(checks["checkpoint_dir"], "dl.checkpoint_path 所在目录")
    split_ratio = _validate_ratio(lcfg.get("split_ratio", 0.7), "labels.split_ratio")
    split_seed = _validate_non_negative_int(lcfg.get("split_seed", 42), "labels.split_seed")
    prefer_v1 = _validate_bool(inp_cfg.get("prefer_v1", True), "input.prefer_v1")
    fallback_quick = _validate_bool(inp_cfg.get("fallback_quick", True), "input.fallback_quick")
    class_field = str(lcfg.get("class_field", "class")).strip()
    positive_value = str(lcfg.get("positive_value", "1")).strip()
    negative_value = str(lcfg.get("negative_value", "0")).strip()
    if not class_field:
        raise ValueError("labels.class_field 不能为空。")
    if positive_value.lower() == negative_value.lower():
        raise ValueError("labels.positive_value 和 labels.negative_value 不能相同。")
    grid_size = _validate_positive_float(lcfg.get("grid_size", 1000.0), "labels.grid_size")
    tile_size = _validate_positive_int(dcfg.get("tile_size", 512), "dl.tile_size")
    stride = _validate_positive_int(dcfg.get("stride", 384), "dl.stride")
    batch_size = _validate_positive_int(dcfg.get("batch_size", 8), "dl.batch_size")
    epochs = _validate_positive_int(dcfg.get("epochs", 8), "dl.epochs")
    mc_dropout_passes = _validate_positive_int(dcfg.get("mc_dropout_passes", 4), "dl.mc_dropout_passes")
    sample_tile_size = _validate_positive_int(
        dcfg.get("sample_tile_size", dcfg.get("tile_size", 512)), "dl.sample_tile_size"
    )
    max_patches = _validate_positive_int(dcfg.get("max_patches", 2500), "dl.max_patches")
    aux_warmup_epochs = _validate_non_negative_int(dcfg.get("aux_warmup_epochs", 0), "dl.aux_warmup_epochs")
    if stride > tile_size:
        raise ValueError(f"dl.stride 不能大于 dl.tile_size，当前为 {stride} > {tile_size}。")
    rounds = _validate_positive_int(loop_cfg.get("rounds", 2), "rl_loop.rounds")
    patience = _validate_positive_int(loop_cfg.get("patience", 2), "rl_loop.patience")
    min_delta = _validate_non_negative_float(loop_cfg.get("min_delta", 0.001), "rl_loop.min_delta")
    dl_selection_metric = _validate_choice(
        dcfg.get("selection_metric", "f1"), "dl.selection_metric", VALID_TRAIN_SELECTION_METRICS
    )
    rl_selection_metric = _validate_choice(
        loop_cfg.get("selection_metric", "reward"), "rl_loop.selection_metric", VALID_RL_SELECTION_METRICS
    )
    bandit_config = _resolve_bandit_config(loop_cfg)
    val_threshold = _validate_unit_interval(dcfg.get("val_threshold", 0.5), "dl.val_threshold")
    fusion_stage, fixed_params, _, grid_keys = _validate_fusion_config(cfg)
    scene = resolve_scene_input(
        scene_sh=str(cfg.get("scene_sh") or ""),
        prefer_v1=prefer_v1,
        fallback_quick=fallback_quick,
    )
    checks["input_tif"] = scene.input_tif
    checks["generated_export_dir"] = _ensure_directory_writable(
        scene.generated_export_dir, "scene.generated_export_dir"
    )
    dem_path = dem_cfg.get("path")
    if dem_path:
        checks["dem_path"] = _require_existing_path(str(dem_path), "dem.path")
    warnings: list[str] = []
    if rounds == 1:
        warnings.append("rl_loop.rounds=1，将不会形成多轮闭环优化。")
    if patience > rounds:
        warnings.append("rl_loop.patience 大于 rounds，早停阈值实际上不会触发。")
    report = {
        "schema_version": 2,
        "ok": True,
        "can_run": True,
        "status": "ok" if not warnings else "warning",
        "errors": [],
        "warnings": warnings,
        "checks": checks,
        "config": {
            "labels": {
                "layer": lcfg.get("layer"),
                "class_field": class_field,
                "positive_value": positive_value,
                "negative_value": negative_value,
                "split_ratio": split_ratio,
                "grid_size": grid_size,
                "split_seed": split_seed,
            },
            "dl": {
                "tile_size": tile_size,
                "stride": stride,
                "batch_size": batch_size,
                "epochs": epochs,
                "mc_dropout_passes": mc_dropout_passes,
                "sample_tile_size": sample_tile_size,
                "max_patches": max_patches,
                "aux_warmup_epochs": aux_warmup_epochs,
                "val_threshold": val_threshold,
                "selection_metric": dl_selection_metric,
            },
            "rl_loop": {
                "rounds": rounds,
                "patience": patience,
                "min_delta": min_delta,
                "selection_metric": rl_selection_metric,
                "bandit": bandit_config,
            },
            "fusion": {
                "stage": fusion_stage,
                "grid_keys": grid_keys,
                "fixed_param_keys": sorted(fixed_params.keys()) if isinstance(fixed_params, dict) else [],
            },
        },
    }
    report_path = os.path.join(work, "preflight_check.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return {"report_path": report_path, **report}


def cmd_prepare_input(cfg: dict[str, Any]) -> dict[str, Any]:
    """Crop / reproject the source raster into ``work/input.tif`` + scene meta."""
    work = ensure_work(cfg)
    inp_cfg = cfg.get("input", {})
    scene = resolve_scene_input(
        scene_sh=cfg["scene_sh"],
        prefer_v1=_validate_bool(inp_cfg.get("prefer_v1", True), "input.prefer_v1"),
        fallback_quick=_validate_bool(inp_cfg.get("fallback_quick", True), "input.fallback_quick"),
    )
    out_path = wf(work, "input")
    pmin, pmax = cfg.get("preprocess", {}).get("clip_percentiles", [2, 98])
    stats = normalize_raster_to_file(scene.input_tif, out_path, float(pmin), float(pmax))
    meta = {
        "scene_id": scene.scene_id,
        "cropped_raster": scene.cropped_raster,
        "generated_export_dir": scene.generated_export_dir,
        "input_tif": scene.input_tif,
        "prepared_tif": out_path,
        "normalize_stats": stats,
    }
    with open(os.path.join(work, "scene_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _load_scene_meta(work: str) -> dict[str, Any]:
    p = os.path.join(work, "scene_meta.json")
    if not os.path.exists(p):
        raise FileNotFoundError("scene_meta.json missing. Run prepare-input first.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def cmd_build_spec_tex(cfg: dict[str, Any]) -> dict[str, str]:
    """Compute spectral / texture probability rasters from the prepared input."""
    work = ensure_work(cfg)
    prepared = wf(work, "input")
    if not os.path.exists(prepared):
        cmd_prepare_input(cfg)
    s_cfg = cfg.get("spectral", {})
    local_block_size = int(s_cfg.get("local_block_size", 51))
    spec_halo = max(local_block_size // 2 + 2, 32)
    core_size = 2048

    def spectral_processor(arr: np.ndarray):
        return spectral_probability(
            arr,
            quantile=float(s_cfg.get("quantile", 0.35)),
            otsu_weight=float(s_cfg.get("otsu_weight", 0.4)),
            quantile_weight=float(s_cfg.get("quantile_weight", 0.3)),
            local_weight=float(s_cfg.get("local_weight", 0.3)),
            local_block_size=local_block_size,
        )

    process_single_input_to_outputs(
        input_path=prepared,
        output_specs=[(wf(work, "prob_spec"), "float32"), (wf(work, "conf_spec"), "float32")],
        core_size=core_size,
        halo=spec_halo,
        processor=spectral_processor,
    )

    t_cfg = cfg.get("texture", {})
    tex_windows = [int(v) for v in t_cfg.get("windows", [7, 15, 31])]
    tex_halo = max(max(tex_windows, default=31) // 2 + 2, 32)

    def texture_processor(arr: np.ndarray):
        prob_tex, tex_complexity = texture_probability(
            arr,
            windows=tex_windows,
            lbp_radius=int(t_cfg.get("lbp_radius", 1)),
            lbp_points=int(t_cfg.get("lbp_points", 8)),
        )
        conf_tex = (2.0 * np.abs(prob_tex - 0.5)).astype(np.float32)
        return prob_tex, tex_complexity, conf_tex

    process_single_input_to_outputs(
        input_path=prepared,
        output_specs=[
            (wf(work, "prob_tex"), "float32"),
            (wf(work, "tex_complexity"), "float32"),
            (wf(work, "conf_tex"), "float32"),
        ],
        core_size=core_size,
        halo=tex_halo,
        processor=texture_processor,
    )
    return {
        "prob_spec": wf(work, "prob_spec"),
        "conf_spec": wf(work, "conf_spec"),
        "prob_tex": wf(work, "prob_tex"),
        "tex_complexity": wf(work, "tex_complexity"),
        "conf_tex": wf(work, "conf_tex"),
    }


def _label_geometry_from_raster(prepared: str) -> LabelGeometry:
    """Read CRS and grid origin from the prepared raster.

    Mirrors what the label commands need from
    :func:`forestseg.io_raster.read_band1` but packages it as the typed
    :class:`LabelGeometry` bundle shared by the prepare / check label
    commands.
    """
    raster = read_band1(prepared)
    crs = raster.profile.get("crs")
    transform = raster.profile.get("transform")
    origin_x = float(transform.c) if transform is not None else 0.0
    origin_y = float(transform.f) if transform is not None else 0.0
    return LabelGeometry(crs=crs, origin_x=origin_x, origin_y=origin_y)


def _validate_label_points_for_spec(spec: LabelSpec, geometry: LabelGeometry) -> dict[str, Any]:
    """Dispatch to ``validate_label_points*`` based on ``spec.mode``.

    Lives in :mod:`forestseg.cli` (rather than ``_label_resolution``)
    so the test suite can monkeypatch
    ``forestseg.cli.validate_label_points`` /
    ``forestseg.cli.validate_label_points_from_two_files`` and have
    those overrides apply to the actual call sites at runtime.
    """
    if spec.mode == "dual":
        assert spec.positive_path is not None and spec.negative_path is not None
        return validate_label_points_from_two_files(
            positive_path=spec.positive_path,
            negative_path=spec.negative_path,
            positive_layer=spec.layer,
            negative_layer=spec.layer,
            class_field=spec.class_field,
            positive_value=spec.positive_value,
            negative_value=spec.negative_value,
            target_crs=geometry.crs,
            grid_size=spec.grid_size,
            origin_x=geometry.origin_x,
            origin_y=geometry.origin_y,
            train_ratio=spec.train_ratio,
            seed=spec.split_seed,
        )
    assert spec.single_path is not None
    return validate_label_points(
        path=spec.single_path,
        layer=spec.layer,
        class_field=spec.class_field,
        positive_value=spec.positive_value,
        negative_value=spec.negative_value,
        target_crs=geometry.crs,
        grid_size=spec.grid_size,
        origin_x=geometry.origin_x,
        origin_y=geometry.origin_y,
        train_ratio=spec.train_ratio,
        seed=spec.split_seed,
    )


def _read_label_points_for_spec(spec: LabelSpec, geometry: LabelGeometry) -> list[LabelPoint]:
    """Dispatch to ``read_label_points*`` based on ``spec.mode``.

    Lives in :mod:`forestseg.cli` for the same reason as
    :func:`_validate_label_points_for_spec` — preserving the
    monkeypatch contract on the ``forestseg.cli`` module surface.
    """
    if spec.mode == "dual":
        assert spec.positive_path is not None and spec.negative_path is not None
        return read_label_points_from_two_files(
            positive_path=spec.positive_path,
            negative_path=spec.negative_path,
            positive_layer=spec.layer,
            negative_layer=spec.layer,
            class_field=spec.class_field,
            positive_value=spec.positive_value,
            negative_value=spec.negative_value,
            target_crs=geometry.crs,
            grid_size=spec.grid_size,
            origin_x=geometry.origin_x,
            origin_y=geometry.origin_y,
        )
    assert spec.single_path is not None
    return read_label_points(
        path=spec.single_path,
        layer=spec.layer,
        class_field=spec.class_field,
        positive_value=spec.positive_value,
        negative_value=spec.negative_value,
        target_crs=geometry.crs,
        grid_size=spec.grid_size,
        origin_x=geometry.origin_x,
        origin_y=geometry.origin_y,
    )


def cmd_prepare_label_points(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate label sources, snap to grid, and write train/val splits."""
    work = ensure_work(cfg)
    prepared = wf(work, "input")
    if not os.path.exists(prepared):
        cmd_prepare_input(cfg)
    lcfg = cfg.get("labels", {})
    spec = resolve_label_spec(lcfg)
    geometry = _label_geometry_from_raster(prepared)
    validation_report = _validate_label_points_for_spec(spec, geometry)
    validation_risk = dict(validation_report.get("risk") or {})
    if not bool(validation_risk.get("can_run", True)):
        blocking = list(validation_risk.get("blocking") or [])
        raise ValueError("；".join(blocking) if blocking else "标签检查未通过，无法生成训练/验证样本。")

    points = _read_label_points_for_spec(spec, geometry)
    train_points, val_points, split_meta = split_points_by_grid(
        points,
        train_ratio=spec.train_ratio,
        seed=spec.split_seed,
    )
    train_path = save_points_json(wf(work, "samples_train"), train_points, meta=split_meta)
    val_path = save_points_json(wf(work, "samples_val"), val_points, meta=split_meta)
    preview_train = os.path.join(work, "label_train_preview.gpkg")
    preview_val = os.path.join(work, "label_val_preview.gpkg")
    export_points_preview(preview_train, train_points, crs=geometry.crs, layer="train")
    export_points_preview(preview_val, val_points, crs=geometry.crs, layer="val")
    return {
        "samples_train": train_path,
        "samples_val": val_path,
        "split": split_meta,
        "preview_train": preview_train,
        "preview_val": preview_val,
    }


def cmd_check_label_points(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run label validation only (no train/val split, no IO mutations)."""
    work = ensure_work(cfg)
    prepared = wf(work, "input")
    if not os.path.exists(prepared):
        cmd_prepare_input(cfg)
    lcfg = cfg.get("labels", {})
    spec = resolve_label_spec(lcfg)
    geometry = _label_geometry_from_raster(prepared)
    report = _validate_label_points_for_spec(spec, geometry)
    risk = dict(report.get("risk") or {})
    report["schema_version"] = 2
    report["status"] = str(risk.get("status", "ok"))
    report["can_run"] = bool(risk.get("can_run", True))
    report["warnings"] = list(risk.get("warnings") or [])
    report["errors"] = list(risk.get("blocking") or [])
    report_path = os.path.join(work, "label_check_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return {"report_path": report_path, **report}


def cmd_build_feature_stack(cfg: dict[str, Any]) -> dict[str, Any]:
    """Assemble the multi-channel feature stack used by the DL trainer."""
    work = ensure_work(cfg)
    prepared = wf(work, "input")
    if not os.path.exists(prepared):
        cmd_prepare_input(cfg)
    for k in ("prob_spec", "prob_tex", "tex_complexity"):
        if not os.path.exists(wf(work, k)):
            cmd_build_spec_tex(cfg)
            break
    dem_cfg = cfg.get("dem", {})
    dem_path = dem_cfg.get("path")
    dem_out = os.path.join(work, "dem_prepared.tif") if dem_path else None
    meta = build_feature_stack(
        feature_paths=[prepared, wf(work, "prob_spec"), wf(work, "prob_tex"), wf(work, "tex_complexity")],
        feature_names=["image", "spec_prob", "tex_prob", "tex_complexity"],
        output_path=wf(work, "feature_stack"),
        dem_path=str(dem_path) if dem_path else None,
        dem_output_path=dem_out,
        feature_transforms=_feature_feedback_transforms(work),
    )
    return meta


def cmd_train_or_load_dl(cfg: dict[str, Any], force_train: bool = False) -> dict[str, Any]:
    """Train (or reuse) the DL model and emit ``prob_dl`` + uncertainty rasters."""
    work = ensure_work(cfg)
    if not os.path.exists(wf(work, "feature_stack")):
        cmd_build_feature_stack(cfg)
    if not os.path.exists(wf(work, "samples_train")) or not os.path.exists(wf(work, "samples_val")):
        cmd_prepare_label_points(cfg)
    dcfg = cfg.get("dl", {})
    _validate_choice(dcfg.get("selection_metric", "f1"), "dl.selection_metric", VALID_TRAIN_SELECTION_METRICS)
    tile_size = _validate_positive_int(dcfg.get("tile_size", 512), "dl.tile_size")
    stride = _validate_positive_int(dcfg.get("stride", 384), "dl.stride")
    if stride > tile_size:
        raise ValueError(f"dl.stride 不能大于 dl.tile_size，当前为 {stride} > {tile_size}。")
    _validate_positive_int(dcfg.get("batch_size", 8), "dl.batch_size")
    _validate_positive_int(dcfg.get("epochs", 8), "dl.epochs")
    _validate_positive_int(dcfg.get("mc_dropout_passes", 4), "dl.mc_dropout_passes")
    ckpt_path = dcfg.get("checkpoint_path", os.path.join(work, "checkpoints", "best.pt"))
    trained = False
    train_info: dict[str, Any] = {}
    needs_training = force_train or not os.path.exists(ckpt_path)
    if needs_training:
        _validate_unit_interval(dcfg.get("val_threshold", 0.5), "dl.val_threshold")
        train_points, _ = load_points_json(wf(work, "samples_train"))
        val_points, _ = load_points_json(wf(work, "samples_val"))
        result = train_supervised_model(
            feature_stack_path=wf(work, "feature_stack"),
            train_points=train_points,
            val_points=val_points,
            cfg=dcfg,
            checkpoint_path=ckpt_path,
        )
        trained = True
        train_info = {
            "checkpoint_path": result.checkpoint_path,
            "best_score": result.best_score,
            "metrics": result.metrics,
        }
        save_metrics(wf(work, "metrics_val"), result.metrics)
    batch_size = _validate_positive_int(dcfg.get("batch_size", 8), "dl.batch_size")
    mc_dropout_passes = _validate_positive_int(dcfg.get("mc_dropout_passes", 4), "dl.mc_dropout_passes")
    infer_to_files(
        input_path=wf(work, "feature_stack"),
        checkpoint_path=ckpt_path,
        encoder_name=str(dcfg.get("encoder_name", "resnet34")),
        tile_size=int(dcfg.get("tile_size", 512)),
        stride=int(dcfg.get("stride", 384)),
        batch_size=batch_size,
        prob_out_path=wf(work, "prob_dl"),
        unc_out_path=wf(work, "unc_dl"),
        mc_dropout_passes=mc_dropout_passes,
    )
    return {
        "trained": trained,
        "train_info": train_info,
        "checkpoint_path": ckpt_path,
        "prob_dl_path": wf(work, "prob_dl"),
        "unc_dl_path": wf(work, "unc_dl"),
    }


def cmd_run_rl_fusion(cfg: dict[str, Any], stage_override: int | None = None) -> dict[str, Any]:
    """Pick fusion params (grid or bandit), fuse probabilities, append rl_history."""
    work = ensure_work(cfg)
    loop_cfg = cfg.get("rl_loop", {})
    rl_selection_metric = _validate_choice(
        loop_cfg.get("selection_metric", "reward"),
        "rl_loop.selection_metric",
        VALID_RL_SELECTION_METRICS,
    )
    bandit_config = _resolve_bandit_config(loop_cfg)
    fusion_stage, fixed_params, grid_params, _ = _validate_fusion_config(cfg, stage_override=stage_override)
    history_path = wf(work, "rl_history")
    history = _load_rl_history(history_path)
    for key in ("input", "prob_spec", "prob_tex", "prob_dl", "tex_complexity"):
        if not os.path.exists(wf(work, key)):
            if key == "input":
                cmd_prepare_input(cfg)
            elif key in ("prob_spec", "prob_tex", "tex_complexity"):
                cmd_build_spec_tex(cfg)
            else:
                cmd_train_or_load_dl(cfg)
    preview_dim = int(cfg.get("fusion", {}).get("preview_max_dim", 2048))
    img01_small = read_downsampled_band1(wf(work, "input"), max_dim=preview_dim).arr
    prob_spec_small = read_downsampled_band1(wf(work, "prob_spec"), max_dim=preview_dim).arr
    prob_tex_small = read_downsampled_band1(wf(work, "prob_tex"), max_dim=preview_dim).arr
    prob_dl_small = read_downsampled_band1(wf(work, "prob_dl"), max_dim=preview_dim).arr
    tex_complexity_small = read_downsampled_band1(wf(work, "tex_complexity"), max_dim=preview_dim).arr
    state = build_state(prob_dl_small, prob_spec_small, prob_tex_small, tex_complexity_small, img01_small)
    metrics: dict[str, Any] = {}
    validation_reward: dict[str, Any] = {}
    stage = fusion_stage

    bandit = BanditRuntime(bandit_config=bandit_config, fusion_stage=stage, work=work)

    if not os.path.exists(wf(work, "samples_val")):
        cmd_prepare_label_points(cfg)
    val_points, split_meta = load_points_json(wf(work, "samples_val"))

    runtime_grid_params = bandit.select_action(grid_params)

    selected, validation_reward, stage_used = choose_fusion_by_validation(
        stage=stage,
        prob_dl_path=wf(work, "prob_dl"),
        prob_spec_path=wf(work, "prob_spec"),
        prob_tex_path=wf(work, "prob_tex"),
        val_points=val_points,
        fixed_params=fixed_params,
        grid_params=runtime_grid_params,
    )

    bandit.record_reward(float(validation_reward.get("reward", 0.0)))

    with open(wf(work, "feature_feedback"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "lambda_spec": selected.lambda_spec,
                "lambda_tex": selected.lambda_tex,
                "threshold": selected.threshold,
                "min_area_m2": selected.min_area_m2,
                "morph_kernel": selected.morph_kernel,
                "shadow_penalty": selected.shadow_penalty,
                "reward": validation_reward.get("reward", 0.0),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    def fusion_processor(prob_dl_arr: np.ndarray, prob_spec_arr: np.ndarray, prob_tex_arr: np.ndarray) -> np.ndarray:
        return fuse_probabilities(prob_dl_arr, prob_spec_arr, prob_tex_arr, selected)

    process_aligned_inputs_to_output(
        input_paths=[wf(work, "prob_dl"), wf(work, "prob_spec"), wf(work, "prob_tex")],
        output_path=wf(work, "prob_fused"),
        dtype="float32",
        core_size=4096,
        halo=0,
        processor=fusion_processor,
    )
    if os.path.exists(wf(work, "metrics_val")):
        with open(wf(work, "metrics_val"), encoding="utf-8") as f:
            metrics = json.load(f)
    selected_score = validation_reward.get("reward")
    if rl_selection_metric != "reward":
        if rl_selection_metric not in validation_reward:
            raise ValueError(f"rl_loop.selection_metric={rl_selection_metric} 未在 validation_reward 中找到可用分数。")
        selected_score = validation_reward[rl_selection_metric]
    rl_payload = {
        "schema_version": 2,
        "round": None,
        "selection_metric": rl_selection_metric,
        "score": selected_score,
        "reward": validation_reward.get("reward"),
        "stage_used": stage_used,
        "params": {
            "lambda_spec": selected.lambda_spec,
            "lambda_tex": selected.lambda_tex,
            "threshold": selected.threshold,
            "min_area_m2": selected.min_area_m2,
            "morph_kernel": selected.morph_kernel,
            "shadow_penalty": selected.shadow_penalty,
        },
        "policy_type": bandit.policy_type,
        "bandit": bandit.meta,
        "state": state,
        "train_metrics": metrics,
        "validation_metrics": validation_reward,
        "split_meta": split_meta,
        "preview_shape": list(prob_dl_small.shape),
    }
    history.append(rl_payload)
    for idx, entry in enumerate(history, start=1):
        if isinstance(entry, dict) and not entry.get("round"):
            entry["round"] = idx
    with open(os.path.join(work, "fusion_selected.json"), "w", encoding="utf-8") as f:
        json.dump(rl_payload, f, ensure_ascii=False, indent=2)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return {
        "stage_used": stage_used,
        "params": rl_payload["params"],
        "score": selected_score,
        "prob_fused": wf(work, "prob_fused"),
        "metrics": metrics,
        "validation_reward": validation_reward,
    }


def _load_selected_params(work: str, cfg: dict[str, Any]) -> FusionParams:
    p = os.path.join(work, "fusion_selected.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            obj = json.load(f)
        prm = obj.get("params", {})
        return FusionParams(
            lambda_spec=float(prm.get("lambda_spec", 0.2)),
            lambda_tex=float(prm.get("lambda_tex", 0.2)),
            threshold=float(prm.get("threshold", 0.5)),
            min_area_m2=float(prm.get("min_area_m2", 200)),
            morph_kernel=int(prm.get("morph_kernel", 3)),
            shadow_penalty=float(prm.get("shadow_penalty", 0.5)),
        )
    fp = cfg.get("fusion", {}).get("fixed_params", {})
    return FusionParams(
        lambda_spec=float(fp.get("lambda_spec", 0.2)),
        lambda_tex=float(fp.get("lambda_tex", 0.2)),
        threshold=float(fp.get("threshold", 0.5)),
        min_area_m2=float(fp.get("min_area_m2", 200)),
        morph_kernel=int(fp.get("morph_kernel", 3)),
        shadow_penalty=float(fp.get("shadow_penalty", 0.5)),
    )


def cmd_postprocess_export(cfg: dict[str, Any]) -> dict[str, Any]:
    """Thresholding + morphology on ``prob_fused``; export final raster + vector."""
    work = ensure_work(cfg)
    if not os.path.exists(wf(work, "prob_fused")):
        cmd_run_rl_fusion(cfg)
    meta = _load_scene_meta(work)
    cropped_raster = meta["cropped_raster"]
    output_dir = meta["generated_export_dir"]
    profile = read_band1(wf(work, "input")).profile
    params = _load_selected_params(work, cfg)
    ppcfg = cfg.get("postprocess", {})
    core_size = 2048
    halo = max(int(ppcfg.get("closing_kernel", params.morph_kernel)) * 8, 64)

    def post_processor(prob_fused_arr: np.ndarray, img01_arr: np.ndarray, tex_arr: np.ndarray) -> np.ndarray:
        return postprocess_mask(
            prob_fused=prob_fused_arr,
            img01=img01_arr,
            tex_complexity=tex_arr,
            threshold=float(params.threshold),
            pixel_area_m2=pixel_area_m2(profile),
            min_area_m2=float(params.min_area_m2),
            opening_kernel=int(ppcfg.get("opening_kernel", params.morph_kernel)),
            closing_kernel=int(ppcfg.get("closing_kernel", params.morph_kernel)),
            shadow_penalty=float(params.shadow_penalty),
        )

    process_aligned_inputs_to_output(
        input_paths=[wf(work, "prob_fused"), wf(work, "input"), wf(work, "tex_complexity")],
        output_path=wf(work, "mask_final"),
        dtype="uint8",
        core_size=core_size,
        halo=halo,
        processor=post_processor,
    )
    out_files = export_generated_rasters(
        output_dir=output_dir,
        cropped_raster=cropped_raster,
        prob_spec_path=wf(work, "prob_spec"),
        prob_tex_path=wf(work, "prob_tex"),
        prob_dl_path=wf(work, "prob_dl"),
        prob_fused_path=wf(work, "prob_fused"),
        mask_final_path=wf(work, "mask_final"),
    )
    gpkg = os.path.join(output_dir, f"{cropped_raster}_forest_final.gpkg")
    export_vector_streaming(out_files["mask_final"], gpkg)
    params_dict = {
        "lambda_spec": params.lambda_spec,
        "lambda_tex": params.lambda_tex,
        "threshold": params.threshold,
        "min_area_m2": params.min_area_m2,
        "morph_kernel": params.morph_kernel,
        "shadow_penalty": params.shadow_penalty,
    }
    param_path = export_selected_params(output_dir, cropped_raster, params_dict)
    return {
        **out_files,
        "vector": gpkg,
        "params_json": param_path,
        "metrics_val": wf(work, "metrics_val"),
        "rl_history": wf(work, "rl_history"),
    }


def _persist_closed_loop_outputs(
    *,
    metrics_history_path: str,
    summary_path: str,
    metrics_round_history: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Write the per-round metrics history and the run summary to disk."""
    with open(metrics_history_path, "w", encoding="utf-8") as f:
        json.dump(metrics_round_history, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def cmd_run_closed_loop(cfg: dict[str, Any], stage_override: int | None = None) -> dict[str, Any]:
    """Drive the multi-round closed-loop training + RL-fusion + select-best workflow.

    Maintains a :class:`_ClosedLoopState` for the lifetime of the call;
    every checkpoint that the failure summary depends on (validated
    rl_loop fields, accumulated history, current pipeline stage, etc.)
    is recorded on that state object so the ``except`` branch can
    produce a consistent schema-v2 summary regardless of how far the
    run got.
    """
    work = ensure_work(cfg)
    _log_progress("run-closed-loop: preflight")
    summary_path = wf(work, "closed_loop_summary")
    metrics_history_path = os.path.join(work, "metrics_round_history.json")
    state = _ClosedLoopState(
        artifact_paths_view=_build_closed_loop_artifacts(
            work,
            metrics_history_path=metrics_history_path,
            summary_path=summary_path,
            wf=wf,
        ),
    )

    try:
        state.preflight = cmd_preflight_check(_with_stage_override(cfg, stage_override))
        loop_cfg = cfg.get("rl_loop", {})
        state.rounds = _validate_positive_int(loop_cfg.get("rounds", 2), "rl_loop.rounds")
        state.patience = _validate_positive_int(loop_cfg.get("patience", 2), "rl_loop.patience")
        state.min_delta = _validate_non_negative_float(loop_cfg.get("min_delta", 0.001), "rl_loop.min_delta")
        state.selection_metric = _validate_choice(
            loop_cfg.get("selection_metric", "reward"),
            "rl_loop.selection_metric",
            VALID_RL_SELECTION_METRICS,
        )

        state.current_stage = "prepare_input"
        _log_progress("run-closed-loop: prepare-input")
        cmd_prepare_input(cfg)
        state.current_stage = "build_spec_tex"
        _log_progress("run-closed-loop: build-spec-tex")
        cmd_build_spec_tex(cfg)
        state.current_stage = "prepare_label_points"
        _log_progress("run-closed-loop: prepare-label-points")
        cmd_prepare_label_points(cfg)

        rounds = state.rounds
        selection_metric = state.selection_metric
        patience = state.patience
        min_delta = state.min_delta
        for round_idx in range(rounds):
            round_no = round_idx + 1
            _log_progress(f"run-closed-loop: round {round_no}/{rounds} start")
            state.current_round = round_no
            round_paths = _round_artifact_paths(work, round_no)
            state.current_stage = "build_feature_stack"
            _log_progress(f"run-closed-loop: round {round_no}/{rounds} build-feature-stack")
            cmd_build_feature_stack(cfg)
            feature_meta_path = os.path.splitext(wf(work, "feature_stack"))[0] + "_meta.json"
            feature_meta: dict[str, Any] = {}
            if os.path.exists(feature_meta_path):
                with open(feature_meta_path, encoding="utf-8") as f:
                    feature_meta = json.load(f)
                with open(round_paths["feature_meta"], "w", encoding="utf-8") as f:
                    json.dump(feature_meta, f, ensure_ascii=False, indent=2)

            state.current_stage = "train_or_load_dl"
            _log_progress(f"run-closed-loop: round {round_no}/{rounds} train-or-load-dl")
            train_out = cmd_train_or_load_dl(cfg, force_train=True)
            train_metrics = train_out.get("train_info", {}).get("metrics", {})
            with open(round_paths["train_metrics"], "w", encoding="utf-8") as f:
                json.dump(train_metrics, f, ensure_ascii=False, indent=2)

            state.current_stage = "run_rl_fusion"
            _log_progress(f"run-closed-loop: round {round_no}/{rounds} run-rl-fusion")
            rl_out = cmd_run_rl_fusion(cfg, stage_override=stage_override)
            validation_metrics = rl_out.get("validation_reward") or {}
            reward, score = _closed_loop_reward_and_score(validation_metrics, selection_metric)
            rl_payload_path = os.path.join(work, "fusion_selected.json")
            with open(rl_payload_path, encoding="utf-8") as f:
                rl_payload = json.load(f)
            rl_payload["round"] = round_no
            rl_payload["selection_metric"] = selection_metric
            rl_payload["score"] = score
            with open(rl_payload_path, "w", encoding="utf-8") as f:
                json.dump(rl_payload, f, ensure_ascii=False, indent=2)
            _rewrite_latest_rl_history_entry(wf(work, "rl_history"), round_no, selection_metric, score)
            feature_feedback_path = wf(work, "feature_feedback")
            state.current_stage = "snapshot_round_artifacts"
            _snapshot_required_json(rl_payload_path, round_paths["rl_payload"], f"round {round_no} rl_payload")
            _snapshot_required_json(
                feature_feedback_path, round_paths["feature_feedback"], f"round {round_no} feature_feedback"
            )
            _snapshot_required_json(
                wf(work, "metrics_val"), round_paths["metrics_val"], f"round {round_no} metrics_val"
            )
            _snapshot_required_file(wf(work, "prob_dl"), round_paths["prob_dl"], f"round {round_no} prob_dl")
            _copy_file_if_exists(wf(work, "unc_dl"), round_paths["unc_dl"])
            _snapshot_required_file(wf(work, "prob_fused"), round_paths["prob_fused"], f"round {round_no} prob_fused")
            artifact_paths = _build_round_artifacts_map(round_paths)
            loop_entry, metrics_entry = _build_round_history_entries(
                round_no=round_no,
                selection_metric=selection_metric,
                score=score,
                reward=reward,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                artifacts=artifact_paths,
                rl_out=rl_out,
                train_out=train_out,
                feedback_path=round_paths["feature_feedback"],
                feature_meta=feature_meta,
            )
            state.loop_history.append(loop_entry)
            state.metrics_round_history.append(metrics_entry)

            if state.best_score is None or score > state.best_score + min_delta:
                state.best_score = score
                state.best_round = round_no
                state.best_entry = loop_entry
                state.stagnant_rounds = 0
                _log_progress(
                    f"run-closed-loop: round {round_no}/{rounds} best updated ({selection_metric}={score:.6f})"
                )
            else:
                state.stagnant_rounds += 1
                _log_progress(
                    f"run-closed-loop: round {round_no}/{rounds} no improvement"
                    f" ({selection_metric}={score:.6f}, stagnant={state.stagnant_rounds}/{patience})"
                )
                if state.stagnant_rounds >= patience:
                    state.loop_stopped_early = True
                    _log_progress("run-closed-loop: early stop triggered")
                    break

        if state.best_entry:
            _log_progress(f"run-closed-loop: restore best round {state.best_round}")
            state.current_stage = "restore_best_round"
            restore_entries, state.optional_restore_labels = _build_best_round_restore_plan(
                work,
                state.best_entry.get("artifacts", {}),
                wf=wf,
            )
            state.restored_labels = _restore_outputs_atomically(restore_entries)
            state.failed_restore_labels = []
            state.restore_outcome = _optional_restore_outcome(
                state.optional_restore_labels,
                state.restored_labels,
                state.failed_restore_labels,
            )

        summary = state.build_summary(status="ok")
        _persist_closed_loop_outputs(
            metrics_history_path=metrics_history_path,
            summary_path=summary_path,
            metrics_round_history=state.metrics_round_history,
            summary=summary,
        )
        _log_progress("run-closed-loop: completed")
        return summary
    except Exception as exc:
        raised_exc = exc
        if isinstance(exc, RestoreOutputsError):
            state.restored_labels = exc.restored_labels
            state.failed_restore_labels = (
                [exc.failed_label] if exc.failed_label in state.optional_restore_labels else []
            )
            raised_exc = exc.__cause__ or ValueError(str(exc))
        if state.best_entry and state.current_stage == "restore_best_round":
            state.restore_outcome = _optional_restore_outcome(
                state.optional_restore_labels,
                state.restored_labels,
                state.failed_restore_labels,
            )
        failure_summary = state.build_summary(
            status="failed",
            failure={
                "stage": state.current_stage,
                "round": state.current_round,
                "error_type": type(raised_exc).__name__,
                "message": str(raised_exc),
            },
        )
        _persist_closed_loop_outputs(
            metrics_history_path=metrics_history_path,
            summary_path=summary_path,
            metrics_round_history=state.metrics_round_history,
            summary=failure_summary,
        )
        raise raised_exc


def cmd_run_all(cfg: dict[str, Any], force_train: bool = False, stage_override: int | None = None) -> dict[str, Any]:
    """End-to-end driver: preflight -> stages -> closed loop -> postprocess."""
    _log_progress("run-all: preflight-check")
    cmd_preflight_check(_with_stage_override(cfg, stage_override))
    _log_progress("run-all: prepare-input")
    cmd_prepare_input(cfg)
    _log_progress("run-all: build-spec-tex")
    cmd_build_spec_tex(cfg)
    _log_progress("run-all: prepare-label-points")
    cmd_prepare_label_points(cfg)
    _log_progress("run-all: build-feature-stack")
    cmd_build_feature_stack(cfg)
    if force_train:
        _log_progress("run-all: train-or-load-dl (force-train)")
        cmd_train_or_load_dl(cfg, force_train=True)
    _log_progress("run-all: run-closed-loop")
    cmd_run_closed_loop(cfg, stage_override=stage_override)
    _log_progress("run-all: postprocess-export")
    result = cmd_postprocess_export(cfg)
    _log_progress("run-all: completed")
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KH-4 supervised forest extraction pipeline")
    p.add_argument(
        "command",
        choices=[
            "prepare-input",
            "build-spec-tex",
            "check-label-points",
            "preflight-check",
            "prepare-label-points",
            "build-feature-stack",
            "train-or-load-dl",
            "run-rl-fusion",
            "run-closed-loop",
            "postprocess-export",
            "run-all",
        ],
    )
    p.add_argument("--config", default="E:/CORONA/DeepLearning/configs/pipeline.yaml")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--stage", type=int, default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = load_cfg(args.config)
    if args.command == "prepare-input":
        out = cmd_prepare_input(cfg)
    elif args.command == "build-spec-tex":
        out = cmd_build_spec_tex(cfg)
    elif args.command == "check-label-points":
        out = cmd_check_label_points(cfg)
    elif args.command == "preflight-check":
        out = cmd_preflight_check(_with_stage_override(cfg, args.stage))
    elif args.command == "prepare-label-points":
        out = cmd_prepare_label_points(cfg)
    elif args.command == "build-feature-stack":
        out = cmd_build_feature_stack(cfg)
    elif args.command == "train-or-load-dl":
        out = cmd_train_or_load_dl(cfg, force_train=args.force_train)
    elif args.command == "run-rl-fusion":
        out = cmd_run_rl_fusion(cfg, stage_override=args.stage)
    elif args.command == "run-closed-loop":
        out = cmd_run_closed_loop(cfg, stage_override=args.stage)
    elif args.command == "postprocess-export":
        out = cmd_postprocess_export(cfg)
    elif args.command == "run-all":
        out = cmd_run_all(cfg, force_train=args.force_train, stage_override=args.stage)
    else:
        raise ValueError(args.command)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
