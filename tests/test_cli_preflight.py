import json
import os
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from forestseg.cli import (
    _load_rl_history,
    cmd_check_label_points,
    cmd_preflight_check,
    cmd_prepare_input,
    cmd_prepare_label_points,
    cmd_run_all,
    cmd_run_closed_loop,
    cmd_run_rl_fusion,
    cmd_train_or_load_dl,
    main,
)
from forestseg.fusion import FusionParams
from forestseg.labels import load_points_json
from forestseg.scene_runtime import _parse_scene_file_fallback, _source_scene, resolve_scene_input
from forestseg.train import train_supervised_model

REQUIRED_FUSION_GRID = {
    "lambda_spec": [0.2],
    "lambda_tex": [0.2],
    "threshold": [0.5],
}

OPTIONAL_FUSION_POSTPROCESS_GRID = {
    "min_area_m2": [200.0],
    "morph_kernel": [3],
    "shadow_penalty": [0.5],
}


def _make_fusion_params(
    *, threshold, lambda_spec=0.2, lambda_tex=0.2, min_area_m2=200.0, morph_kernel=3, shadow_penalty=0.5
):
    return {
        "lambda_spec": lambda_spec,
        "lambda_tex": lambda_tex,
        "threshold": threshold,
        "min_area_m2": min_area_m2,
        "morph_kernel": morph_kernel,
        "shadow_penalty": shadow_penalty,
    }


def _make_feedback_payload(*, threshold, reward, **params):
    return {**_make_fusion_params(threshold=threshold, **params), "reward": reward}


def _make_fusion_selected_payload(*, threshold=None, params=None):
    payload_params = params if params is not None else {"threshold": threshold}
    return {"params": payload_params}


def _make_train_result(*, checkpoint_path="ckpt.pt", train_metrics=None, trained=True):
    return {
        "checkpoint_path": checkpoint_path,
        "trained": trained,
        "train_info": {"metrics": deepcopy(train_metrics) if train_metrics is not None else {}},
    }


def _make_fixed_validation_selection(*, reward=0.7, f1=0.7, stage_used="validation_stage1"):
    return (
        FusionParams(
            lambda_spec=0.1,
            lambda_tex=0.0,
            threshold=0.5,
            min_area_m2=200.0,
            morph_kernel=3,
            shadow_penalty=0.5,
        ),
        {"reward": reward, "f1": f1},
        stage_used,
    )


def _make_rl_result(*, reward, f1, params=None, stage_used="validation_stage1"):
    return {
        "validation_reward": {"reward": reward, "f1": f1},
        "stage_used": stage_used,
        "params": deepcopy(params) if params is not None else {},
    }


def _make_rl_history_payload(
    *,
    round_no,
    score,
    reward,
    threshold,
    selection_metric="reward",
    stage_used="validation_stage1",
    train_metrics=None,
    validation_metrics=None,
    params=None,
    **extra,
):
    payload_params = deepcopy(params) if params is not None else _make_fusion_params(threshold=threshold)
    payload_validation_metrics = validation_metrics if validation_metrics is not None else {"reward": reward}
    payload_train_metrics = train_metrics if train_metrics is not None else {}
    payload = {
        "schema_version": 2,
        "round": round_no,
        "selection_metric": selection_metric,
        "score": score,
        "reward": reward,
        "stage_used": stage_used,
        "params": payload_params,
        "train_metrics": payload_train_metrics,
        "validation_metrics": payload_validation_metrics,
    }
    payload.update(extra)
    return payload


def _assert_path_suffix(path: str, suffix: str) -> None:
    normalized_path = path.replace("\\", "/")
    normalized_suffix = suffix.replace("\\", "/")
    assert normalized_path.endswith(normalized_suffix)


def _assert_round_history_schema_consistency(summary_entry, metrics_entry, *, expected):
    assert summary_entry["schema_version"] == 2
    assert metrics_entry["schema_version"] == 2
    assert summary_entry["stage_used"] == metrics_entry["stage_used"] == expected["stage_used"]
    assert summary_entry["params"]["threshold"] == metrics_entry["params"]["threshold"] == expected["threshold"]
    assert summary_entry["checkpoint_path"] == metrics_entry["checkpoint_path"] == expected["checkpoint_path"]
    assert summary_entry["trained"] == metrics_entry["trained"] == expected["trained"]
    _assert_path_suffix(summary_entry["feedback_path"], expected["feature_feedback_suffix"])
    _assert_path_suffix(metrics_entry["feedback_path"], expected["feature_feedback_suffix"])
    assert summary_entry["feature_meta"] == metrics_entry["feature_meta"] == expected["feature_meta"]
    assert summary_entry["artifacts"] == metrics_entry["artifacts"]
    _assert_path_suffix(summary_entry["artifacts"]["fusion_selected"], expected["fusion_selected_suffix"])
    _assert_path_suffix(
        summary_entry["artifacts"]["rl_payload"],
        expected.get("rl_payload_suffix", expected["fusion_selected_suffix"]),
    )
    _assert_path_suffix(summary_entry["artifacts"]["feature_feedback"], expected["feature_feedback_suffix"])
    _assert_path_suffix(summary_entry["artifacts"]["metrics_val"], expected["metrics_val_suffix"])
    _assert_path_suffix(summary_entry["artifacts"]["prob_dl"], expected["prob_dl_suffix"])
    _assert_path_suffix(summary_entry["artifacts"]["unc_dl"], expected["unc_dl_suffix"])
    _assert_path_suffix(summary_entry["artifacts"]["prob_fused"], expected["prob_fused_suffix"])
    assert summary_entry["selection_metric"] == metrics_entry["selection_metric"] == expected["selection_metric"]
    assert summary_entry["score"] == metrics_entry["score"] == expected["score"]
    assert summary_entry["reward"] == metrics_entry["reward"] == expected["reward"]


def _assert_rl_history_entry(entry, *, expected):
    assert entry["round"] == expected["round"]
    assert entry["selection_metric"] == expected["selection_metric"]
    assert entry["score"] == pytest.approx(expected["score"])
    assert entry["reward"] == pytest.approx(expected["reward"])
    if "stage_used" in expected:
        assert entry["stage_used"] == expected["stage_used"]
    if "threshold" in expected:
        assert entry["params"]["threshold"] == pytest.approx(expected["threshold"])
    if "validation_f1" in expected:
        assert entry["validation_metrics"]["f1"] == pytest.approx(expected["validation_f1"])


def _assert_failed_closed_loop_summary(summary, *, expected):
    assert summary["schema_version"] == 2
    assert summary["status"] == "failed"
    assert summary["failure"]["stage"] == expected["stage"]
    assert summary["failure"]["round"] == expected["round"]
    assert summary["failure"]["error_type"] == expected["error_type"]
    message = str(summary["failure"]["message"])
    if "message" in expected:
        assert message == expected["message"]
    if "message_contains" in expected:
        assert expected["message_contains"] in message
    assert summary["loop"]["rounds_requested"] == expected["rounds_requested"]
    assert summary["loop"]["rounds_completed"] == expected["rounds_completed"]
    assert summary["loop"]["stopped_early"] is expected["stopped_early"]
    if "selection_metric" in expected:
        assert summary["loop"]["selection_metric"] == expected["selection_metric"]
        assert summary["best"]["selection_metric"] == expected["selection_metric"]
    if "patience" in expected:
        assert summary["loop"]["patience"] == expected["patience"]
    if "min_delta" in expected:
        expected_min_delta = expected["min_delta"]
        if expected_min_delta is None:
            assert summary["loop"]["min_delta"] is None
        else:
            assert summary["loop"]["min_delta"] == pytest.approx(expected_min_delta)
    if expected.get("history_empty"):
        assert summary["history"] == []
    if "history_len" in expected:
        assert len(summary["history"]) == expected["history_len"]
    if "restore_outcome" in expected:
        assert summary["best"]["restore_outcome"] == expected["restore_outcome"]
    if expected.get("best_empty"):
        assert summary["best"]["round"] is None
        assert summary["best"]["score"] is None
        assert summary["best"]["reward"] is None
        assert summary["best"]["entry"] is None


def _assert_absent_rl_fusion_outputs(work_dir: Path) -> None:
    assert not (work_dir / "fusion_selected.json").exists()
    assert not (work_dir / "feature_feedback.json").exists()
    assert not (work_dir / "prob_fused.tif").exists()


def _assert_postprocess_params(params: dict) -> None:
    assert params["min_area_m2"] == 200.0
    assert params["morph_kernel"] == 3
    assert params["shadow_penalty"] == 0.5


def _assert_report_contract(report: dict, *, status: str, can_run: bool = True) -> None:
    assert report["schema_version"] == 2
    assert report["status"] == status
    assert report["can_run"] is can_run


def _make_preflight_stage_report(stage: int) -> dict:
    return {"status": "ok", "config": {"fusion": {"stage": stage}}}


def _stub_closed_loop_prereqs(monkeypatch, *, preflight=None, build_feature_stack=None) -> None:
    monkeypatch.setattr(
        "forestseg.cli.cmd_preflight_check",
        preflight or (lambda current_cfg: {"status": "ok"}),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_prepare_input",
        lambda current_cfg: None,
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_build_spec_tex",
        lambda current_cfg: None,
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_prepare_label_points",
        lambda current_cfg: None,
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_build_feature_stack",
        build_feature_stack or (lambda current_cfg: None),
    )


def _make_partial_rl_history_entry(
    *,
    selection_metric="f1",
    threshold=0.5,
    validation_metrics=None,
    score=0.1,
    reward=None,
    params=None,
    train_metrics=None,
):
    entry = {
        "params": deepcopy(params) if params is not None else _make_fusion_params(threshold=threshold, lambda_spec=0.1),
        "validation_metrics": validation_metrics if validation_metrics is not None else {"reward": 0.1, "f1": 0.1},
    }
    if selection_metric is not None:
        entry["selection_metric"] = selection_metric
    if score is not None:
        entry["score"] = score
    if reward is not None:
        entry["reward"] = reward
    if train_metrics is not None:
        entry["train_metrics"] = train_metrics
    return entry


def _stub_rl_history_fail_fast_before_lazy_setup(monkeypatch) -> None:
    monkeypatch.setattr(
        "forestseg.cli.cmd_prepare_input",
        lambda current_cfg: (_ for _ in ()).throw(AssertionError("prepare_input should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_build_spec_tex",
        lambda current_cfg: (_ for _ in ()).throw(AssertionError("build_spec_tex should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: (_ for _ in ()).throw(AssertionError("train_or_load_dl should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_prepare_label_points",
        lambda current_cfg: (_ for _ in ()).throw(AssertionError("prepare_label_points should not run")),
    )


def _assert_invalid_rl_history_rejected(
    monkeypatch,
    cfg: dict,
    work_dir: Path,
    history_text: str,
    *,
    match: str = "rl_history",
    preserve_history: bool = False,
    assert_outputs_absent: bool = False,
) -> None:
    _write_basic_rl_fusion_inputs(work_dir)
    history_path = work_dir / "rl_history.json"
    history_path.write_text(history_text, encoding="utf-8")
    _stub_rl_fusion_validation_runtime(monkeypatch)

    with pytest.raises(ValueError, match=match):
        cmd_run_rl_fusion(cfg)

    if preserve_history:
        assert history_path.read_text(encoding="utf-8") == history_text
    if assert_outputs_absent:
        _assert_absent_rl_fusion_outputs(work_dir)


def _write_basic_rl_fusion_inputs(work_dir: Path, *, include_prob_dl: bool = True) -> None:
    keys = ["prob_spec", "prob_tex", "tex_complexity"]
    if include_prob_dl:
        keys.insert(2, "prob_dl")
    for key in keys:
        (work_dir / f"{key}.tif").write_text("placeholder", encoding="utf-8")
    (work_dir / "samples_val.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")


def _stub_rl_fusion_runtime(
    monkeypatch,
    *,
    preview=None,
    stub_prepare_input: bool = False,
    stub_save_metrics: bool = False,
) -> None:
    if stub_prepare_input:
        monkeypatch.setattr(
            "forestseg.cli.cmd_prepare_input",
            lambda current_cfg: None,
        )
    fake_preview = preview or type("R", (), {"arr": np.zeros((2, 2), dtype=float)})()
    monkeypatch.setattr(
        "forestseg.cli.read_downsampled_band1",
        lambda *args, **kwargs: fake_preview,
    )
    monkeypatch.setattr(
        "forestseg.cli.build_state",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "forestseg.cli.process_aligned_inputs_to_output",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "forestseg.cli.read_band1",
        lambda path: type("Raster", (), {"profile": {"transform": None}})(),
    )
    if stub_save_metrics:
        monkeypatch.setattr(
            "forestseg.cli.save_metrics",
            lambda *args, **kwargs: None,
        )


def _stub_rl_fusion_validation_runtime(monkeypatch, *, preview=None) -> None:
    _stub_rl_fusion_runtime(monkeypatch, preview=preview)
    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("choose_fusion_by_validation should not run")),
    )


def _stub_rl_fusion_fail_fast_runtime(monkeypatch, *, guard_infer_to_files: bool = False) -> None:
    monkeypatch.setattr(
        "forestseg.cli.read_downsampled_band1",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("read_downsampled_band1 should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.build_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("build_state should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.process_aligned_inputs_to_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("process_aligned_inputs_to_output should not run")
        ),
    )
    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("choose_fusion_by_validation should not run")),
    )
    if guard_infer_to_files:
        monkeypatch.setattr(
            "forestseg.cli.infer_to_files",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("infer_to_files should not run")),
        )


def _stub_label_validation_without_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        "forestseg.cli.cmd_prepare_input",
        lambda current_cfg: None,
    )
    monkeypatch.setattr(
        "forestseg.cli.read_band1",
        lambda path: type(
            "Raster",
            (),
            {"profile": {"crs": None, "transform": None}},
        )(),
    )


def _make_scene_sh(tmp_path):
    scene_sh = tmp_path / "scene.sh"
    scene_sh.write_text(
        'SCENE_ID="x"\nCROPPED_RASTER="x"\nGENERATED_EXPORT_DIR="x"\n',
        encoding="utf-8",
    )
    return scene_sh


def _stub_scene_input(
    monkeypatch,
    input_tif: str = "x.tif",
    *,
    generated_export_dir: str = "x",
    scene_id: str = "x",
    cropped_raster: str = "x",
    capture_calls: dict | None = None,
):
    def fake_resolve_scene_input(scene_sh, prefer_v1=True, fallback_quick=True):
        if capture_calls is not None:
            capture_calls["scene_sh"] = scene_sh
            capture_calls["prefer_v1"] = prefer_v1
            capture_calls["fallback_quick"] = fallback_quick
        return type(
            "Scene",
            (),
            {
                "scene_id": scene_id,
                "cropped_raster": cropped_raster,
                "generated_export_dir": generated_export_dir,
                "input_tif": input_tif,
            },
        )()

    monkeypatch.setattr(
        "forestseg.cli.resolve_scene_input",
        fake_resolve_scene_input,
    )


def _make_prepared_workdir(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return work_dir


def _make_base_cfg(
    tmp_path,
    *,
    fusion_grid=None,
    fusion_fixed_params=None,
    dl_overrides=None,
    rl_overrides=None,
):
    positive_label_file = tmp_path / "labels_positive.gpkg"
    negative_label_file = tmp_path / "labels_negative.gpkg"
    positive_label_file.write_text("placeholder", encoding="utf-8")
    negative_label_file.write_text("placeholder", encoding="utf-8")
    dl_cfg = {
        "checkpoint_path": str(tmp_path / "work" / "checkpoints" / "best.pt"),
        "tile_size": 512,
        "stride": 384,
        "batch_size": 2,
        "epochs": 1,
        "val_threshold": 0.5,
        "selection_metric": "f1",
    }
    if dl_overrides:
        dl_cfg.update(dl_overrides)
    rl_cfg = {
        "rounds": 1,
        "patience": 2,
    }
    if rl_overrides:
        rl_cfg.update(rl_overrides)
    return {
        "work_dir": str(tmp_path / "work"),
        "scene_sh": str(_make_scene_sh(tmp_path)),
        "labels": {
            "positive_path": str(positive_label_file),
            "negative_path": str(negative_label_file),
            "split_ratio": 0.7,
            "grid_size": 1000.0,
        },
        "dl": dl_cfg,
        "fusion": {
            "grid": fusion_grid or deepcopy(REQUIRED_FUSION_GRID),
            "fixed_params": fusion_fixed_params
            or {
                "lambda_spec": 0.2,
                "lambda_tex": 0.2,
                "threshold": 0.5,
                "min_area_m2": 200.0,
                "morph_kernel": 3,
                "shadow_penalty": 0.5,
            },
        },
        "rl_loop": rl_cfg,
        "dem": {
            "path": None,
        },
    }


def test_ensure_directory_writable_ignores_probe_cleanup_failure(monkeypatch, tmp_path):
    target_dir = tmp_path / "probe-dir"
    target_dir.mkdir()

    probe_path = target_dir / ".preflight-write-test.tmp"

    class DummyProbe:
        def __init__(self, name):
            self.name = str(name)

        def __enter__(self):
            Path(self.name).write_text(
                "probe",
                encoding="utf-8",
            )
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "forestseg.cli.tempfile.NamedTemporaryFile",
        lambda *args, **kwargs: DummyProbe(probe_path),
    )
    monkeypatch.setattr(
        "forestseg.cli.os.remove",
        lambda path: (_ for _ in ()).throw(OSError("cleanup blocked")),
    )

    from forestseg.cli import _ensure_directory_writable

    assert _ensure_directory_writable(
        str(target_dir),
        "test.dir",
    ) == str(target_dir)


def test_cmd_preflight_check_raises_for_unwritable_checkpoint_dir(monkeypatch, tmp_path):
    cfg = _make_base_cfg(tmp_path)
    checkpoint_dir = os.path.dirname(cfg["dl"]["checkpoint_path"])

    from forestseg.cli import _ensure_directory_writable as original_ensure_directory_writable

    def fake_ensure_directory_writable(path, label):
        if path == checkpoint_dir:
            raise PermissionError("blocked checkpoint dir")
        return original_ensure_directory_writable(path, label)

    monkeypatch.setattr(
        "forestseg.cli._ensure_directory_writable",
        fake_ensure_directory_writable,
    )

    with pytest.raises(PermissionError, match="blocked checkpoint dir"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    "bad_ratio",
    [float("nan"), float("inf")],
)
def test_cmd_preflight_check_rejects_non_finite_split_ratio(
    tmp_path,
    bad_ratio,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["labels"]["split_ratio"] = bad_ratio
    with pytest.raises(ValueError, match="labels.split_ratio"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    "bad_grid_size",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_cmd_preflight_check_rejects_invalid_grid_size(
    tmp_path,
    bad_grid_size,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["labels"]["grid_size"] = bad_grid_size
    with pytest.raises(ValueError, match="labels.grid_size"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    "bad_split_seed",
    [-1, -2.5, True, None, "oops"],
)
def test_cmd_preflight_check_rejects_invalid_split_seed(
    tmp_path,
    bad_split_seed,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["labels"]["split_seed"] = bad_split_seed
    with pytest.raises(ValueError, match="labels.split_seed"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    ("override_key", "bad_value"),
    [
        ("batch_size", 0),
        ("epochs", 0),
        ("mc_dropout_passes", 0),
        ("sample_tile_size", 0),
        ("max_patches", 0),
        ("batch_size", -1),
        ("epochs", -1),
        ("mc_dropout_passes", -1),
        ("sample_tile_size", -1),
        ("max_patches", -1),
        ("batch_size", 1.5),
        ("epochs", 1.5),
        ("mc_dropout_passes", 1.5),
        ("sample_tile_size", 1.5),
        ("max_patches", 1.5),
        ("batch_size", True),
        ("epochs", True),
        ("mc_dropout_passes", True),
        ("sample_tile_size", True),
        ("max_patches", True),
        ("batch_size", None),
        ("epochs", None),
        ("mc_dropout_passes", None),
        ("sample_tile_size", None),
        ("max_patches", None),
        ("batch_size", "oops"),
        ("epochs", "oops"),
        ("mc_dropout_passes", "oops"),
        ("sample_tile_size", "oops"),
        ("max_patches", "oops"),
    ],
)
def test_cmd_preflight_check_rejects_invalid_dl_positive_ints(
    tmp_path,
    override_key,
    bad_value,
):
    cfg = _make_base_cfg(tmp_path, dl_overrides={override_key: bad_value})
    with pytest.raises(ValueError, match=rf"dl\.{override_key}"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    "bad_aux_warmup_epochs",
    [-1, -2.5, True, None, "oops"],
)
def test_cmd_preflight_check_rejects_invalid_aux_warmup_epochs(
    tmp_path,
    bad_aux_warmup_epochs,
):
    cfg = _make_base_cfg(tmp_path, dl_overrides={"aux_warmup_epochs": bad_aux_warmup_epochs})
    with pytest.raises(ValueError, match="dl.aux_warmup_epochs"):
        cmd_preflight_check(cfg)


def test_cmd_preflight_check_rejects_stride_greater_than_tile_size(
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path, dl_overrides={"tile_size": 128, "stride": 256})
    with pytest.raises(ValueError, match=r"dl\.stride"):
        cmd_preflight_check(cfg)


def test_cmd_preflight_check_resolves_scene_input(
    monkeypatch,
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path)
    captured = {}

    _stub_scene_input(monkeypatch, capture_calls=captured)

    report = cmd_preflight_check(cfg)

    assert captured["scene_sh"] == cfg["scene_sh"]
    assert captured["prefer_v1"] is True
    assert captured["fallback_quick"] is True
    assert report["checks"]["input_tif"] == "x.tif"


def test_cmd_preflight_check_uses_work_dir_for_bare_checkpoint_filename(
    monkeypatch,
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path, dl_overrides={"checkpoint_path": "best.pt"})
    _stub_scene_input(monkeypatch)

    report = cmd_preflight_check(cfg)

    assert report["checks"]["checkpoint_dir"] == str(tmp_path / "work")


def test_cmd_preflight_check_rejects_unwritable_generated_export_dir(
    monkeypatch,
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path)
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    _stub_scene_input(monkeypatch, generated_export_dir=str(export_dir))

    real_named_temporary_file = tempfile.NamedTemporaryFile

    def fake_named_temporary_file(*args, **kwargs):
        if kwargs.get("dir") == str(export_dir):
            raise OSError("blocked export dir")
        return real_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(
        "forestseg.cli.tempfile.NamedTemporaryFile",
        fake_named_temporary_file,
    )

    with pytest.raises(PermissionError, match=r"scene\.generated_export_dir"):
        cmd_preflight_check(cfg)


def test_cmd_prepare_input_normalizes_boolean_flags(
    monkeypatch,
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["input"] = {"prefer_v1": "false", "fallback_quick": "true"}
    captured = {}

    _stub_scene_input(monkeypatch, capture_calls=captured)
    monkeypatch.setattr(
        "forestseg.cli.normalize_raster_to_file",
        lambda *args, **kwargs: {"ok": True},
    )

    cmd_prepare_input(cfg)

    assert captured["prefer_v1"] is False
    assert captured["fallback_quick"] is True


@pytest.mark.parametrize(
    "field_name",
    ["prefer_v1", "fallback_quick"],
)
def test_cmd_prepare_input_rejects_invalid_boolean_flag(
    monkeypatch,
    tmp_path,
    field_name,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["input"] = {field_name: "maybe"}

    monkeypatch.setattr("forestseg.cli.resolve_scene_input", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match=rf"input\.{field_name}"):
        cmd_prepare_input(cfg)


@pytest.mark.parametrize(
    "field_name",
    ["prefer_v1", "fallback_quick"],
)
def test_cmd_preflight_check_rejects_invalid_input_boolean_flag(
    tmp_path,
    field_name,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["input"] = {field_name: "maybe"}

    with pytest.raises(ValueError, match=rf"input\.{field_name}"):
        cmd_preflight_check(cfg)


def test_resolve_scene_input_respects_disabled_fallback(
    monkeypatch,
    tmp_path,
):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "x_utm_v1.tif").write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "forestseg.scene_runtime._source_scene",
        lambda scene_sh: {
            "SCENE_ID": "x",
            "CROPPED_RASTER": "x",
            "GENERATED_EXPORT_DIR": str(export_dir),
        },
    )

    with pytest.raises(FileNotFoundError, match="Checked"):
        resolve_scene_input(str(tmp_path / "scene.sh"), prefer_v1=False, fallback_quick=False)


def test_parse_scene_file_fallback_prefers_explicit_export_dir(
    tmp_path,
):
    explicit_export_dir = tmp_path / "custom_export"
    scene_sh = tmp_path / "scene.sh"
    scene_sh.write_text(
        f'SCENE_ID="x"\nCROPPED_RASTER="x"\nGENERATED_EXPORT_DIR="{explicit_export_dir}"\n',
        encoding="utf-8",
    )

    parsed = _parse_scene_file_fallback(str(scene_sh))

    assert Path(parsed["GENERATED_EXPORT_DIR"]) == explicit_export_dir


def test_source_scene_raises_on_bash_source_failure(
    monkeypatch,
    tmp_path,
):
    scene_sh = tmp_path / "scene.sh"
    scene_sh.write_text('SCENE_ID="x"\n', encoding="utf-8")

    def fake_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output="syntax error")

    monkeypatch.setattr("forestseg.scene_runtime.subprocess.check_output", fake_check_output)

    with pytest.raises(RuntimeError, match="scene.sh"):
        _source_scene(str(scene_sh))


def test_resolve_scene_input_raises_on_source_failure(
    monkeypatch,
    tmp_path,
):
    scene_sh = tmp_path / "scene.sh"
    scene_sh.write_text('SCENE_ID="x"\n', encoding="utf-8")

    def fake_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output="syntax error")

    monkeypatch.setattr("forestseg.scene_runtime.subprocess.check_output", fake_check_output)
    monkeypatch.setattr(
        "forestseg.scene_runtime._parse_scene_file_fallback",
        lambda scene_path: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    with pytest.raises(RuntimeError, match="scene.sh"):
        resolve_scene_input(str(scene_sh))


def test_resolve_scene_input_static_fallback_honors_explicit_generated_export_dir(
    monkeypatch,
    tmp_path,
):
    explicit_export_dir = tmp_path / "custom_export"
    explicit_export_dir.mkdir()
    scene_sh = tmp_path / "scene.sh"
    scene_sh.write_text(
        f'SCENE_ID="x"\nCROPPED_RASTER="x"\nGENERATED_EXPORT_DIR="{explicit_export_dir}"\n',
        encoding="utf-8",
    )
    expected_input = explicit_export_dir / "x_utm_v1.tif"
    expected_input.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "forestseg.scene_runtime.subprocess.check_output",
        lambda *args, **kwargs: "SCENE_ID=x\nCROPPED_RASTER=x\n",
    )

    result = resolve_scene_input(str(scene_sh), prefer_v1=True, fallback_quick=False)

    assert Path(result.generated_export_dir) == explicit_export_dir
    assert Path(result.input_tif) == expected_input


def test_cmd_preflight_check_accepts_stage0_with_fixed_params_only(
    monkeypatch,
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["fusion"]["stage"] = 0
    cfg["fusion"]["grid"] = {}

    _stub_scene_input(monkeypatch)

    report = cmd_preflight_check(cfg)

    assert report["config"]["fusion"]["stage"] == 0
    assert report["config"]["fusion"]["grid_keys"] == []
    assert sorted(report["config"]["fusion"]["fixed_param_keys"]) == [
        "lambda_spec",
        "lambda_tex",
        "min_area_m2",
        "morph_kernel",
        "shadow_penalty",
        "threshold",
    ]


def test_cmd_run_rl_fusion_accepts_stage0_with_fixed_params_only_when_preflight_is_skipped(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    cfg["fusion"]["stage"] = 0
    cfg["fusion"]["grid"] = {}

    _write_basic_rl_fusion_inputs(work_dir)

    monkeypatch.setattr(
        "forestseg.cli.save_metrics",
        lambda *args, **kwargs: None,
    )
    captured = {}
    monkeypatch.setattr(
        "forestseg.cli.cmd_prepare_input",
        lambda current_cfg: None,
    )
    _stub_rl_fusion_validation_runtime(monkeypatch)

    def fake_choose_fusion_by_validation(**kwargs):
        captured["stage"] = kwargs["stage"]
        captured["fixed_params"] = kwargs["fixed_params"]
        captured["grid_params"] = kwargs["grid_params"]
        return (
            FusionParams(**cfg["fusion"]["fixed_params"]),
            {"reward": 0.8},
            "validation_stage0",
        )

    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        fake_choose_fusion_by_validation,
    )

    result = cmd_run_rl_fusion(cfg)

    assert captured["stage"] == 0
    assert captured["fixed_params"] == cfg["fusion"]["fixed_params"]
    assert captured["grid_params"] == {}
    assert result["stage_used"] == "validation_stage0"


def test_cmd_preflight_check_rejects_stage0_without_fixed_params(
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["fusion"]["fixed_params"] = {}
    cfg["fusion"]["stage"] = 0
    cfg["fusion"]["grid"] = {}

    with pytest.raises(ValueError, match=r"fusion\.fixed_params"):
        cmd_preflight_check(cfg)


def test_cmd_run_rl_fusion_rejects_stage0_without_fixed_params_when_preflight_is_skipped(
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["fusion"]["fixed_params"] = {}
    cfg["fusion"]["stage"] = 0
    cfg["fusion"]["grid"] = {}

    with pytest.raises(ValueError, match=r"fusion\.fixed_params"):
        cmd_run_rl_fusion(cfg)


@pytest.mark.parametrize(
    ("grid_key", "bad_values", "message"),
    [
        ("lambda_spec", [-0.1], r"fusion\.grid\.lambda_spec\[0\]"),
        ("lambda_tex", [1.1], r"fusion\.grid\.lambda_tex\[0\]"),
        ("threshold", [1.5], r"fusion\.grid\.threshold\[0\]"),
    ],
)
def test_cmd_preflight_check_rejects_invalid_fusion_grid_candidate_values(
    tmp_path,
    grid_key,
    bad_values,
    message,
):
    grid = deepcopy(REQUIRED_FUSION_GRID)
    grid[grid_key] = bad_values
    cfg = _make_base_cfg(tmp_path, fusion_grid=grid)

    with pytest.raises(ValueError, match=message):
        cmd_preflight_check(cfg)


def test_cmd_preflight_check_accepts_optional_postprocess_grid_keys(
    monkeypatch,
    tmp_path,
):
    cfg = _make_base_cfg(
        tmp_path,
        fusion_grid={**deepcopy(REQUIRED_FUSION_GRID), **deepcopy(OPTIONAL_FUSION_POSTPROCESS_GRID)},
    )

    _stub_scene_input(monkeypatch)

    report = cmd_preflight_check(cfg)

    assert sorted(report["config"]["fusion"]["grid_keys"]) == [
        "lambda_spec",
        "lambda_tex",
        "min_area_m2",
        "morph_kernel",
        "shadow_penalty",
        "threshold",
    ]


@pytest.mark.parametrize(
    ("grid_key", "bad_values", "message"),
    [
        ("min_area_m2", [0], r"fusion\.grid\.min_area_m2\[0\]"),
        ("morph_kernel", [0], r"fusion\.grid\.morph_kernel\[0\]"),
        ("shadow_penalty", [-0.2], r"fusion\.grid\.shadow_penalty\[0\]"),
    ],
)
def test_cmd_preflight_check_rejects_invalid_optional_postprocess_grid_values(
    monkeypatch,
    tmp_path,
    grid_key,
    bad_values,
    message,
):
    grid = {**deepcopy(REQUIRED_FUSION_GRID), **deepcopy(OPTIONAL_FUSION_POSTPROCESS_GRID)}
    grid[grid_key] = bad_values
    cfg = _make_base_cfg(tmp_path, fusion_grid=grid)

    _stub_scene_input(monkeypatch)

    with pytest.raises(ValueError, match=message):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    ("override_key", "bad_value"),
    [
        ("rounds", 0),
        ("patience", 0),
        ("rounds", -1),
        ("patience", -1),
        ("rounds", 1.5),
        ("patience", 1.5),
        ("rounds", True),
        ("patience", True),
    ],
)
def test_cmd_preflight_check_rejects_invalid_rl_loop_values(
    tmp_path,
    override_key,
    bad_value,
):
    cfg = _make_base_cfg(tmp_path, rl_overrides={override_key: bad_value})
    with pytest.raises(ValueError, match=rf"rl_loop\.{override_key}"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    "bad_min_delta",
    [-0.001, float("nan"), float("inf"), True, None, "oops"],
)
def test_cmd_preflight_check_rejects_invalid_rl_min_delta(
    tmp_path,
    bad_min_delta,
):
    cfg = _make_base_cfg(tmp_path, rl_overrides={"min_delta": bad_min_delta})
    with pytest.raises(ValueError, match="rl_loop.min_delta"):
        cmd_preflight_check(cfg)


def test_cmd_preflight_check_rejects_bandit_min_epsilon_greater_than_epsilon(
    tmp_path,
):
    cfg = _make_base_cfg(
        tmp_path,
        rl_overrides={
            "bandit": {
                "enabled": True,
                "epsilon": 0.1,
                "min_epsilon": 0.2,
            }
        },
    )
    with pytest.raises(ValueError, match="rl_loop.bandit.min_epsilon"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    ("bad_key", "bad_value"),
    [
        ("epsilon", 1.1),
        ("epsilon", -0.1),
        ("alpha", 1.1),
        ("epsilon_decay", -0.1),
        ("seed", 0),
        ("enabled", "oops"),
    ],
)
def test_cmd_preflight_check_rejects_invalid_bandit_config_values(
    tmp_path,
    bad_key,
    bad_value,
):
    cfg = _make_base_cfg(
        tmp_path,
        rl_overrides={
            "bandit": {
                bad_key: bad_value,
            }
        },
    )
    with pytest.raises(ValueError, match=rf"rl_loop\.bandit\.{bad_key}"):
        cmd_preflight_check(cfg)


def test_cmd_preflight_check_accepts_default_bandit_config(
    monkeypatch,
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path, rl_overrides={"min_delta": 0.0})

    _stub_scene_input(monkeypatch)

    report = cmd_preflight_check(cfg)

    assert report["config"]["rl_loop"]["min_delta"] == 0.0
    assert report["config"]["rl_loop"]["bandit"] == {
        "enabled": True,
        "epsilon": 0.2,
        "min_epsilon": 0.05,
        "epsilon_decay": 0.95,
        "alpha": 0.3,
        "seed": 42,
    }


@pytest.mark.parametrize(
    "ok_threshold",
    [0.0, 1.0],
)
def test_cmd_preflight_check_accepts_boundary_val_threshold(
    monkeypatch,
    tmp_path,
    ok_threshold,
):
    cfg = _make_base_cfg(tmp_path, dl_overrides={"val_threshold": ok_threshold})

    _stub_scene_input(monkeypatch)

    report = cmd_preflight_check(cfg)

    assert report["config"]["dl"]["val_threshold"] == ok_threshold


@pytest.mark.parametrize(
    "bad_stage",
    [None, "oops", True, False, -1, "-1"],
)
def test_cmd_preflight_check_rejects_non_integer_fusion_stage(
    tmp_path,
    bad_stage,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["fusion"]["stage"] = bad_stage

    with pytest.raises((TypeError, ValueError), match="stage|int|integer|0 或 1"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    "bad_stage",
    [1.5, -1.5, "1.5", 2, "2"],
)
def test_cmd_preflight_check_rejects_fractional_fusion_stage(
    tmp_path,
    bad_stage,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["fusion"]["stage"] = bad_stage

    with pytest.raises((TypeError, ValueError), match="stage|0 或 1"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    "grid_key",
    ["lambda_spec", "lambda_tex", "threshold"],
)
def test_cmd_preflight_check_rejects_positive_stage_without_required_grid(
    monkeypatch,
    tmp_path,
    grid_key,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["fusion"]["stage"] = 1
    cfg["fusion"]["grid"] = {key: value for key, value in REQUIRED_FUSION_GRID.items() if key != grid_key}

    with pytest.raises(ValueError, match="fusion.grid"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    "grid_key",
    ["lambda_spec", "lambda_tex", "threshold"],
)
def test_cmd_run_rl_fusion_rejects_positive_stage_without_required_grid_when_preflight_is_skipped(
    tmp_path,
    grid_key,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["fusion"]["stage"] = 1
    cfg["fusion"]["grid"] = {key: value for key, value in REQUIRED_FUSION_GRID.items() if key != grid_key}

    with pytest.raises(ValueError, match="fusion.grid"):
        cmd_run_rl_fusion(cfg)


@pytest.mark.parametrize(
    "bad_stage",
    [None, "oops", True, False, -1, "-1"],
)
def test_cmd_run_rl_fusion_rejects_non_integer_fusion_stage_before_lazy_setup(
    tmp_path,
    bad_stage,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["fusion"]["stage"] = bad_stage

    with pytest.raises((TypeError, ValueError), match="stage|int|integer|0 或 1"):
        cmd_run_rl_fusion(cfg)


@pytest.mark.parametrize(
    "bad_stage",
    [1.5, -1.5, "1.5", 2, "2"],
)
def test_cmd_run_rl_fusion_rejects_fractional_fusion_stage_before_lazy_setup(
    tmp_path,
    bad_stage,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["fusion"]["stage"] = bad_stage

    with pytest.raises((TypeError, ValueError), match="stage|0 或 1"):
        cmd_run_rl_fusion(cfg)


@pytest.mark.parametrize(
    "bad_threshold",
    [-0.1, 1.1, float("nan"), float("inf"), True, None, "oops"],
)
def test_cmd_train_or_load_dl_rejects_invalid_val_threshold_before_training(
    monkeypatch,
    tmp_path,
    bad_threshold,
):
    work_dir = _make_prepared_workdir(tmp_path)
    (work_dir / "feature_stack.tif").write_text("placeholder", encoding="utf-8")
    (work_dir / "samples_train.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")
    (work_dir / "samples_val.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")

    monkeypatch.setattr(
        "forestseg.cli.load_points_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_points_json should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.train_supervised_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("train_supervised_model should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.infer_to_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("infer_to_files should not run")),
    )

    cfg = _make_base_cfg(tmp_path, dl_overrides={"val_threshold": bad_threshold})
    cfg["work_dir"] = str(work_dir)

    with pytest.raises(ValueError, match="dl.val_threshold"):
        cmd_train_or_load_dl(cfg, force_train=True)


@pytest.mark.parametrize(
    "ok_threshold",
    [0.0, 1.0],
)
def test_cmd_train_or_load_dl_accepts_boundary_val_threshold_when_training(
    monkeypatch,
    tmp_path,
    ok_threshold,
):
    work_dir = _make_prepared_workdir(tmp_path)
    (work_dir / "feature_stack.tif").write_text("placeholder", encoding="utf-8")
    (work_dir / "samples_train.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")
    (work_dir / "samples_val.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")

    captured = {}

    monkeypatch.setattr("forestseg.cli.load_points_json", lambda *args, **kwargs: ([], {}))

    def fake_train_supervised_model(*args, **kwargs):
        captured["val_threshold"] = kwargs["cfg"]["val_threshold"]
        return type(
            "TrainResult",
            (),
            {
                "checkpoint_path": str(work_dir / "checkpoints" / "best.pt"),
                "best_score": 0.8,
                "metrics": {"f1": 0.8},
            },
        )()

    monkeypatch.setattr("forestseg.cli.train_supervised_model", fake_train_supervised_model)
    monkeypatch.setattr("forestseg.cli.save_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "forestseg.cli.infer_to_files",
        lambda **kwargs: None,
    )

    cfg = _make_base_cfg(tmp_path, dl_overrides={"val_threshold": ok_threshold})
    cfg["work_dir"] = str(work_dir)

    result = cmd_train_or_load_dl(cfg, force_train=True)

    assert captured["val_threshold"] == ok_threshold
    assert result["trained"] is True


def test_cmd_preflight_check_rejects_false_val_threshold(tmp_path):
    cfg = _make_base_cfg(tmp_path, dl_overrides={"val_threshold": False})

    with pytest.raises(ValueError, match="dl.val_threshold"):
        cmd_preflight_check(cfg)


def test_cmd_run_rl_fusion_rejects_invalid_fusion_grid_candidates_when_preflight_is_skipped(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    grid = deepcopy(REQUIRED_FUSION_GRID)
    grid["threshold"] = [1.5]
    cfg = _make_base_cfg(tmp_path, fusion_grid=grid)
    cfg["work_dir"] = str(work_dir)

    for key in ("prob_spec", "prob_tex", "prob_dl", "tex_complexity"):
        (work_dir / f"{key}.tif").write_text("placeholder", encoding="utf-8")
    (work_dir / "samples_val.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")

    _stub_rl_fusion_fail_fast_runtime(monkeypatch)

    with pytest.raises(ValueError, match=r"fusion\.grid\.threshold\[0\]"):
        cmd_run_rl_fusion(cfg)


def test_cmd_run_rl_fusion_allows_invalid_val_threshold_with_existing_artifacts(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, dl_overrides={"val_threshold": "oops"})
    cfg["work_dir"] = str(work_dir)

    _write_basic_rl_fusion_inputs(work_dir)
    monkeypatch.setattr("forestseg.cli.save_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cmd_train_or_load_dl should not run")),
    )
    monkeypatch.setattr("forestseg.cli.cmd_prepare_input", lambda current_cfg: None)
    _stub_rl_fusion_validation_runtime(monkeypatch)

    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        lambda **kwargs: (
            FusionParams(**_make_fusion_params(threshold=0.6)),
            {"reward": 0.1, "f1": 0.8},
            "validation_stage1",
        ),
    )

    result = cmd_run_rl_fusion(cfg)

    assert result["score"] == 0.1


def test_cmd_run_rl_fusion_allows_invalid_val_threshold_when_lazy_build_uses_existing_checkpoint(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, dl_overrides={"val_threshold": "oops"})
    cfg["work_dir"] = str(work_dir)

    for key in ("input", "prob_spec", "prob_tex", "tex_complexity"):
        (work_dir / f"{key}.tif").write_text("placeholder", encoding="utf-8")
    (work_dir / "feature_stack.tif").write_text("placeholder", encoding="utf-8")
    (work_dir / "samples_train.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")
    (work_dir / "samples_val.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")
    checkpoint_path = work_dir / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        "forestseg.cli.cmd_prepare_input",
        lambda current_cfg: None,
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_build_spec_tex",
        lambda current_cfg: None,
    )
    monkeypatch.setattr(
        "forestseg.cli.read_downsampled_band1",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("read_downsampled_band1 should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.build_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("build_state should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.process_aligned_inputs_to_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("process_aligned_inputs_to_output should not run")
        ),
    )
    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("choose_fusion_by_validation should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.infer_to_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("infer_to_files should not run")),
    )

    with pytest.raises(AssertionError, match="infer_to_files should not run"):
        cmd_run_rl_fusion(cfg)


def test_cmd_run_rl_fusion_rejects_malformed_rl_history_before_lazy_setup(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    original_history = '{"broken": '
    (work_dir / "rl_history.json").write_text(original_history, encoding="utf-8")

    _stub_rl_history_fail_fast_before_lazy_setup(monkeypatch)

    with pytest.raises(ValueError, match="rl_history"):
        cmd_run_rl_fusion(cfg)

    assert (work_dir / "rl_history.json").read_text(encoding="utf-8") == original_history
    _assert_absent_rl_fusion_outputs(work_dir)


def test_cmd_run_rl_fusion_rejects_invalid_rl_history_entries(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _assert_invalid_rl_history_rejected(
        monkeypatch,
        cfg,
        work_dir,
        '["oops"]',
        preserve_history=True,
        assert_outputs_absent=True,
    )


def test_cmd_run_rl_fusion_rejects_dict_rl_history_before_lazy_setup(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    original_history = json.dumps(
        _make_partial_rl_history_entry(),
    )
    (work_dir / "rl_history.json").write_text(original_history, encoding="utf-8")

    _stub_rl_history_fail_fast_before_lazy_setup(monkeypatch)

    with pytest.raises(ValueError, match="Invalid rl_history payload"):
        cmd_run_rl_fusion(cfg)

    assert (work_dir / "rl_history.json").read_text(encoding="utf-8") == original_history
    _assert_absent_rl_fusion_outputs(work_dir)


def test_cmd_run_rl_fusion_rejects_empty_dict_rl_history_list_entry(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _assert_invalid_rl_history_rejected(
        monkeypatch,
        cfg,
        work_dir,
        "[{}]",
        preserve_history=True,
        assert_outputs_absent=True,
    )


def test_cmd_run_rl_fusion_normalizes_partial_rl_history_entry_missing_selected_metric(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _write_basic_rl_fusion_inputs(work_dir)
    history_path = work_dir / "rl_history.json"
    history_path.write_text(
        json.dumps(
            [
                _make_partial_rl_history_entry(
                    validation_metrics={"reward": 0.1},
                    score=None,
                )
            ]
        ),
        encoding="utf-8",
    )

    _stub_rl_fusion_runtime(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        lambda **kwargs: _make_fixed_validation_selection(),
    )

    result = cmd_run_rl_fusion(cfg)

    assert result["validation_reward"]["reward"] == pytest.approx(0.7)
    normalized_history = json.loads(history_path.read_text(encoding="utf-8"))
    assert normalized_history[0]["selection_metric"] == "reward"
    assert normalized_history[0]["params"]["threshold"] == pytest.approx(0.5)


def test_cmd_run_rl_fusion_normalizes_downgraded_selection_metric_with_legacy_score(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _write_basic_rl_fusion_inputs(work_dir)
    history_path = work_dir / "rl_history.json"
    history_path.write_text(
        json.dumps(
            [
                _make_partial_rl_history_entry(
                    selection_metric="f1",
                    validation_metrics={"reward": 0.1},
                    score=0.9,
                )
            ]
        ),
        encoding="utf-8",
    )

    _stub_rl_fusion_runtime(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        lambda **kwargs: _make_fixed_validation_selection(),
    )

    result = cmd_run_rl_fusion(cfg)

    assert result["validation_reward"]["reward"] == pytest.approx(0.7)
    normalized_history = json.loads(history_path.read_text(encoding="utf-8"))
    assert normalized_history[0]["selection_metric"] == "reward"
    assert normalized_history[0]["score"] == pytest.approx(0.1)
    assert normalized_history[0]["validation_metrics"]["reward"] == pytest.approx(0.1)


def test_cmd_run_rl_fusion_rejects_partial_rl_history_entry_with_invalid_params(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _assert_invalid_rl_history_rejected(
        monkeypatch,
        cfg,
        work_dir,
        json.dumps(
            [
                _make_partial_rl_history_entry(
                    params=_make_fusion_params(threshold="oops", lambda_spec=0.1),
                )
            ]
        ),
        assert_outputs_absent=True,
    )


def test_cmd_run_rl_fusion_rejects_partial_rl_history_entry_with_mismatched_score(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _assert_invalid_rl_history_rejected(
        monkeypatch,
        cfg,
        work_dir,
        json.dumps(
            [
                _make_partial_rl_history_entry(
                    validation_metrics={"reward": 0.1, "f1": 0.2},
                    score=0.3,
                )
            ]
        ),
    )


def test_cmd_run_rl_fusion_rejects_partial_rl_history_entry_with_mismatched_reward(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _assert_invalid_rl_history_rejected(
        monkeypatch,
        cfg,
        work_dir,
        json.dumps(
            [
                _make_partial_rl_history_entry(
                    selection_metric="reward",
                    validation_metrics={"reward": 0.1},
                    score=0.1,
                    reward=0.2,
                )
            ]
        ),
    )


def test_cmd_run_rl_fusion_normalizes_accepted_rl_history_entries(
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    history_path = work_dir / "rl_history.json"
    history_path.write_text(
        json.dumps(
            [
                {
                    "selection_metric": " F1 ",
                    "reward": "0.1",
                    "round": "1",
                    "score": "0.2",
                    "params": {
                        "lambda_spec": "0.1",
                        "lambda_tex": "0.2",
                        "threshold": "0.5",
                        "min_area_m2": "200.0",
                        "morph_kernel": "3",
                        "shadow_penalty": "0.5",
                    },
                    "validation_metrics": {"reward": "0.1", "f1": "0.2"},
                    "train_metrics": {},
                }
            ]
        ),
        encoding="utf-8",
    )

    normalized_history = _load_rl_history(str(history_path))

    assert normalized_history[0]["selection_metric"] == "f1"
    assert normalized_history[0]["reward"] == 0.1
    assert normalized_history[0]["round"] == 1
    assert normalized_history[0]["score"] == 0.2
    assert normalized_history[0]["params"]["threshold"] == 0.5
    assert normalized_history[0]["params"]["morph_kernel"] == 3
    assert normalized_history[0]["validation_metrics"]["reward"] == 0.1
    assert normalized_history[0]["validation_metrics"]["f1"] == 0.2


def test_cmd_run_rl_fusion_rejects_malformed_rl_history(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _write_basic_rl_fusion_inputs(work_dir)
    original_history = '{"broken": '
    (work_dir / "rl_history.json").write_text(original_history, encoding="utf-8")

    _stub_rl_fusion_runtime(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        lambda **kwargs: _make_fixed_validation_selection(),
    )

    with pytest.raises(ValueError, match="rl_history"):
        cmd_run_rl_fusion(cfg)

    assert (work_dir / "rl_history.json").read_text(encoding="utf-8") == original_history
    _assert_absent_rl_fusion_outputs(work_dir)


def test_cmd_run_rl_fusion_persists_selected_fixed_postprocess_params(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _write_basic_rl_fusion_inputs(work_dir)

    _stub_rl_fusion_runtime(monkeypatch, stub_prepare_input=True, stub_save_metrics=True)
    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        lambda **kwargs: _make_fixed_validation_selection(),
    )

    result = cmd_run_rl_fusion(cfg)
    payload = json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8"))

    _assert_postprocess_params(result["params"])
    _assert_postprocess_params(payload["params"])


def test_cmd_run_rl_fusion_real_selection_keeps_fixed_postprocess_params(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(
        tmp_path,
        fusion_grid={
            "lambda_spec": [0.1, 0.9],
            "lambda_tex": [0.0],
            "threshold": [0.5],
            "min_area_m2": [100.0, 500.0],
            "morph_kernel": [1, 7],
            "shadow_penalty": [0.1, 0.9],
        },
        fusion_fixed_params={
            "lambda_spec": 0.2,
            "lambda_tex": 0.2,
            "threshold": 0.5,
            "min_area_m2": 200.0,
            "morph_kernel": 3,
            "shadow_penalty": 0.5,
        },
    )
    cfg["work_dir"] = str(work_dir)

    _write_basic_rl_fusion_inputs(work_dir)
    (work_dir / "samples_val.json").write_text(
        json.dumps(
            {
                "points": [
                    {"label": 0},
                    {"label": 0},
                    {"label": 1},
                    {"label": 1},
                ],
                "meta": {},
            }
        ),
        encoding="utf-8",
    )

    _stub_rl_fusion_runtime(monkeypatch, stub_prepare_input=True, stub_save_metrics=True)

    sample_values = {
        str(work_dir / "prob_dl.tif"): [0.2, 0.2, 0.8, 0.8],
        str(work_dir / "prob_spec.tif"): [0.9, 0.9, 0.1, 0.1],
        str(work_dir / "prob_tex.tif"): [0.2, 0.2, 0.8, 0.8],
    }

    def fake_sample(path, points):
        return sample_values[path]

    monkeypatch.setattr("forestseg.rl_policy.sample_raster_at_points", fake_sample)

    result = cmd_run_rl_fusion(cfg)
    payload = json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8"))

    assert result["stage_used"] == "validation_stage1"
    assert result["params"]["lambda_spec"] == 0.1
    _assert_postprocess_params(result["params"])
    _assert_postprocess_params(payload["params"])


def test_cmd_run_rl_fusion_bandit_stage1_forwards_single_candidate_grid(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(
        tmp_path,
        fusion_grid={
            "lambda_spec": [0.1, 0.4],
            "lambda_tex": [0.2, 0.8],
            "threshold": [0.3, 0.6],
            "min_area_m2": [200.0],
            "morph_kernel": [3],
            "shadow_penalty": [0.5],
        },
        rl_overrides={"bandit": {"enabled": True, "epsilon": 0.0, "min_epsilon": 0.0, "seed": 7}},
    )
    cfg["work_dir"] = str(work_dir)

    _write_basic_rl_fusion_inputs(work_dir)
    _stub_rl_fusion_runtime(monkeypatch, stub_prepare_input=True, stub_save_metrics=True)

    captured = {}

    def fake_choose_fusion_by_validation(**kwargs):
        captured["grid_params"] = deepcopy(kwargs["grid_params"])
        return _make_fixed_validation_selection()

    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        fake_choose_fusion_by_validation,
    )

    cmd_run_rl_fusion(cfg)

    forwarded_grid = captured["grid_params"]
    assert len(forwarded_grid["lambda_spec"]) == 1
    assert len(forwarded_grid["lambda_tex"]) == 1
    assert len(forwarded_grid["threshold"]) == 1
    assert forwarded_grid["lambda_spec"][0] in cfg["fusion"]["grid"]["lambda_spec"]
    assert forwarded_grid["lambda_tex"][0] in cfg["fusion"]["grid"]["lambda_tex"]
    assert forwarded_grid["threshold"][0] in cfg["fusion"]["grid"]["threshold"]

    payload = json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8"))
    assert payload["policy_type"] == "bandit"
    assert payload["bandit"]["enabled"] is True
    assert payload["bandit"]["action_id"] is not None


def test_cmd_run_rl_fusion_bandit_state_updates_across_calls(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(
        tmp_path,
        rl_overrides={
            "bandit": {
                "enabled": True,
                "epsilon": 0.8,
                "min_epsilon": 0.05,
                "epsilon_decay": 0.5,
                "alpha": 0.5,
                "seed": 11,
            }
        },
    )
    cfg["work_dir"] = str(work_dir)

    _write_basic_rl_fusion_inputs(work_dir)
    _stub_rl_fusion_runtime(monkeypatch, stub_prepare_input=True, stub_save_metrics=True)

    rewards = iter([0.2, 1.0])

    def fake_choose_fusion_by_validation(**kwargs):
        reward = next(rewards)
        return (
            FusionParams(**_make_fusion_params(threshold=kwargs["grid_params"]["threshold"][0])),
            {"reward": reward, "f1": reward},
            "validation_stage1",
        )

    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        fake_choose_fusion_by_validation,
    )

    first = cmd_run_rl_fusion(cfg)
    second = cmd_run_rl_fusion(cfg)

    assert first["score"] == pytest.approx(0.2)
    assert second["score"] == pytest.approx(1.0)

    bandit_state = json.loads((work_dir / "bandit_state.json").read_text(encoding="utf-8"))
    assert bandit_state["steps"] == 2
    assert bandit_state["n"] == [2]
    assert bandit_state["q"][0] == pytest.approx(0.55)
    assert bandit_state["epsilon"] == pytest.approx(0.2)
    assert bandit_state["min_epsilon"] == pytest.approx(0.05)
    assert bandit_state["epsilon_decay"] == pytest.approx(0.5)
    assert bandit_state["action_space_signature"] == [
        {
            "lambda_spec": 0.2,
            "lambda_tex": 0.2,
            "threshold": 0.5,
        }
    ]

    payload = json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8"))
    history = json.loads((work_dir / "rl_history.json").read_text(encoding="utf-8"))
    assert payload["policy_type"] == "bandit"
    assert payload["bandit"]["enabled"] is True
    assert payload["bandit"]["epsilon_before"] == pytest.approx(0.4)
    assert payload["bandit"]["epsilon_after"] == pytest.approx(0.2)
    assert history[-1]["policy_type"] == "bandit"


def test_cmd_preflight_check_rejects_invalid_rl_selection_metric(
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path, rl_overrides={"selection_metric": "auc"})
    with pytest.raises(ValueError, match="rl_loop.selection_metric"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize(
    ("override_key", "bad_value"),
    [
        ("batch_size", True),
        ("epochs", True),
        ("max_patches", True),
        ("sample_tile_size", 0),
        ("sample_tile_size", -1),
        ("sample_tile_size", 1.5),
        ("sample_tile_size", True),
        ("aux_warmup_epochs", -1),
        ("aux_warmup_epochs", -2.5),
        ("aux_warmup_epochs", True),
        ("batch_size", None),
        ("epochs", None),
        ("max_patches", None),
        ("sample_tile_size", None),
        ("aux_warmup_epochs", None),
        ("batch_size", "oops"),
        ("epochs", "oops"),
        ("max_patches", "oops"),
        ("sample_tile_size", "oops"),
        ("aux_warmup_epochs", "oops"),
    ],
)
def test_train_supervised_model_rejects_invalid_positive_ints_before_raster_io(
    monkeypatch,
    tmp_path,
    override_key,
    bad_value,
):
    feature_stack_path = tmp_path / "feature_stack.tif"
    checkpoint_path = tmp_path / "best.pt"
    cfg = {
        "tile_size": 512,
        "batch_size": 2,
        "epochs": 1,
        "max_patches": 16,
        "val_threshold": 0.5,
        "selection_metric": "f1",
    }
    cfg[override_key] = bad_value

    monkeypatch.setattr(
        "forestseg.train.rasterio.open",
        lambda path: (_ for _ in ()).throw(AssertionError("rasterio.open should not be called")),
    )

    with pytest.raises(ValueError, match=override_key):
        train_supervised_model(
            feature_stack_path=str(feature_stack_path),
            train_points=[],
            val_points=[],
            cfg=cfg,
            checkpoint_path=str(checkpoint_path),
        )


def test_train_supervised_model_returns_json_serializable_metrics(
    monkeypatch,
    tmp_path,
):
    feature_stack_path = tmp_path / "feature_stack.tif"
    checkpoint_path = tmp_path / "best.pt"
    feature_stack_path.write_text("placeholder", encoding="utf-8")

    class DummyDataset:
        count = 2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyModel:
        def train(self):
            return None

        def eval(self):
            return None

        def parameters(self):
            return []

        def state_dict(self):
            return {"weights": 1}

        def __call__(self, x):
            return x[:, :1, :, :].clone().detach().requires_grad_(True)

    class DummyOptimizer:
        def zero_grad(self, set_to_none=True):
            return None

        def step(self):
            return None

    class DummyLoss:
        def __call__(self, logits, y, mask=None):
            return logits.mean() * 0

    saved_checkpoint = {}

    monkeypatch.setattr("forestseg.train.rasterio.open", lambda path: DummyDataset())
    monkeypatch.setattr("forestseg.train.build_model", lambda **kwargs: DummyModel())
    monkeypatch.setattr(
        "forestseg.train.to_device",
        lambda model: (model, "cpu"),
    )
    monkeypatch.setattr(
        "forestseg.train.nn.BCEWithLogitsLoss",
        lambda reduction="none": lambda logits, y: torch.ones_like(logits),
    )
    monkeypatch.setattr(
        "forestseg.train.DiceLoss",
        lambda: DummyLoss(),
    )
    monkeypatch.setattr(
        "forestseg.train.optim.AdamW",
        lambda params, lr, weight_decay: DummyOptimizer(),
    )
    monkeypatch.setattr(
        "forestseg.train.make_training_batch",
        lambda **kwargs: (
            np.zeros((1, 2, 4, 4), dtype=np.float32),
            np.zeros((1, 1, 4, 4), dtype=np.float32),
            np.ones((1, 1, 4, 4), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        "forestseg.train.evaluate_points",
        lambda **kwargs: (
            np.array([0, 1], dtype=np.uint8),
            np.array([0.1, 0.9], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        "forestseg.train.torch.save",
        lambda obj, path: saved_checkpoint.update({"payload": obj, "path": path}),
    )

    result = train_supervised_model(
        feature_stack_path=str(feature_stack_path),
        train_points=[{"id": 1}],
        val_points=[{"id": 2}],
        cfg={"batch_size": 1, "epochs": 1, "max_patches": 1, "selection_metric": "accuracy", "val_threshold": 0.5},
        checkpoint_path=str(checkpoint_path),
    )

    strict_json = json.dumps(result.metrics, ensure_ascii=False, allow_nan=False)
    decoded = json.loads(strict_json)
    assert result.metrics["selection_metric"] == "accuracy"
    assert result.metrics["history"][0]["selection_metric"] == "accuracy"
    assert saved_checkpoint["path"] == str(checkpoint_path)
    assert saved_checkpoint["payload"]["best_score"] == result.best_score
    assert "model_state_dict" in saved_checkpoint["payload"]
    assert isinstance(saved_checkpoint["payload"]["model_state_dict"], dict)
    assert isinstance(decoded["best_score"], float)
    assert isinstance(decoded["selection_metric"], str)
    assert isinstance(decoded["history"][0]["f1"], float)
    assert isinstance(decoded["history"][0]["train_loss"], float)
    assert isinstance(decoded["history"][0]["epoch"], int)
    assert isinstance(decoded["history"][0]["threshold"], float)


def test_cmd_train_or_load_dl_rejects_invalid_selection_metric_before_training(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    (work_dir / "feature_stack.tif").write_text("placeholder", encoding="utf-8")
    (work_dir / "samples_train.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")
    (work_dir / "samples_val.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")

    monkeypatch.setattr("forestseg.cli.infer_to_files", lambda **kwargs: None)

    cfg = _make_base_cfg(tmp_path, dl_overrides={"selection_metric": " auc "})
    cfg["work_dir"] = str(work_dir)
    with pytest.raises(ValueError, match="dl.selection_metric"):
        cmd_train_or_load_dl(cfg, force_train=True)


def test_cmd_train_or_load_dl_rejects_stride_greater_than_tile_size_before_training(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    (work_dir / "feature_stack.tif").write_text("placeholder", encoding="utf-8")
    (work_dir / "samples_train.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")
    (work_dir / "samples_val.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")

    monkeypatch.setattr(
        "forestseg.cli.infer_to_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("infer_to_files should not run")),
    )

    cfg = _make_base_cfg(tmp_path, dl_overrides={"tile_size": 128, "stride": 256})
    cfg["work_dir"] = str(work_dir)

    with pytest.raises(ValueError, match=r"dl\.stride"):
        cmd_train_or_load_dl(cfg, force_train=True)


@pytest.mark.parametrize(
    ("override_key", "bad_value"),
    [
        ("tile_size", True),
        ("tile_size", None),
        ("tile_size", "oops"),
        ("epochs", True),
        ("epochs", None),
        ("epochs", "oops"),
        ("val_threshold", -0.1),
        ("val_threshold", 1.1),
        ("val_threshold", True),
        ("val_threshold", None),
        ("val_threshold", "oops"),
    ],
)
def test_cmd_train_or_load_dl_rejects_invalid_training_inputs_before_training_runtime(
    monkeypatch,
    tmp_path,
    override_key,
    bad_value,
):
    work_dir = _make_prepared_workdir(tmp_path)
    (work_dir / "feature_stack.tif").write_text("placeholder", encoding="utf-8")
    (work_dir / "samples_train.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")
    (work_dir / "samples_val.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")

    monkeypatch.setattr(
        "forestseg.cli.load_points_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("load_points_json should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.train_supervised_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("train_supervised_model should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.infer_to_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("infer_to_files should not run")),
    )

    cfg = _make_base_cfg(tmp_path, dl_overrides={override_key: bad_value})
    cfg["work_dir"] = str(work_dir)

    match_text = rf"dl\.{override_key}"
    with pytest.raises(ValueError, match=match_text):
        cmd_train_or_load_dl(cfg, force_train=True)


@pytest.mark.parametrize(
    ("override_key", "bad_value"),
    [
        ("batch_size", True),
        ("mc_dropout_passes", True),
        ("batch_size", None),
        ("mc_dropout_passes", None),
        ("batch_size", "oops"),
        ("mc_dropout_passes", "oops"),
    ],
)
def test_cmd_train_or_load_dl_rejects_invalid_positive_ints_before_training(
    monkeypatch,
    tmp_path,
    override_key,
    bad_value,
):
    work_dir = _make_prepared_workdir(tmp_path)
    (work_dir / "feature_stack.tif").write_text("placeholder", encoding="utf-8")
    (work_dir / "samples_train.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")
    (work_dir / "samples_val.json").write_text('{"points": [], "meta": {}}', encoding="utf-8")

    monkeypatch.setattr("forestseg.cli.infer_to_files", lambda **kwargs: None)

    cfg = _make_base_cfg(tmp_path, dl_overrides={override_key: bad_value})
    cfg["work_dir"] = str(work_dir)
    with pytest.raises(ValueError, match=rf"dl\.{override_key}"):
        cmd_train_or_load_dl(cfg, force_train=True)


def test_main_applies_stage_override_to_preflight_check(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    monkeypatch.setattr(
        "forestseg.cli._build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(command="preflight-check", config="cfg.yaml", force_train=False, stage=0)
        ),
    )
    monkeypatch.setattr(
        "forestseg.cli.load_cfg",
        lambda path: {"fusion": {"stage": 1, "grid": {}, "fixed_params": {}}},
    )

    def fake_preflight(current_cfg):
        captured["stage"] = current_cfg["fusion"]["stage"]
        return _make_preflight_stage_report(current_cfg["fusion"]["stage"])

    monkeypatch.setattr("forestseg.cli.cmd_preflight_check", fake_preflight)

    main()

    assert captured["stage"] == 0
    assert json.loads(capsys.readouterr().out)["config"]["fusion"]["stage"] == 0


@pytest.mark.parametrize(
    "command_name",
    ["run-closed-loop", "run-all"],
)
def test_main_forwards_stage_override_to_cli_commands(
    monkeypatch,
    capsys,
    command_name,
):
    captured = {}

    monkeypatch.setattr(
        "forestseg.cli._build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(command=command_name, config="cfg.yaml", force_train=False, stage=0)
        ),
    )
    monkeypatch.setattr(
        "forestseg.cli.load_cfg",
        lambda path: {"fusion": {"stage": 1, "grid": {}, "fixed_params": {}}},
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_preflight_check",
        lambda current_cfg: (_ for _ in ()).throw(
            AssertionError("cmd_preflight_check should not run directly from main")
        ),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_run_closed_loop",
        lambda current_cfg, stage_override=None: (
            captured.update({"command": "run-closed-loop", "stage_override": stage_override})
            or {"status": "ok", "stage_override": stage_override}
        ),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_run_all",
        lambda current_cfg, force_train=False, stage_override=None: (
            captured.update({"command": "run-all", "force_train": force_train, "stage_override": stage_override})
            or {"status": "ok", "stage_override": stage_override}
        ),
    )

    main()

    assert captured["command"] == command_name
    assert captured["stage_override"] == 0
    assert json.loads(capsys.readouterr().out)["stage_override"] == 0


def test_cmd_run_closed_loop_applies_stage_override_to_preflight(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    cfg["fusion"]["stage"] = 1
    cfg["fusion"]["grid"] = {}

    captured = {}

    def fake_preflight(current_cfg):
        captured["preflight_stage"] = current_cfg["fusion"]["stage"]
        return _make_preflight_stage_report(current_cfg["fusion"]["stage"])

    _stub_closed_loop_prereqs(monkeypatch, preflight=fake_preflight)
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: _make_train_result(train_metrics={"f1": 0.8}),
    )

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        captured["runtime_stage_override"] = stage_override
        params = _make_fusion_params(threshold=0.5)
        (work_dir / "fusion_selected.json").write_text(
            json.dumps(
                {
                    "params": params,
                    "reward": 0.8,
                    "selection_metric": "reward",
                    "score": 0.8,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (work_dir / "feature_feedback.json").write_text(json.dumps({"reward": 0.8}), encoding="utf-8")
        (work_dir / "metrics_val.json").write_text(json.dumps({"f1": 0.8}), encoding="utf-8")
        (work_dir / "prob_dl.tif").write_bytes(b"dl")
        (work_dir / "prob_fused.tif").write_bytes(b"fused")
        (work_dir / "rl_history.json").write_text(
            json.dumps(
                [
                    _make_rl_history_payload(
                        round_no=1,
                        score=0.8,
                        reward=0.8,
                        threshold=0.5,
                        params=params,
                        train_metrics={"f1": 0.8},
                        validation_metrics={"reward": 0.8, "f1": 0.8},
                        stage_used="validation_stage0",
                    )
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return _make_rl_result(reward=0.8, f1=0.8, params=params, stage_used="validation_stage0")

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    summary = cmd_run_closed_loop(cfg, stage_override=0)

    assert captured["preflight_stage"] == 0
    assert captured["runtime_stage_override"] == 0
    assert summary["preflight"]["config"]["fusion"]["stage"] == 0


def test_cmd_run_all_applies_stage_override_to_preflight_and_closed_loop(
    monkeypatch,
    tmp_path,
):
    cfg = _make_base_cfg(tmp_path)
    cfg["fusion"]["stage"] = 1
    cfg["fusion"]["grid"] = {}

    captured = {}

    def fake_preflight(current_cfg):
        captured["preflight_stage"] = current_cfg["fusion"]["stage"]
        return {"status": "ok"}

    monkeypatch.setattr("forestseg.cli.cmd_preflight_check", fake_preflight)
    monkeypatch.setattr(
        "forestseg.cli.cmd_prepare_input",
        lambda current_cfg: captured.setdefault("prepare_input_called", True),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_build_spec_tex",
        lambda current_cfg: captured.setdefault("build_spec_tex_called", True),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_prepare_label_points",
        lambda current_cfg: captured.setdefault("prepare_label_points_called", True),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_build_feature_stack",
        lambda current_cfg: captured.setdefault("build_feature_stack_called", True),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: captured.setdefault("force_train", force_train),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_run_closed_loop",
        lambda current_cfg, stage_override=None: captured.update({"closed_loop_stage_override": stage_override}),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_postprocess_export",
        lambda current_cfg: {"status": "done"},
    )

    result = cmd_run_all(cfg, force_train=False, stage_override=0)

    assert captured["preflight_stage"] == 0
    assert captured["closed_loop_stage_override"] == 0
    assert result == {"status": "done"}


def test_cmd_run_closed_loop_preserves_stale_rl_history_until_first_round_succeeds(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    stale_history_path = work_dir / "rl_history.json"
    stale_selected_path = work_dir / "fusion_selected.json"
    stale_feedback_path = work_dir / "feature_feedback.json"
    stale_history_path.write_text('[{"round": 99, "reward": -1.0}]', encoding="utf-8")
    stale_selected_path.write_text('{"params": {"threshold": 0.9}}', encoding="utf-8")
    stale_feedback_path.write_text('{"threshold": 0.9}', encoding="utf-8")
    stale_artifact_state_before_calls = []

    def fake_build_feature_stack(current_cfg):
        stale_artifact_state_before_calls.append(
            {
                "rl_history": stale_history_path.exists(),
                "fusion_selected": stale_selected_path.exists(),
                "feature_feedback": stale_feedback_path.exists(),
            }
        )

    _stub_closed_loop_prereqs(monkeypatch, build_feature_stack=fake_build_feature_stack)
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: (
            (work_dir / "prob_dl.tif").write_bytes(b"round-1-dl"),
            (work_dir / "metrics_val.json").write_text(
                json.dumps({"f1": 0.8}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            _make_train_result(train_metrics={"f1": 0.8}),
        )[-1],
    )

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        stale_history_path.write_text(
            json.dumps(
                [
                    _make_rl_history_payload(
                        round_no=1,
                        score=0.2,
                        reward=0.2,
                        threshold=0.2,
                        train_metrics={"f1": 0.8},
                        validation_metrics={"reward": 0.2, "f1": 0.7},
                    )
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        stale_selected_path.write_text(
            json.dumps(
                _make_fusion_selected_payload(params=_make_fusion_params(threshold=0.2)),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        stale_feedback_path.write_text(
            json.dumps(_make_feedback_payload(threshold=0.2, reward=0.2), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_fused.tif").write_bytes(b"round-1")
        return _make_rl_result(reward=0.2, f1=0.7)

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    summary = cmd_run_closed_loop(cfg)

    assert stale_artifact_state_before_calls == [
        {"rl_history": True, "fusion_selected": True, "feature_feedback": True}
    ]
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(stale_history_path),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    stale_history = _load_rl_history(str(stale_history_path))
    assert stale_history[0]["round"] == 1
    assert stale_history[0]["selection_metric"] == "reward"
    assert stale_history[0]["score"] == 0.2
    assert stale_history[0]["reward"] == 0.2


def test_cmd_run_closed_loop_preserves_stale_rl_artifacts_when_first_round_fails(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    stale_history_path = work_dir / "rl_history.json"
    stale_selected_path = work_dir / "fusion_selected.json"
    stale_feedback_path = work_dir / "feature_feedback.json"
    stale_history_text = '[{"round": 99, "reward": -1.0}]'
    stale_selected_text = '{"params": {"threshold": 0.9}}'
    stale_feedback_text = '{"threshold": 0.9}'
    stale_history_path.write_text(stale_history_text, encoding="utf-8")
    stale_selected_path.write_text(stale_selected_text, encoding="utf-8")
    stale_feedback_path.write_text(stale_feedback_text, encoding="utf-8")

    _stub_closed_loop_prereqs(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: _make_train_result(train_metrics={"f1": 0.8}),
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_run_rl_fusion",
        lambda current_cfg, stage_override=None: (_ for _ in ()).throw(ValueError("boom during fusion")),
    )

    with pytest.raises(ValueError, match="boom during fusion"):
        cmd_run_closed_loop(cfg)

    assert stale_history_path.read_text(encoding="utf-8") == stale_history_text
    assert stale_selected_path.read_text(encoding="utf-8") == stale_selected_text
    assert stale_feedback_path.read_text(encoding="utf-8") == stale_feedback_text


def test_cmd_run_closed_loop_selects_best_by_selection_metric_even_when_reward_is_lower(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 2, "patience": 2, "selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)

    round_payloads = [
        {
            "checkpoint_path": "ckpt-round-1.pt",
            "train_metrics": {"f1": 0.8},
            "metrics_val": {"f1": 0.9, "loss": 0.1},
            "validation_reward": {"reward": 0.1, "f1": 0.9},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.4, lambda_spec=0.1, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.4, reward=0.1, lambda_spec=0.1, lambda_tex=0.0),
            "prob_dl": b"best-round-dl",
            "unc_dl": b"best-round-unc",
            "prob_fused": b"best-round-fused",
            "payload": _make_rl_history_payload(
                round_no=1,
                score=0.1,
                reward=0.1,
                threshold=0.4,
                lambda_spec=0.1,
                lambda_tex=0.0,
                train_metrics={"f1": 0.9, "loss": 0.1},
                validation_metrics={"reward": 0.1, "f1": 0.9},
                state={"round": 1},
                split_meta={},
                preview_shape=[2, 2],
            ),
        },
        {
            "checkpoint_path": "ckpt-round-2.pt",
            "train_metrics": {"f1": 0.3},
            "metrics_val": {"f1": 0.1, "loss": 0.9},
            "validation_reward": {"reward": 0.9, "f1": 0.1},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.8, lambda_spec=0.9, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.8, reward=0.9, lambda_spec=0.9, lambda_tex=0.0),
            "prob_dl": b"last-round-dl",
            "unc_dl": b"last-round-unc",
            "prob_fused": b"last-round-fused",
            "payload": _make_rl_history_payload(
                round_no=2,
                score=0.9,
                reward=0.9,
                threshold=0.8,
                lambda_spec=0.9,
                lambda_tex=0.0,
                train_metrics={"f1": 0.1, "loss": 0.9},
                validation_metrics={"reward": 0.9, "f1": 0.1},
                state={"round": 2},
                split_meta={},
                preview_shape=[2, 2],
            ),
        },
    ]
    current_round = {"value": 0}

    def fake_train_or_load_dl(current_cfg, force_train=False):
        payload = round_payloads[current_round["value"]]
        (work_dir / "metrics_val.json").write_text(
            json.dumps(payload["metrics_val"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_dl.tif").write_bytes(payload["prob_dl"])
        (work_dir / "unc_dl.tif").write_bytes(payload["unc_dl"])
        return _make_train_result(
            checkpoint_path=payload["checkpoint_path"],
            train_metrics=payload["train_metrics"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_train_or_load_dl", fake_train_or_load_dl)

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        payload = round_payloads[current_round["value"]]
        current_round["value"] += 1
        (work_dir / "fusion_selected.json").write_text(
            json.dumps(payload["payload"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "rl_history.json").write_text(
            json.dumps([payload["payload"]], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "feature_feedback.json").write_text(
            json.dumps(payload["feedback"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_fused.tif").write_bytes(payload["prob_fused"])
        return _make_rl_result(
            reward=payload["validation_reward"]["reward"],
            f1=payload["validation_reward"]["f1"],
            params=payload["params"],
            stage_used=payload["stage_used"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    summary = cmd_run_closed_loop(cfg)
    persisted_summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    metrics_history = json.loads((work_dir / "metrics_round_history.json").read_text(encoding="utf-8"))

    assert summary == persisted_summary
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    assert len(metrics_history) == 2
    assert summary["schema_version"] == 2
    assert summary["status"] == "ok"
    assert summary["loop"] == {
        "rounds_requested": 2,
        "rounds_completed": 2,
        "selection_metric": "f1",
        "patience": 2,
        "min_delta": 0.001,
        "stopped_early": False,
    }
    assert summary["best"]["round"] == 1
    assert summary["best"]["selection_metric"] == "f1"
    assert summary["best"]["score"] == 0.9
    assert summary["best"]["reward"] == 0.1
    assert summary["best"]["entry"] == summary["history"][0]
    _assert_round_history_schema_consistency(
        summary["history"][0],
        metrics_history[0],
        expected={
            "stage_used": "validation_stage1",
            "threshold": 0.4,
            "checkpoint_path": "ckpt-round-1.pt",
            "trained": True,
            "feature_feedback_suffix": "round_01\\feature_feedback.json",
            "fusion_selected_suffix": "round_01\\fusion_selected.json",
            "metrics_val_suffix": "round_01\\metrics_val.json",
            "prob_dl_suffix": "round_01\\prob_dl.tif",
            "unc_dl_suffix": "round_01\\unc_dl.tif",
            "prob_fused_suffix": "round_01\\prob_fused.tif",
            "feature_meta": {},
            "selection_metric": "f1",
            "score": 0.9,
            "reward": 0.1,
        },
    )
    _assert_round_history_schema_consistency(
        summary["history"][1],
        metrics_history[1],
        expected={
            "stage_used": "validation_stage1",
            "threshold": 0.8,
            "checkpoint_path": "ckpt-round-2.pt",
            "trained": True,
            "feature_feedback_suffix": "round_02\\feature_feedback.json",
            "fusion_selected_suffix": "round_02\\fusion_selected.json",
            "metrics_val_suffix": "round_02\\metrics_val.json",
            "prob_dl_suffix": "round_02\\prob_dl.tif",
            "unc_dl_suffix": "round_02\\unc_dl.tif",
            "prob_fused_suffix": "round_02\\prob_fused.tif",
            "feature_meta": {},
            "selection_metric": "f1",
            "score": 0.1,
            "reward": 0.9,
        },
    )
    best_entry = summary["best"]["entry"]
    best_artifacts = best_entry["artifacts"]
    assert json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8")) == json.loads(
        Path(best_artifacts["fusion_selected"]).read_text(encoding="utf-8")
    )
    assert json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8")) == json.loads(
        Path(best_artifacts["rl_payload"]).read_text(encoding="utf-8")
    )
    assert json.loads((work_dir / "feature_feedback.json").read_text(encoding="utf-8")) == json.loads(
        Path(best_artifacts["feature_feedback"]).read_text(encoding="utf-8")
    )
    assert (work_dir / "prob_dl.tif").read_bytes() == Path(best_artifacts["prob_dl"]).read_bytes()
    assert (work_dir / "unc_dl.tif").read_bytes() == Path(best_artifacts["unc_dl"]).read_bytes()
    assert (work_dir / "prob_fused.tif").read_bytes() == Path(best_artifacts["prob_fused"]).read_bytes()
    assert json.loads((work_dir / "metrics_val.json").read_text(encoding="utf-8")) == json.loads(
        Path(best_artifacts["metrics_val"]).read_text(encoding="utf-8")
    )
    assert best_entry["feedback_path"] == best_artifacts["feature_feedback"]


def test_cmd_run_closed_loop_writes_schema_version_into_round_history(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 1, "patience": 1, "selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: (
            (work_dir / "prob_dl.tif").write_bytes(b"round-dl"),
            (work_dir / "unc_dl.tif").write_bytes(b"round-unc"),
            (work_dir / "metrics_val.json").write_text(
                json.dumps({"f1": 0.7}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            _make_train_result(train_metrics={"f1": 0.8}),
        )[-1],
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_run_rl_fusion",
        lambda current_cfg, stage_override=None: (
            (work_dir / "fusion_selected.json").write_text(
                json.dumps(
                    _make_fusion_selected_payload(params=_make_fusion_params(threshold=0.7)),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            ),
            (work_dir / "feature_feedback.json").write_text(
                json.dumps(_make_feedback_payload(threshold=0.7, reward=0.1), ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            (work_dir / "rl_history.json").write_text(
                json.dumps(
                    [
                        _make_rl_history_payload(
                            round_no=1,
                            score=0.1,
                            reward=0.1,
                            threshold=0.7,
                            train_metrics={"f1": 0.8},
                            validation_metrics={"reward": 0.1, "f1": 0.7},
                        )
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            ),
            (work_dir / "prob_fused.tif").write_bytes(b"round-fused"),
            _make_rl_result(reward=0.1, f1=0.7, params=_make_fusion_params(threshold=0.7)),
        )[-1],
    )

    summary = cmd_run_closed_loop(cfg)
    persisted_summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    metrics_history = json.loads((work_dir / "metrics_round_history.json").read_text(encoding="utf-8"))

    assert summary == persisted_summary

    assert summary["schema_version"] == 2
    assert summary["status"] == "ok"
    assert summary["loop"] == {
        "rounds_requested": 1,
        "rounds_completed": 1,
        "selection_metric": "f1",
        "patience": 1,
        "min_delta": 0.001,
        "stopped_early": False,
    }
    assert summary["best"]["round"] == 1
    assert summary["best"]["selection_metric"] == "f1"
    assert summary["best"]["score"] == 0.7
    assert summary["best"]["reward"] == 0.1
    assert summary["best"]["restore_outcome"] == {
        "restored_optional_artifacts": ["unc_dl"],
        "failed_optional_artifacts": [],
        "skipped_optional_artifacts": [],
    }
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    _assert_round_history_schema_consistency(
        summary["history"][0],
        metrics_history[0],
        expected={
            "stage_used": "validation_stage1",
            "threshold": 0.7,
            "checkpoint_path": "ckpt.pt",
            "trained": True,
            "feature_feedback_suffix": "round_01\\feature_feedback.json",
            "fusion_selected_suffix": "round_01\\fusion_selected.json",
            "metrics_val_suffix": "round_01\\metrics_val.json",
            "prob_dl_suffix": "round_01\\prob_dl.tif",
            "unc_dl_suffix": "round_01\\unc_dl.tif",
            "prob_fused_suffix": "round_01\\prob_fused.tif",
            "feature_meta": {},
            "selection_metric": "f1",
            "score": 0.7,
            "reward": 0.1,
        },
    )


def test_cmd_run_closed_loop_rewrites_rl_history_to_loop_selection_metric(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 2, "patience": 2, "selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)

    round_payloads = [
        {
            "train_metrics": {"f1": 0.8},
            "metrics_val": {"f1": 0.9},
            "validation_reward": {"reward": 0.1, "f1": 0.9},
            "params": _make_fusion_params(threshold=0.4, lambda_spec=0.1, lambda_tex=0.0),
        },
        {
            "train_metrics": {"f1": 0.3},
            "metrics_val": {"f1": 0.1},
            "validation_reward": {"reward": 0.9, "f1": 0.1},
            "params": _make_fusion_params(threshold=0.8, lambda_spec=0.9, lambda_tex=0.0),
        },
    ]
    current_round = {"value": 0}

    def fake_train_or_load_dl(current_cfg, force_train=False):
        payload = round_payloads[current_round["value"]]
        (work_dir / "metrics_val.json").write_text(
            json.dumps(payload["metrics_val"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_dl.tif").write_bytes(f"dl-{current_round['value'] + 1}".encode())
        (work_dir / "unc_dl.tif").write_bytes(f"unc-{current_round['value'] + 1}".encode())
        return _make_train_result(train_metrics=payload["train_metrics"])

    monkeypatch.setattr("forestseg.cli.cmd_train_or_load_dl", fake_train_or_load_dl)

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        payload = round_payloads[current_round["value"]]
        round_no = current_round["value"] + 1
        current_round["value"] += 1
        history_path = work_dir / "rl_history.json"
        history = []
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        history.append(
            _make_rl_history_payload(
                round_no=round_no,
                score=payload["validation_reward"]["reward"],
                reward=payload["validation_reward"]["reward"],
                threshold=payload["params"]["threshold"],
                train_metrics=payload["train_metrics"],
                validation_metrics=payload["validation_reward"],
                params=payload["params"],
            )
        )
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        (work_dir / "fusion_selected.json").write_text(
            json.dumps(_make_fusion_selected_payload(params=payload["params"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "feature_feedback.json").write_text(
            json.dumps(
                _make_feedback_payload(
                    threshold=payload["params"]["threshold"],
                    reward=payload["validation_reward"]["reward"],
                    lambda_spec=payload["params"]["lambda_spec"],
                    lambda_tex=payload["params"]["lambda_tex"],
                    min_area_m2=payload["params"]["min_area_m2"],
                    morph_kernel=payload["params"]["morph_kernel"],
                    shadow_penalty=payload["params"]["shadow_penalty"],
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (work_dir / "prob_fused.tif").write_bytes(f"fused-{round_no}".encode())
        return _make_rl_result(
            reward=payload["validation_reward"]["reward"],
            f1=payload["validation_reward"]["f1"],
            params=payload["params"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    cmd_run_closed_loop(cfg)
    history = _load_rl_history(str(work_dir / "rl_history.json"))

    assert len(history) == 2
    _assert_rl_history_entry(
        history[0],
        expected={
            "round": 1,
            "selection_metric": "f1",
            "score": 0.9,
            "reward": 0.1,
            "stage_used": "validation_stage1",
            "threshold": 0.4,
        },
    )
    _assert_rl_history_entry(
        history[1],
        expected={
            "round": 2,
            "selection_metric": "f1",
            "score": 0.1,
            "reward": 0.9,
            "stage_used": "validation_stage1",
            "threshold": 0.8,
            "validation_f1": 0.1,
        },
    )
    selected_payload = json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8"))

    assert selected_payload["selection_metric"] == history[0]["selection_metric"]
    assert selected_payload["score"] == history[0]["score"]


def test_cmd_run_closed_loop_marks_success_summary_as_stopped_early_when_patience_breaks(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 3, "patience": 1, "selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)

    round_payloads = [
        {
            "train_metrics": {"f1": 0.8},
            "metrics_val": {"f1": 0.9},
            "validation_reward": {"reward": 0.1, "f1": 0.9},
            "params": _make_fusion_params(threshold=0.4, lambda_spec=0.1, lambda_tex=0.0),
        },
        {
            "train_metrics": {"f1": 0.3},
            "metrics_val": {"f1": 0.1},
            "validation_reward": {"reward": 0.9, "f1": 0.1},
            "params": _make_fusion_params(threshold=0.8, lambda_spec=0.9, lambda_tex=0.0),
        },
    ]
    current_round = {"value": 0}

    def fake_train_or_load_dl(current_cfg, force_train=False):
        payload = round_payloads[current_round["value"]]
        (work_dir / "metrics_val.json").write_text(
            json.dumps(payload["metrics_val"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_dl.tif").write_bytes(f"dl-{current_round['value'] + 1}".encode())
        (work_dir / "unc_dl.tif").write_bytes(f"unc-{current_round['value'] + 1}".encode())
        return _make_train_result(train_metrics=payload["train_metrics"])

    monkeypatch.setattr("forestseg.cli.cmd_train_or_load_dl", fake_train_or_load_dl)

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        payload = round_payloads[current_round["value"]]
        round_no = current_round["value"] + 1
        current_round["value"] += 1
        history_path = work_dir / "rl_history.json"
        history = []
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        history.append(
            _make_rl_history_payload(
                round_no=round_no,
                score=payload["validation_reward"]["reward"],
                reward=payload["validation_reward"]["reward"],
                threshold=payload["params"]["threshold"],
                train_metrics=payload["train_metrics"],
                validation_metrics=payload["validation_reward"],
                params=payload["params"],
            )
        )
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        (work_dir / "fusion_selected.json").write_text(
            json.dumps(_make_fusion_selected_payload(params=payload["params"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "feature_feedback.json").write_text(
            json.dumps(
                _make_feedback_payload(
                    threshold=payload["params"]["threshold"],
                    reward=payload["validation_reward"]["reward"],
                    lambda_spec=payload["params"]["lambda_spec"],
                    lambda_tex=payload["params"]["lambda_tex"],
                    min_area_m2=payload["params"]["min_area_m2"],
                    morph_kernel=payload["params"]["morph_kernel"],
                    shadow_penalty=payload["params"]["shadow_penalty"],
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (work_dir / "prob_fused.tif").write_bytes(f"fused-{round_no}".encode())
        return _make_rl_result(
            reward=payload["validation_reward"]["reward"],
            f1=payload["validation_reward"]["f1"],
            params=payload["params"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    summary = cmd_run_closed_loop(cfg)
    persisted_summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    metrics_history = json.loads((work_dir / "metrics_round_history.json").read_text(encoding="utf-8"))

    assert current_round["value"] == 2
    assert summary == persisted_summary
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    assert len(summary["history"]) == 2
    assert len(metrics_history) == 2
    assert summary["loop"] == {
        "rounds_requested": 3,
        "rounds_completed": 2,
        "selection_metric": "f1",
        "patience": 1,
        "min_delta": 0.001,
        "stopped_early": True,
    }
    assert summary["best"]["round"] == 1
    assert summary["best"]["score"] == 0.9
    assert summary["best"]["reward"] == 0.1
    assert summary["best"]["entry"] == summary["history"][0]
    assert summary["history"][0]["artifacts"] == metrics_history[0]["artifacts"]
    assert summary["history"][1]["artifacts"] == metrics_history[1]["artifacts"]
    assert metrics_history[0]["round"] == 1
    assert metrics_history[0]["selection_metric"] == "f1"
    assert metrics_history[0]["score"] == 0.9
    assert metrics_history[0]["reward"] == 0.1
    assert metrics_history[1]["round"] == 2
    assert metrics_history[1]["selection_metric"] == "f1"
    assert metrics_history[1]["score"] == 0.1
    assert metrics_history[1]["reward"] == 0.9


def test_cmd_run_closed_loop_writes_failure_summary_for_round_stage_errors(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 2, "patience": 2})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: _make_train_result(train_metrics={"f1": 0.8}),
    )

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        raise ValueError("boom during fusion")

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    with pytest.raises(ValueError, match="boom during fusion"):
        cmd_run_closed_loop(cfg)

    persisted_summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    history = json.loads((work_dir / "metrics_round_history.json").read_text(encoding="utf-8"))

    _assert_failed_closed_loop_summary(
        persisted_summary,
        expected={
            "stage": "run_rl_fusion",
            "round": 1,
            "error_type": "ValueError",
            "message_contains": "boom during fusion",
            "rounds_requested": 2,
            "rounds_completed": 0,
            "selection_metric": "reward",
            "patience": 2,
            "min_delta": 0.001,
            "stopped_early": False,
            "history_empty": True,
            "best_empty": True,
            "restore_outcome": {
                "restored_optional_artifacts": [],
                "failed_optional_artifacts": [],
                "skipped_optional_artifacts": [],
            },
        },
    )
    assert persisted_summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    assert not history


def test_cmd_run_closed_loop_writes_failure_summary_for_best_round_restore_errors(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 1, "patience": 1})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: (
            (work_dir / "metrics_val.json").write_text(
                json.dumps({"f1": 0.9}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            (work_dir / "prob_dl.tif").write_bytes(b"best-round-dl"),
            (work_dir / "unc_dl.tif").write_bytes(b"best-round-unc"),
            _make_train_result(train_metrics={"f1": 0.9}),
        )[-1],
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_run_rl_fusion",
        lambda current_cfg, stage_override=None: (
            (work_dir / "fusion_selected.json").write_text(
                json.dumps(_make_fusion_selected_payload(threshold=0.4), ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            (work_dir / "rl_history.json").write_text(
                json.dumps(
                    [
                        _make_rl_history_payload(
                            round_no=1,
                            score=0.9,
                            reward=0.9,
                            threshold=0.4,
                            train_metrics={"f1": 0.9},
                            validation_metrics={"reward": 0.9, "f1": 0.9},
                        )
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            ),
            (work_dir / "feature_feedback.json").write_text(
                json.dumps(_make_feedback_payload(threshold=0.4, reward=0.9), ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            (work_dir / "prob_fused.tif").write_bytes(b"best-round-fused"),
            _make_rl_result(reward=0.9, f1=0.9),
        )[-1],
    )

    original_require_json = __import__("forestseg.cli", fromlist=["_require_json_copy"])._require_json_copy

    def fake_require_json(src, dst, label):
        if label == "best round metrics_val":
            raise ValueError(f"Missing {label}: {src}")
        return original_require_json(src, dst, label)

    monkeypatch.setattr("forestseg.cli._require_json_copy", fake_require_json)

    with pytest.raises(ValueError, match="best round metrics_val"):
        cmd_run_closed_loop(cfg)

    summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    history = json.loads((work_dir / "metrics_round_history.json").read_text(encoding="utf-8"))

    _assert_failed_closed_loop_summary(
        summary,
        expected={
            "stage": "restore_best_round",
            "round": 1,
            "error_type": "ValueError",
            "message_contains": "best round metrics_val",
            "rounds_requested": 1,
            "rounds_completed": 1,
            "selection_metric": "reward",
            "patience": 1,
            "min_delta": 0.001,
            "stopped_early": False,
            "history_len": 1,
            "restore_outcome": {
                "restored_optional_artifacts": [],
                "failed_optional_artifacts": [],
                "skipped_optional_artifacts": ["unc_dl"],
            },
        },
    )
    assert len(summary["history"]) == len(history) == 1
    assert summary["history"][0]["schema_version"] == 2
    assert history[0]["schema_version"] == 2
    assert summary["history"][0]["selection_metric"] == history[0]["selection_metric"] == "reward"
    assert summary["history"][0]["score"] == history[0]["score"] == pytest.approx(0.9)
    assert summary["history"][0]["reward"] == history[0]["reward"] == pytest.approx(0.9)
    assert summary["history"][0]["artifacts"] == history[0]["artifacts"]
    assert summary["best"]["entry"] == summary["history"][0]
    assert summary["best"]["entry"]["selection_metric"] == history[0]["selection_metric"]
    assert summary["best"]["entry"]["score"] == history[0]["score"]
    assert summary["best"]["entry"]["reward"] == history[0]["reward"]
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    _assert_path_suffix(summary["history"][0]["artifacts"]["fusion_selected"], "round_01\\fusion_selected.json")
    _assert_path_suffix(summary["history"][0]["artifacts"]["metrics_val"], "round_01\\metrics_val.json")
    _assert_path_suffix(summary["history"][0]["artifacts"]["unc_dl"], "round_01\\unc_dl.tif")
    _assert_rl_history_entry(
        history[0],
        expected={
            "round": 1,
            "selection_metric": "reward",
            "score": 0.9,
            "reward": 0.9,
            "validation_f1": 0.9,
        },
    )


def test_cmd_run_closed_loop_raises_when_best_round_metrics_restore_is_missing(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 1, "patience": 1})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: (
            (work_dir / "metrics_val.json").write_text(
                json.dumps({"f1": 0.9}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            (work_dir / "prob_dl.tif").write_bytes(b"best-round-dl"),
            (work_dir / "unc_dl.tif").write_bytes(b"best-round-unc"),
            _make_train_result(train_metrics={"f1": 0.9}),
        )[-1],
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_run_rl_fusion",
        lambda current_cfg, stage_override=None: (
            (work_dir / "fusion_selected.json").write_text(
                json.dumps(_make_fusion_selected_payload(threshold=0.4), ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            (work_dir / "rl_history.json").write_text(
                json.dumps(
                    [
                        _make_rl_history_payload(
                            round_no=1,
                            score=0.9,
                            reward=0.9,
                            threshold=0.4,
                            train_metrics={"f1": 0.9},
                            validation_metrics={"reward": 0.9, "f1": 0.9},
                        )
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            ),
            (work_dir / "feature_feedback.json").write_text(
                json.dumps(_make_feedback_payload(threshold=0.4, reward=0.9), ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            (work_dir / "prob_fused.tif").write_bytes(b"best-round-fused"),
            _make_rl_result(reward=0.9, f1=0.9),
        )[-1],
    )

    original_require_json = __import__("forestseg.cli", fromlist=["_require_json_copy"])._require_json_copy

    def fake_require_json(src, dst, label):
        if label == "best round metrics_val":
            raise ValueError(f"Missing {label}: {src}")
        return original_require_json(src, dst, label)

    monkeypatch.setattr("forestseg.cli._require_json_copy", fake_require_json)

    with pytest.raises(ValueError, match="best round metrics_val"):
        cmd_run_closed_loop(cfg)


def test_cmd_run_closed_loop_raises_when_round_artifact_snapshot_is_missing(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 2, "patience": 2})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: _make_train_result(train_metrics={"f1": 0.8}),
    )

    round_payloads = [
        {
            "validation_reward": {"reward": 0.9, "f1": 0.9},
            "stage_used": "validation_stage1",
            "params": {"threshold": 0.4},
            "feedback": _make_feedback_payload(threshold=0.4, reward=0.9),
            "prob_fused": b"best-round-fused",
            "remove_feedback_after_round": True,
        },
        {
            "validation_reward": {"reward": 0.1, "f1": 0.1},
            "stage_used": "validation_stage1",
            "params": {"threshold": 0.8},
            "feedback": _make_feedback_payload(threshold=0.8, reward=0.1),
            "prob_fused": b"last-round-fused",
            "remove_feedback_after_round": False,
        },
    ]
    call_index = {"value": 0}

    original_copy_json = __import__("forestseg.cli", fromlist=["_snapshot_required_json"])._snapshot_required_json

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        payload = round_payloads[call_index["value"]]
        round_no = call_index["value"] + 1
        call_index["value"] += 1
        (work_dir / "fusion_selected.json").write_text(
            json.dumps(_make_fusion_selected_payload(params=payload["params"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "feature_feedback.json").write_text(
            json.dumps(payload["feedback"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "rl_history.json").write_text(
            json.dumps(
                [
                    _make_rl_history_payload(
                        round_no=round_no,
                        score=payload["validation_reward"]["reward"],
                        reward=payload["validation_reward"]["reward"],
                        threshold=payload["params"]["threshold"],
                        stage_used=payload["stage_used"],
                        train_metrics={"f1": 0.8},
                        validation_metrics=payload["validation_reward"],
                    )
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (work_dir / "prob_fused.tif").write_bytes(payload["prob_fused"])
        return _make_rl_result(
            reward=payload["validation_reward"]["reward"],
            f1=payload["validation_reward"]["f1"],
            params=payload["params"],
            stage_used=payload["stage_used"],
        )

    def fake_copy_json_if_exists(src, dst, label):
        normalized_src = str(src).replace("\\", "/")
        if label == "round 1 feature_feedback" and normalized_src.endswith("feature_feedback.json"):
            raise ValueError(f"Missing {label}: {src}")
        return original_copy_json(src, dst, label)

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)
    monkeypatch.setattr("forestseg.cli._snapshot_required_json", fake_copy_json_if_exists)

    with pytest.raises(ValueError, match="round 1 feature_feedback"):
        cmd_run_closed_loop(cfg)

    summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    history = json.loads((work_dir / "metrics_round_history.json").read_text(encoding="utf-8"))

    _assert_failed_closed_loop_summary(
        summary,
        expected={
            "stage": "snapshot_round_artifacts",
            "round": 1,
            "error_type": "ValueError",
            "message_contains": "round 1 feature_feedback",
            "rounds_requested": 2,
            "rounds_completed": 0,
            "selection_metric": "reward",
            "patience": 2,
            "min_delta": 0.001,
            "stopped_early": False,
            "history_empty": True,
            "best_empty": True,
            "restore_outcome": {
                "restored_optional_artifacts": [],
                "failed_optional_artifacts": [],
                "skipped_optional_artifacts": [],
            },
        },
    )
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    assert not history


def test_cmd_run_closed_loop_keeps_live_artifacts_unchanged_when_best_round_restore_fails(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 2, "patience": 2, "selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)

    round_payloads = [
        {
            "checkpoint_path": "ckpt-round-1.pt",
            "train_metrics": {"f1": 0.8},
            "metrics_val": {"f1": 0.9, "loss": 0.1},
            "validation_reward": {"reward": 0.1, "f1": 0.9},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.4, lambda_spec=0.1, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.4, reward=0.1, lambda_spec=0.1, lambda_tex=0.0),
            "prob_dl": b"best-round-dl",
            "unc_dl": b"best-round-unc",
            "prob_fused": b"best-round-fused",
            "payload": _make_rl_history_payload(
                round_no=1,
                score=0.1,
                reward=0.1,
                threshold=0.4,
                lambda_spec=0.1,
                lambda_tex=0.0,
                train_metrics={"f1": 0.9, "loss": 0.1},
                validation_metrics={"reward": 0.1, "f1": 0.9},
            ),
        },
        {
            "checkpoint_path": "ckpt-round-2.pt",
            "train_metrics": {"f1": 0.3},
            "metrics_val": {"f1": 0.1, "loss": 0.9},
            "validation_reward": {"reward": 0.9, "f1": 0.1},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.8, lambda_spec=0.9, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.8, reward=0.9, lambda_spec=0.9, lambda_tex=0.0),
            "prob_dl": b"live-round-dl",
            "unc_dl": b"live-round-unc",
            "prob_fused": b"live-round-fused",
            "payload": _make_rl_history_payload(
                round_no=2,
                score=0.9,
                reward=0.9,
                threshold=0.8,
                lambda_spec=0.9,
                lambda_tex=0.0,
                train_metrics={"f1": 0.1, "loss": 0.9},
                validation_metrics={"reward": 0.9, "f1": 0.1},
            ),
        },
    ]
    current_round = {"value": 0}

    def fake_train_or_load_dl(current_cfg, force_train=False):
        payload = round_payloads[current_round["value"]]
        (work_dir / "metrics_val.json").write_text(
            json.dumps(payload["metrics_val"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_dl.tif").write_bytes(payload["prob_dl"])
        (work_dir / "unc_dl.tif").write_bytes(payload["unc_dl"])
        return _make_train_result(
            checkpoint_path=payload["checkpoint_path"],
            train_metrics=payload["train_metrics"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_train_or_load_dl", fake_train_or_load_dl)

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        payload = round_payloads[current_round["value"]]
        current_round["value"] += 1
        (work_dir / "fusion_selected.json").write_text(
            json.dumps(payload["payload"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "rl_history.json").write_text(
            json.dumps([payload["payload"]], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "feature_feedback.json").write_text(
            json.dumps(payload["feedback"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_fused.tif").write_bytes(payload["prob_fused"])
        return _make_rl_result(
            reward=payload["validation_reward"]["reward"],
            f1=payload["validation_reward"]["f1"],
            params=payload["params"],
            stage_used=payload["stage_used"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    original_require_json = __import__("forestseg.cli", fromlist=["_require_json_copy"])._require_json_copy

    def fake_require_json(src, dst, label):
        if label == "best round metrics_val":
            raise ValueError(f"Missing {label}: {src}")
        return original_require_json(src, dst, label)

    monkeypatch.setattr("forestseg.cli._require_json_copy", fake_require_json)

    with pytest.raises(ValueError, match="best round metrics_val"):
        cmd_run_closed_loop(cfg)

    summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    history = json.loads((work_dir / "metrics_round_history.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failure"]["stage"] == "restore_best_round"
    assert summary["failure"]["error_type"] == "ValueError"
    assert "best round metrics_val" in summary["failure"]["message"]
    assert summary["best"]["restore_outcome"] == {
        "restored_optional_artifacts": [],
        "failed_optional_artifacts": [],
        "skipped_optional_artifacts": ["unc_dl"],
    }
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    assert set(summary["best"]["entry"]["artifacts"]) == {
        "fusion_selected",
        "rl_payload",
        "feature_feedback",
        "metrics_val",
        "prob_dl",
        "unc_dl",
        "prob_fused",
        "train_metrics",
        "feature_meta",
    }
    _assert_path_suffix(summary["best"]["entry"]["artifacts"]["fusion_selected"], "round_01\\fusion_selected.json")
    _assert_path_suffix(summary["best"]["entry"]["artifacts"]["feature_feedback"], "round_01\\feature_feedback.json")
    _assert_path_suffix(summary["best"]["entry"]["artifacts"]["metrics_val"], "round_01\\metrics_val.json")
    _assert_path_suffix(summary["best"]["entry"]["artifacts"]["unc_dl"], "round_01\\unc_dl.tif")
    assert len(summary["history"]) == len(history) == 2
    assert summary["best"]["entry"] == summary["history"][0]
    assert (work_dir / "prob_dl.tif").read_bytes() == b"live-round-dl"
    assert (work_dir / "unc_dl.tif").read_bytes() == b"live-round-unc"
    assert (work_dir / "prob_fused.tif").read_bytes() == b"live-round-fused"
    assert json.loads((work_dir / "metrics_val.json").read_text(encoding="utf-8")) == {"f1": 0.1, "loss": 0.9}
    assert json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8"))["params"][
        "threshold"
    ] == pytest.approx(0.8)
    assert json.loads((work_dir / "feature_feedback.json").read_text(encoding="utf-8"))["threshold"] == pytest.approx(
        0.8
    )


def test_cmd_run_closed_loop_restores_best_round_unc_dl(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 2, "patience": 2, "selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)

    round_payloads = [
        {
            "checkpoint_path": "ckpt-round-1.pt",
            "train_metrics": {"f1": 0.8},
            "metrics_val": {"f1": 0.9, "loss": 0.1},
            "validation_reward": {"reward": 0.1, "f1": 0.9},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.4, lambda_spec=0.1, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.4, reward=0.1, lambda_spec=0.1, lambda_tex=0.0),
            "prob_dl": b"best-round-dl",
            "unc_dl": b"best-round-unc",
            "prob_fused": b"best-round-fused",
            "payload": _make_rl_history_payload(
                round_no=1,
                score=0.1,
                reward=0.1,
                threshold=0.4,
                lambda_spec=0.1,
                lambda_tex=0.0,
                train_metrics={"f1": 0.9, "loss": 0.1},
                validation_metrics={"reward": 0.1, "f1": 0.9},
            ),
        },
        {
            "checkpoint_path": "ckpt-round-2.pt",
            "train_metrics": {"f1": 0.3},
            "metrics_val": {"f1": 0.1, "loss": 0.9},
            "validation_reward": {"reward": 0.9, "f1": 0.1},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.8, lambda_spec=0.9, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.8, reward=0.9, lambda_spec=0.9, lambda_tex=0.0),
            "prob_dl": b"last-round-dl",
            "unc_dl": b"last-round-unc",
            "prob_fused": b"last-round-fused",
            "payload": _make_rl_history_payload(
                round_no=2,
                score=0.9,
                reward=0.9,
                threshold=0.8,
                lambda_spec=0.9,
                lambda_tex=0.0,
                train_metrics={"f1": 0.1, "loss": 0.9},
                validation_metrics={"reward": 0.9, "f1": 0.1},
            ),
        },
    ]
    current_round = {"value": 0}

    def fake_train_or_load_dl(current_cfg, force_train=False):
        payload = round_payloads[current_round["value"]]
        (work_dir / "metrics_val.json").write_text(
            json.dumps(payload["metrics_val"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_dl.tif").write_bytes(payload["prob_dl"])
        (work_dir / "unc_dl.tif").write_bytes(payload["unc_dl"])
        return _make_train_result(
            checkpoint_path=payload["checkpoint_path"],
            train_metrics=payload["train_metrics"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_train_or_load_dl", fake_train_or_load_dl)

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        payload = round_payloads[current_round["value"]]
        current_round["value"] += 1
        (work_dir / "fusion_selected.json").write_text(
            json.dumps(payload["payload"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "rl_history.json").write_text(
            json.dumps([payload["payload"]], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "feature_feedback.json").write_text(
            json.dumps(payload["feedback"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_fused.tif").write_bytes(payload["prob_fused"])
        return _make_rl_result(
            reward=payload["validation_reward"]["reward"],
            f1=payload["validation_reward"]["f1"],
            params=payload["params"],
            stage_used=payload["stage_used"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    summary = cmd_run_closed_loop(cfg)

    best_artifacts = summary["best"]["entry"]["artifacts"]
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    assert summary["best"]["round"] == 1
    assert set(best_artifacts) == {
        "fusion_selected",
        "rl_payload",
        "feature_feedback",
        "metrics_val",
        "prob_dl",
        "unc_dl",
        "prob_fused",
        "train_metrics",
        "feature_meta",
    }
    _assert_path_suffix(best_artifacts["fusion_selected"], "round_01\\fusion_selected.json")
    _assert_path_suffix(best_artifacts["feature_feedback"], "round_01\\feature_feedback.json")
    _assert_path_suffix(best_artifacts["metrics_val"], "round_01\\metrics_val.json")
    _assert_path_suffix(best_artifacts["unc_dl"], "round_01\\unc_dl.tif")
    assert (work_dir / "unc_dl.tif").read_bytes() == b"best-round-unc"


def test_cmd_run_closed_loop_preserves_live_unc_dl_when_best_round_snapshot_is_legacy(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 2, "patience": 2, "selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)

    round_payloads = [
        {
            "checkpoint_path": "ckpt-round-1.pt",
            "train_metrics": {"f1": 0.8},
            "metrics_val": {"f1": 0.9, "loss": 0.1},
            "validation_reward": {"reward": 0.1, "f1": 0.9},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.4, lambda_spec=0.1, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.4, reward=0.1, lambda_spec=0.1, lambda_tex=0.0),
            "prob_dl": b"best-round-dl",
            "unc_dl": b"best-round-unc",
            "prob_fused": b"best-round-fused",
            "payload": _make_rl_history_payload(
                round_no=1,
                score=0.1,
                reward=0.1,
                threshold=0.4,
                lambda_spec=0.1,
                lambda_tex=0.0,
                train_metrics={"f1": 0.9, "loss": 0.1},
                validation_metrics={"reward": 0.1, "f1": 0.9},
            ),
        },
        {
            "checkpoint_path": "ckpt-round-2.pt",
            "train_metrics": {"f1": 0.3},
            "metrics_val": {"f1": 0.1, "loss": 0.9},
            "validation_reward": {"reward": 0.9, "f1": 0.1},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.8, lambda_spec=0.9, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.8, reward=0.9, lambda_spec=0.9, lambda_tex=0.0),
            "prob_dl": b"last-round-dl",
            "unc_dl": b"last-round-unc",
            "prob_fused": b"last-round-fused",
            "payload": _make_rl_history_payload(
                round_no=2,
                score=0.9,
                reward=0.9,
                threshold=0.8,
                lambda_spec=0.9,
                lambda_tex=0.0,
                train_metrics={"f1": 0.1, "loss": 0.9},
                validation_metrics={"reward": 0.9, "f1": 0.1},
            ),
        },
    ]
    current_round = {"value": 0}

    def fake_train_or_load_dl(current_cfg, force_train=False):
        payload = round_payloads[current_round["value"]]
        (work_dir / "metrics_val.json").write_text(
            json.dumps(payload["metrics_val"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_dl.tif").write_bytes(payload["prob_dl"])
        (work_dir / "unc_dl.tif").write_bytes(payload["unc_dl"])
        return _make_train_result(
            checkpoint_path=payload["checkpoint_path"],
            train_metrics=payload["train_metrics"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_train_or_load_dl", fake_train_or_load_dl)

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        payload = round_payloads[current_round["value"]]
        current_round["value"] += 1
        (work_dir / "fusion_selected.json").write_text(
            json.dumps(payload["payload"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "rl_history.json").write_text(
            json.dumps([payload["payload"]], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "feature_feedback.json").write_text(
            json.dumps(payload["feedback"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_fused.tif").write_bytes(payload["prob_fused"])
        return _make_rl_result(
            reward=payload["validation_reward"]["reward"],
            f1=payload["validation_reward"]["f1"],
            params=payload["params"],
            stage_used=payload["stage_used"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    original_restore = __import__("forestseg.cli", fromlist=["_restore_outputs_atomically"])._restore_outputs_atomically

    def fake_restore(entries):
        legacy_entries = [entry for entry in entries if entry[2] != "unc_dl"]
        return original_restore(legacy_entries)

    monkeypatch.setattr("forestseg.cli._restore_outputs_atomically", fake_restore)

    summary = cmd_run_closed_loop(cfg)

    best_artifacts = summary["best"]["entry"]["artifacts"]
    assert summary["best"]["round"] == 1
    assert summary["best"]["entry"] == summary["history"][0]
    assert set(best_artifacts) == {
        "fusion_selected",
        "rl_payload",
        "feature_feedback",
        "metrics_val",
        "prob_dl",
        "unc_dl",
        "prob_fused",
        "train_metrics",
        "feature_meta",
    }
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    _assert_path_suffix(best_artifacts["fusion_selected"], "round_01\\fusion_selected.json")
    _assert_path_suffix(best_artifacts["feature_feedback"], "round_01\\feature_feedback.json")
    _assert_path_suffix(best_artifacts["metrics_val"], "round_01\\metrics_val.json")
    _assert_path_suffix(best_artifacts["unc_dl"], "round_01\\unc_dl.tif")
    assert summary["best"]["restore_outcome"] == {
        "restored_optional_artifacts": [],
        "failed_optional_artifacts": [],
        "skipped_optional_artifacts": ["unc_dl"],
    }
    assert (work_dir / "prob_dl.tif").read_bytes() == b"best-round-dl"
    assert (work_dir / "unc_dl.tif").read_bytes() == b"last-round-unc"


def test_cmd_run_closed_loop_reports_optional_restore_failure_for_unc_dl(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 2, "patience": 2, "selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)

    round_payloads = [
        {
            "checkpoint_path": "ckpt-round-1.pt",
            "train_metrics": {"f1": 0.8},
            "metrics_val": {"f1": 0.9, "loss": 0.1},
            "validation_reward": {"reward": 0.1, "f1": 0.9},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.4, lambda_spec=0.1, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.4, reward=0.1, lambda_spec=0.1, lambda_tex=0.0),
            "prob_dl": b"best-round-dl",
            "unc_dl": b"best-round-unc",
            "prob_fused": b"best-round-fused",
            "payload": _make_rl_history_payload(
                round_no=1,
                score=0.1,
                reward=0.1,
                threshold=0.4,
                lambda_spec=0.1,
                lambda_tex=0.0,
                train_metrics={"f1": 0.9, "loss": 0.1},
                validation_metrics={"reward": 0.1, "f1": 0.9},
            ),
        },
        {
            "checkpoint_path": "ckpt-round-2.pt",
            "train_metrics": {"f1": 0.3},
            "metrics_val": {"f1": 0.1, "loss": 0.9},
            "validation_reward": {"reward": 0.9, "f1": 0.1},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.8, lambda_spec=0.9, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.8, reward=0.9, lambda_spec=0.9, lambda_tex=0.0),
            "prob_dl": b"live-round-dl",
            "unc_dl": b"live-round-unc",
            "prob_fused": b"live-round-fused",
            "payload": _make_rl_history_payload(
                round_no=2,
                score=0.9,
                reward=0.9,
                threshold=0.8,
                lambda_spec=0.9,
                lambda_tex=0.0,
                train_metrics={"f1": 0.1, "loss": 0.9},
                validation_metrics={"reward": 0.9, "f1": 0.1},
            ),
        },
    ]
    current_round = {"value": 0}

    def fake_train_or_load_dl(current_cfg, force_train=False):
        payload = round_payloads[current_round["value"]]
        (work_dir / "metrics_val.json").write_text(
            json.dumps(payload["metrics_val"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_dl.tif").write_bytes(payload["prob_dl"])
        (work_dir / "unc_dl.tif").write_bytes(payload["unc_dl"])
        return _make_train_result(
            checkpoint_path=payload["checkpoint_path"],
            train_metrics=payload["train_metrics"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_train_or_load_dl", fake_train_or_load_dl)

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        payload = round_payloads[current_round["value"]]
        current_round["value"] += 1
        (work_dir / "fusion_selected.json").write_text(
            json.dumps(payload["payload"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "rl_history.json").write_text(
            json.dumps([payload["payload"]], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "feature_feedback.json").write_text(
            json.dumps(payload["feedback"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_fused.tif").write_bytes(payload["prob_fused"])
        return _make_rl_result(
            reward=payload["validation_reward"]["reward"],
            f1=payload["validation_reward"]["f1"],
            params=payload["params"],
            stage_used=payload["stage_used"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    original_require_file = __import__("forestseg.cli", fromlist=["_require_file_copy"])._require_file_copy

    def fake_require_file(src, dst, label):
        normalized_src = str(src).replace("\\", "/")
        if normalized_src.endswith("round_01/unc_dl.tif"):
            raise ValueError(f"Missing artifact snapshot: {src}")
        return original_require_file(src, dst, label)

    monkeypatch.setattr("forestseg.cli._require_file_copy", fake_require_file)

    with pytest.raises(ValueError, match=r"unc_dl\.tif"):
        cmd_run_closed_loop(cfg)

    summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    metrics_history = json.loads((work_dir / "metrics_round_history.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failure"]["stage"] == "restore_best_round"
    assert summary["failure"]["round"] == 2
    assert summary["failure"]["error_type"] == "ValueError"
    assert summary["loop"]["stopped_early"] is False
    assert "unc_dl.tif" in summary["failure"]["message"]
    assert "round_01" in summary["failure"]["message"]
    assert summary["best"]["restore_outcome"] == {
        "restored_optional_artifacts": [],
        "failed_optional_artifacts": ["unc_dl"],
        "skipped_optional_artifacts": [],
    }
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    _assert_path_suffix(summary["best"]["entry"]["artifacts"]["unc_dl"], "round_01\\unc_dl.tif")
    assert len(summary["history"]) == len(metrics_history) == 2
    assert summary["best"]["entry"] == summary["history"][0]
    _assert_round_history_schema_consistency(
        summary["history"][0],
        metrics_history[0],
        expected={
            "stage_used": "validation_stage1",
            "threshold": 0.4,
            "checkpoint_path": "ckpt-round-1.pt",
            "trained": True,
            "feature_feedback_suffix": "round_01\\feature_feedback.json",
            "fusion_selected_suffix": "round_01\\fusion_selected.json",
            "metrics_val_suffix": "round_01\\metrics_val.json",
            "prob_dl_suffix": "round_01\\prob_dl.tif",
            "unc_dl_suffix": "round_01\\unc_dl.tif",
            "prob_fused_suffix": "round_01\\prob_fused.tif",
            "feature_meta": {},
            "selection_metric": "f1",
            "score": 0.9,
            "reward": 0.1,
        },
    )
    _assert_round_history_schema_consistency(
        summary["history"][1],
        metrics_history[1],
        expected={
            "stage_used": "validation_stage1",
            "threshold": 0.8,
            "checkpoint_path": "ckpt-round-2.pt",
            "trained": True,
            "feature_feedback_suffix": "round_02\\feature_feedback.json",
            "fusion_selected_suffix": "round_02\\fusion_selected.json",
            "metrics_val_suffix": "round_02\\metrics_val.json",
            "prob_dl_suffix": "round_02\\prob_dl.tif",
            "unc_dl_suffix": "round_02\\unc_dl.tif",
            "prob_fused_suffix": "round_02\\prob_fused.tif",
            "feature_meta": {},
            "selection_metric": "f1",
            "score": 0.1,
            "reward": 0.9,
        },
    )
    assert (work_dir / "unc_dl.tif").read_bytes() == b"live-round-unc"


def test_cmd_run_closed_loop_normalizes_rl_selection_metric(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"selection_metric": " F1 "})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.cmd_train_or_load_dl",
        lambda current_cfg, force_train=False: (
            (work_dir / "prob_dl.tif").write_bytes(b"f1-best-dl"),
            (work_dir / "unc_dl.tif").write_bytes(b"f1-best-unc"),
            (work_dir / "metrics_val.json").write_text(
                json.dumps({"f1": 0.7}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            _make_train_result(train_metrics={"f1": 0.8}),
        )[-1],
    )
    monkeypatch.setattr(
        "forestseg.cli.cmd_run_rl_fusion",
        lambda current_cfg, stage_override=None: (
            (work_dir / "fusion_selected.json").write_text(
                json.dumps(
                    _make_fusion_selected_payload(params=_make_fusion_params(threshold=0.7)),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            ),
            (work_dir / "feature_feedback.json").write_text(
                json.dumps(_make_feedback_payload(threshold=0.7, reward=0.1), ensure_ascii=False, indent=2),
                encoding="utf-8",
            ),
            (work_dir / "rl_history.json").write_text(
                json.dumps(
                    [
                        _make_rl_history_payload(
                            round_no=1,
                            score=0.1,
                            reward=0.1,
                            threshold=0.7,
                            train_metrics={"f1": 0.8},
                            validation_metrics={"reward": 0.1, "f1": 0.7},
                        )
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            ),
            (work_dir / "prob_fused.tif").write_bytes(b"f1-best"),
            _make_rl_result(reward=0.1, f1=0.7, params=_make_fusion_params(threshold=0.7)),
        )[-1],
    )

    summary = cmd_run_closed_loop(cfg)
    persisted_summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    metrics_history = json.loads((work_dir / "metrics_round_history.json").read_text(encoding="utf-8"))
    rl_history = _load_rl_history(str(work_dir / "rl_history.json"))

    assert summary == persisted_summary
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    assert summary["loop"]["selection_metric"] == "f1"
    assert summary["best"]["selection_metric"] == "f1"
    assert summary["best"]["score"] == 0.7
    assert summary["best"]["entry"] == summary["history"][0]
    assert summary["history"][0]["artifacts"] == metrics_history[0]["artifacts"]
    _assert_round_history_schema_consistency(
        summary["history"][0],
        metrics_history[0],
        expected={
            "stage_used": "validation_stage1",
            "threshold": 0.7,
            "checkpoint_path": "ckpt.pt",
            "trained": True,
            "feature_feedback_suffix": "round_01\\feature_feedback.json",
            "fusion_selected_suffix": "round_01\\fusion_selected.json",
            "metrics_val_suffix": "round_01\\metrics_val.json",
            "prob_dl_suffix": "round_01\\prob_dl.tif",
            "unc_dl_suffix": "round_01\\unc_dl.tif",
            "prob_fused_suffix": "round_01\\prob_fused.tif",
            "feature_meta": {},
            "selection_metric": "f1",
            "score": 0.7,
            "reward": 0.1,
        },
    )
    _assert_rl_history_entry(
        summary["history"][0],
        expected={
            "round": 1,
            "selection_metric": "f1",
            "score": 0.7,
            "reward": 0.1,
        },
    )
    assert json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8"))["selection_metric"] == "f1"
    assert json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8"))["score"] == 0.7
    _assert_rl_history_entry(
        rl_history[0],
        expected={
            "round": 1,
            "selection_metric": "f1",
            "score": 0.7,
            "reward": 0.1,
            "validation_f1": 0.7,
        },
    )


def test_cmd_run_rl_fusion_rejects_missing_selected_metric_in_validation_reward(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _write_basic_rl_fusion_inputs(work_dir)
    monkeypatch.setattr("forestseg.cli.save_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr("forestseg.cli.cmd_prepare_input", lambda current_cfg: None)
    _stub_rl_fusion_validation_runtime(monkeypatch)

    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        lambda **kwargs: (
            FusionParams(**_make_fusion_params(threshold=0.6)),
            {"reward": 0.1},
            "validation_stage1",
        ),
    )

    with pytest.raises(ValueError, match="rl_loop.selection_metric"):
        cmd_run_rl_fusion(cfg)


def test_cmd_run_rl_fusion_uses_configured_selection_metric_for_score(
    monkeypatch,
    tmp_path,
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _write_basic_rl_fusion_inputs(work_dir)
    monkeypatch.setattr("forestseg.cli.save_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr("forestseg.cli.cmd_prepare_input", lambda current_cfg: None)
    _stub_rl_fusion_validation_runtime(monkeypatch)

    monkeypatch.setattr(
        "forestseg.cli.choose_fusion_by_validation",
        lambda **kwargs: (
            FusionParams(**_make_fusion_params(threshold=0.6)),
            {"reward": 0.1, "f1": 0.8},
            "validation_stage1",
        ),
    )

    result = cmd_run_rl_fusion(cfg)

    selected_payload = json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8"))
    history_payload = json.loads((work_dir / "rl_history.json").read_text(encoding="utf-8"))

    assert result["score"] == 0.8
    assert selected_payload["selection_metric"] == "f1"
    assert selected_payload["score"] == 0.8
    assert selected_payload["reward"] == 0.1
    assert history_payload[0]["selection_metric"] == "f1"
    assert history_payload[0]["score"] == 0.8
    assert history_payload[0]["reward"] == 0.1


@pytest.mark.parametrize(
    "bad_min_delta",
    [True, None, "oops", float("nan"), float("inf"), -0.001],
)
def test_cmd_run_closed_loop_rejects_invalid_min_delta_without_preflight(monkeypatch, tmp_path, bad_min_delta):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 1, "patience": 2, "min_delta": bad_min_delta})
    cfg["work_dir"] = str(work_dir)

    monkeypatch.setattr(
        "forestseg.cli.cmd_preflight_check",
        lambda current_cfg: {"status": "ok"},
    )
    monkeypatch.setattr("forestseg.cli.cmd_prepare_input", lambda current_cfg: None)

    with pytest.raises(ValueError, match="rl_loop.min_delta"):
        cmd_run_closed_loop(cfg)

    summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    _assert_failed_closed_loop_summary(
        summary,
        expected={
            "stage": "preflight",
            "round": None,
            "error_type": "ValueError",
            "message_contains": "rl_loop.min_delta",
            "rounds_requested": 1,
            "rounds_completed": 0,
            "selection_metric": None,
            "patience": 2,
            "min_delta": None,
            "stopped_early": False,
            "history_empty": True,
            "best_empty": True,
            "restore_outcome": {
                "restored_optional_artifacts": [],
                "failed_optional_artifacts": [],
                "skipped_optional_artifacts": [],
            },
        },
    )
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }


def test_cmd_run_closed_loop_preserves_true_stopped_early_in_failure_summary_after_patience_break(
    monkeypatch, tmp_path
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 3, "patience": 1, "selection_metric": "f1"})
    cfg["work_dir"] = str(work_dir)

    _stub_closed_loop_prereqs(monkeypatch)

    round_payloads = [
        {
            "checkpoint_path": "ckpt-round-1.pt",
            "train_metrics": {"f1": 0.8},
            "metrics_val": {"f1": 0.9, "loss": 0.1},
            "validation_reward": {"reward": 0.1, "f1": 0.9},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.4, lambda_spec=0.1, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.4, reward=0.1, lambda_spec=0.1, lambda_tex=0.0),
            "prob_dl": b"best-round-dl",
            "unc_dl": b"best-round-unc",
            "prob_fused": b"best-round-fused",
            "payload": _make_rl_history_payload(
                round_no=1,
                selection_metric="f1",
                score=0.9,
                reward=0.1,
                threshold=0.4,
                lambda_spec=0.1,
                lambda_tex=0.0,
                train_metrics={"f1": 0.8, "loss": 0.1},
                validation_metrics={"reward": 0.1, "f1": 0.9},
            ),
        },
        {
            "checkpoint_path": "ckpt-round-2.pt",
            "train_metrics": {"f1": 0.3},
            "metrics_val": {"f1": 0.1, "loss": 0.9},
            "validation_reward": {"reward": 0.9, "f1": 0.1},
            "stage_used": "validation_stage1",
            "params": _make_fusion_params(threshold=0.8, lambda_spec=0.9, lambda_tex=0.0),
            "feedback": _make_feedback_payload(threshold=0.8, reward=0.9, lambda_spec=0.9, lambda_tex=0.0),
            "prob_dl": b"live-round-dl",
            "unc_dl": b"live-round-unc",
            "prob_fused": b"live-round-fused",
            "payload": _make_rl_history_payload(
                round_no=2,
                selection_metric="f1",
                score=0.1,
                reward=0.9,
                threshold=0.8,
                lambda_spec=0.9,
                lambda_tex=0.0,
                train_metrics={"f1": 0.3, "loss": 0.9},
                validation_metrics={"reward": 0.9, "f1": 0.1},
            ),
        },
    ]
    current_round = {"value": 0}

    def fake_train_or_load_dl(current_cfg, force_train=False):
        payload = round_payloads[current_round["value"]]
        (work_dir / "metrics_val.json").write_text(
            json.dumps(payload["metrics_val"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_dl.tif").write_bytes(payload["prob_dl"])
        (work_dir / "unc_dl.tif").write_bytes(payload["unc_dl"])
        return _make_train_result(
            checkpoint_path=payload["checkpoint_path"],
            train_metrics=payload["train_metrics"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_train_or_load_dl", fake_train_or_load_dl)

    def fake_run_rl_fusion(current_cfg, stage_override=None):
        payload = round_payloads[current_round["value"]]
        current_round["value"] += 1
        (work_dir / "fusion_selected.json").write_text(
            json.dumps(payload["payload"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "rl_history.json").write_text(
            json.dumps([payload["payload"]], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "feature_feedback.json").write_text(
            json.dumps(payload["feedback"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "prob_fused.tif").write_bytes(payload["prob_fused"])
        return _make_rl_result(
            reward=payload["validation_reward"]["reward"],
            f1=payload["validation_reward"]["f1"],
            params=payload["params"],
            stage_used=payload["stage_used"],
        )

    monkeypatch.setattr("forestseg.cli.cmd_run_rl_fusion", fake_run_rl_fusion)

    original_require_json = __import__("forestseg.cli", fromlist=["_require_json_copy"])._require_json_copy

    def fake_require_json(src, dst, label):
        if label == "best round metrics_val":
            raise ValueError(f"Missing {label}: {src}")
        return original_require_json(src, dst, label)

    monkeypatch.setattr("forestseg.cli._require_json_copy", fake_require_json)

    with pytest.raises(ValueError, match="best round metrics_val"):
        cmd_run_closed_loop(cfg)

    summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    history = json.loads((work_dir / "metrics_round_history.json").read_text(encoding="utf-8"))

    _assert_failed_closed_loop_summary(
        summary,
        expected={
            "stage": "restore_best_round",
            "round": 2,
            "error_type": "ValueError",
            "message_contains": "best round metrics_val",
            "rounds_requested": 3,
            "rounds_completed": 2,
            "selection_metric": "f1",
            "patience": 1,
            "min_delta": 0.001,
            "stopped_early": True,
            "history_len": 2,
            "restore_outcome": {
                "restored_optional_artifacts": [],
                "failed_optional_artifacts": [],
                "skipped_optional_artifacts": ["unc_dl"],
            },
        },
    )
    assert summary["best"]["round"] == 1
    assert summary["best"]["score"] == pytest.approx(0.9)
    assert summary["best"]["reward"] == pytest.approx(0.1)
    assert summary["best"]["entry"] == summary["history"][0]
    assert summary["history"][0]["artifacts"] == history[0]["artifacts"]
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }
    assert set(summary["best"]["entry"]["artifacts"]) == {
        "fusion_selected",
        "rl_payload",
        "feature_feedback",
        "metrics_val",
        "prob_dl",
        "unc_dl",
        "prob_fused",
        "train_metrics",
        "feature_meta",
    }
    _assert_path_suffix(summary["best"]["entry"]["artifacts"]["fusion_selected"], "round_01\\fusion_selected.json")
    _assert_path_suffix(summary["best"]["entry"]["artifacts"]["feature_feedback"], "round_01\\feature_feedback.json")
    _assert_path_suffix(summary["best"]["entry"]["artifacts"]["metrics_val"], "round_01\\metrics_val.json")
    _assert_path_suffix(summary["best"]["entry"]["artifacts"]["unc_dl"], "round_01\\unc_dl.tif")
    assert len(history) == 2
    assert json.loads((work_dir / "fusion_selected.json").read_text(encoding="utf-8"))["params"][
        "threshold"
    ] == pytest.approx(0.8)
    assert json.loads((work_dir / "feature_feedback.json").read_text(encoding="utf-8"))["threshold"] == pytest.approx(
        0.8
    )


def test_cmd_run_closed_loop_rejects_invalid_selection_metric_without_preflight(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(
        tmp_path, rl_overrides={"rounds": 1, "patience": 2, "min_delta": 0.001, "selection_metric": " auc "}
    )
    cfg["work_dir"] = str(work_dir)

    monkeypatch.setattr(
        "forestseg.cli.cmd_preflight_check",
        lambda current_cfg: {"status": "ok"},
    )
    monkeypatch.setattr("forestseg.cli.cmd_prepare_input", lambda current_cfg: None)

    with pytest.raises(ValueError, match="rl_loop.selection_metric"):
        cmd_run_closed_loop(cfg)

    summary = json.loads((work_dir / "closed_loop_summary.json").read_text(encoding="utf-8"))
    _assert_failed_closed_loop_summary(
        summary,
        expected={
            "stage": "preflight",
            "round": None,
            "error_type": "ValueError",
            "message_contains": "rl_loop.selection_metric",
            "rounds_requested": 1,
            "rounds_completed": 0,
            "selection_metric": None,
            "patience": 2,
            "min_delta": 0.001,
            "stopped_early": False,
            "history_empty": True,
            "best_empty": True,
            "restore_outcome": {
                "restored_optional_artifacts": [],
                "failed_optional_artifacts": [],
                "skipped_optional_artifacts": [],
            },
        },
    )
    assert summary["artifacts"] == {
        "fusion_selected": str(work_dir / "fusion_selected.json"),
        "feature_feedback": str(work_dir / "feature_feedback.json"),
        "prob_dl": str(work_dir / "prob_dl.tif"),
        "unc_dl": str(work_dir / "unc_dl.tif"),
        "prob_fused": str(work_dir / "prob_fused.tif"),
        "metrics_val": str(work_dir / "metrics_val.json"),
        "rl_history": str(work_dir / "rl_history.json"),
        "metrics_round_history": str(work_dir / "metrics_round_history.json"),
        "closed_loop_summary": str(work_dir / "closed_loop_summary.json"),
    }


def test_cmd_preflight_check_writes_warning_report(monkeypatch, tmp_path):
    cfg = _make_base_cfg(tmp_path, rl_overrides={"rounds": 2, "patience": 3})
    _stub_scene_input(monkeypatch)
    report = cmd_preflight_check(cfg)

    _assert_report_contract(report, status="warning")
    assert report["warnings"] == ["rl_loop.patience 大于 rounds，早停阈值实际上不会触发。"]
    assert report["errors"] == []
    report_path = tmp_path / "work" / "preflight_check.json"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    _assert_report_contract(saved, status="warning")
    assert saved["warnings"] == ["rl_loop.patience 大于 rounds，早停阈值实际上不会触发。"]
    assert saved["errors"] == []


@pytest.mark.parametrize(
    "command",
    [cmd_prepare_label_points, cmd_check_label_points],
    ids=["prepare", "check"],
)
def test_label_point_commands_reject_invalid_grid_size_without_preflight(monkeypatch, tmp_path, command):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    cfg["labels"]["grid_size"] = float("nan")

    _stub_label_validation_without_preflight(monkeypatch)

    with pytest.raises(ValueError, match="labels.grid_size"):
        command(cfg)


@pytest.mark.parametrize(
    "bad_split_ratio",
    [float("nan"), float("inf")],
)
@pytest.mark.parametrize(
    "command",
    [cmd_prepare_label_points, cmd_check_label_points],
    ids=["prepare", "check"],
)
def test_label_point_commands_reject_invalid_split_ratio_without_preflight(
    monkeypatch, tmp_path, bad_split_ratio, command
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    cfg["labels"]["split_ratio"] = bad_split_ratio

    _stub_label_validation_without_preflight(monkeypatch)

    with pytest.raises(ValueError, match="labels.split_ratio"):
        command(cfg)


@pytest.mark.parametrize(
    "bad_split_seed",
    [-1, -2.5, True, None, "oops"],
)
@pytest.mark.parametrize(
    "command",
    [cmd_prepare_label_points, cmd_check_label_points],
    ids=["prepare", "check"],
)
def test_label_point_commands_reject_invalid_split_seed_without_preflight(
    monkeypatch, tmp_path, bad_split_seed, command
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    cfg["labels"]["split_seed"] = bad_split_seed

    _stub_label_validation_without_preflight(monkeypatch)

    with pytest.raises(ValueError, match="labels.split_seed"):
        command(cfg)


def test_cmd_prepare_label_points_rejects_blocking_label_validation(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)

    _stub_label_validation_without_preflight(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.validate_label_points_from_two_files",
        lambda **kwargs: {
            "risk": {
                "status": "block",
                "can_run": False,
                "warnings": [],
                "blocking": ["缺少源 CRS"],
            }
        },
    )
    monkeypatch.setattr(
        "forestseg.cli.read_label_points_from_two_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("read_label_points_from_two_files should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.export_points_preview",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("export_points_preview should not run")),
    )

    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    with pytest.raises(ValueError, match="缺少源 CRS"):
        cmd_prepare_label_points(cfg)


def test_cmd_prepare_label_points_writes_split_outputs(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)

    _stub_label_validation_without_preflight(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.validate_label_points_from_two_files",
        lambda **kwargs: {
            "risk": {
                "status": "warning",
                "can_run": True,
                "warnings": ["sparse"],
                "blocking": [],
            }
        },
    )
    monkeypatch.setattr(
        "forestseg.cli.read_label_points_from_two_files",
        lambda **kwargs: [
            type("P", (), {"x": 0.0, "y": 0.0, "label": 1, "grid_id": "0_0", "properties": {}})(),
            type("P", (), {"x": 2000.0, "y": 0.0, "label": 0, "grid_id": "2_0", "properties": {}})(),
        ],
    )
    monkeypatch.setattr(
        "forestseg.cli.export_points_preview",
        lambda path, points, crs, layer="labels": path,
    )

    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    result = cmd_prepare_label_points(cfg)
    train_points, train_meta = load_points_json(result["samples_train"])
    val_points, val_meta = load_points_json(result["samples_val"])

    assert (work_dir / "samples_train.json").exists()
    assert (work_dir / "samples_val.json").exists()
    _assert_path_suffix(result["samples_train"], str(work_dir / "samples_train.json"))
    _assert_path_suffix(result["samples_val"], str(work_dir / "samples_val.json"))
    assert len(train_points) == 1
    assert len(val_points) == 1
    assert {point["grid_id"] for point in train_points}.isdisjoint({point["grid_id"] for point in val_points})
    assert train_meta["split_possible"] is True
    assert val_meta["split_possible"] is True
    for key in (
        "train_ratio",
        "seed",
        "grid_count",
        "train_grid_count",
        "val_grid_count",
        "train_points",
        "val_points",
    ):
        assert train_meta[key] == val_meta[key]


def test_cmd_check_label_points_writes_top_level_contract(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)

    _stub_label_validation_without_preflight(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.validate_label_points_from_two_files",
        lambda **kwargs: {
            "path": str(tmp_path / "labels.gpkg"),
            "risk": {
                "status": "warning",
                "can_run": True,
                "warnings": ["w1"],
                "blocking": [],
            },
        },
    )

    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    cfg["dl"] = {}

    report = cmd_check_label_points(cfg)

    _assert_report_contract(report, status="warning")
    assert report["warnings"] == ["w1"]
    assert report["errors"] == []
    report_path = tmp_path / "work" / "label_check_report.json"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    _assert_report_contract(saved, status="warning")
    assert saved["warnings"] == ["w1"]
    assert saved["errors"] == []


def test_cmd_check_label_points_writes_blocking_top_level_contract(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)

    _stub_label_validation_without_preflight(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.validate_label_points_from_two_files",
        lambda **kwargs: {
            "path": str(tmp_path / "labels.gpkg"),
            "risk": {
                "status": "block",
                "can_run": False,
                "warnings": ["w1"],
                "blocking": ["e1", "e2"],
            },
        },
    )

    cfg = _make_base_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    cfg["dl"] = {}

    report = cmd_check_label_points(cfg)

    _assert_report_contract(report, status="block", can_run=False)
    assert report["warnings"] == ["w1"]
    assert report["errors"] == ["e1", "e2"]
    report_path = tmp_path / "work" / "label_check_report.json"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    _assert_report_contract(saved, status="block", can_run=False)
    assert saved["warnings"] == ["w1"]
    assert saved["errors"] == ["e1", "e2"]


def _make_legacy_label_cfg(tmp_path):
    cfg = _make_base_cfg(tmp_path)
    legacy_label_file = tmp_path / "labels_legacy.gpkg"
    legacy_label_file.write_text("placeholder", encoding="utf-8")
    cfg["labels"] = {
        "path": str(legacy_label_file),
        "split_ratio": 0.7,
        "grid_size": 1000.0,
    }
    return cfg


def test_cmd_preflight_check_accepts_legacy_labels_path(monkeypatch, tmp_path):
    cfg = _make_legacy_label_cfg(tmp_path)
    _stub_scene_input(monkeypatch)

    report = cmd_preflight_check(cfg)

    assert report["checks"]["labels_path"] == cfg["labels"]["path"]


def test_cmd_prepare_label_points_uses_legacy_labels_path_fallback(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_legacy_label_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _stub_label_validation_without_preflight(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.validate_label_points",
        lambda **kwargs: {"risk": {"status": "ok", "can_run": True, "warnings": [], "blocking": []}},
    )
    monkeypatch.setattr(
        "forestseg.cli.read_label_points",
        lambda **kwargs: [
            type("P", (), {"x": 0.0, "y": 0.0, "label": 1, "grid_id": "0_0", "properties": {}})(),
            type("P", (), {"x": 2000.0, "y": 0.0, "label": 0, "grid_id": "2_0", "properties": {}})(),
        ],
    )
    monkeypatch.setattr(
        "forestseg.cli.validate_label_points_from_two_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dual-file validator should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.read_label_points_from_two_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dual-file reader should not run")),
    )
    monkeypatch.setattr(
        "forestseg.cli.export_points_preview",
        lambda path, points, crs, layer="labels": path,
    )

    result = cmd_prepare_label_points(cfg)

    assert os.path.exists(result["samples_train"])
    assert os.path.exists(result["samples_val"])


def test_cmd_check_label_points_uses_legacy_labels_path_fallback(monkeypatch, tmp_path):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_legacy_label_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)

    _stub_label_validation_without_preflight(monkeypatch)
    monkeypatch.setattr(
        "forestseg.cli.validate_label_points",
        lambda **kwargs: {
            "path": str(tmp_path / "labels_legacy.gpkg"),
            "risk": {"status": "ok", "can_run": True, "warnings": [], "blocking": []},
        },
    )
    monkeypatch.setattr(
        "forestseg.cli.validate_label_points_from_two_files",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dual-file validator should not run")),
    )

    report = cmd_check_label_points(cfg)

    _assert_report_contract(report, status="ok", can_run=True)


@pytest.mark.parametrize("only_key", ["positive_path", "negative_path"])
def test_cmd_preflight_check_rejects_partial_dual_label_paths_even_with_legacy_path(monkeypatch, tmp_path, only_key):
    cfg = _make_legacy_label_cfg(tmp_path)
    extra_label_file = tmp_path / f"{only_key}.gpkg"
    extra_label_file.write_text("placeholder", encoding="utf-8")
    cfg["labels"][only_key] = str(extra_label_file)

    _stub_scene_input(monkeypatch)

    with pytest.raises(ValueError, match="必须同时配置"):
        cmd_preflight_check(cfg)


@pytest.mark.parametrize("only_key", ["positive_path", "negative_path"])
def test_cmd_prepare_label_points_rejects_partial_dual_label_paths_even_with_legacy_path(
    monkeypatch, tmp_path, only_key
):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_legacy_label_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    extra_label_file = tmp_path / f"{only_key}.gpkg"
    extra_label_file.write_text("placeholder", encoding="utf-8")
    cfg["labels"][only_key] = str(extra_label_file)

    _stub_label_validation_without_preflight(monkeypatch)

    with pytest.raises(ValueError, match="必须同时配置"):
        cmd_prepare_label_points(cfg)


@pytest.mark.parametrize("only_key", ["positive_path", "negative_path"])
def test_cmd_check_label_points_rejects_partial_dual_label_paths_even_with_legacy_path(monkeypatch, tmp_path, only_key):
    work_dir = _make_prepared_workdir(tmp_path)
    cfg = _make_legacy_label_cfg(tmp_path)
    cfg["work_dir"] = str(work_dir)
    extra_label_file = tmp_path / f"{only_key}.gpkg"
    extra_label_file.write_text("placeholder", encoding="utf-8")
    cfg["labels"][only_key] = str(extra_label_file)

    _stub_label_validation_without_preflight(monkeypatch)

    with pytest.raises(ValueError, match="必须同时配置"):
        cmd_check_label_points(cfg)
