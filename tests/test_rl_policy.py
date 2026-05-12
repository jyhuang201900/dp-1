import copy

import numpy as np
import pytest

from forestseg.fusion.policy import choose_fusion, choose_fusion_by_validation

FIXED_PARAMS = {
    "lambda_spec": 0.2,
    "lambda_tex": 0.2,
    "threshold": 0.5,
    "min_area_m2": 200,
    "morph_kernel": 3,
    "shadow_penalty": 0.5,
}

GRID_PARAMS = {
    "lambda_spec": [0.1, 0.2],
    "lambda_tex": [0.1, 0.2],
    "threshold": [0.4, 0.5],
    "min_area_m2": [100],
    "morph_kernel": [3],
    "shadow_penalty": [0.5],
}


@pytest.mark.parametrize("bad_stage", [-1, 2, 3])
def test_choose_fusion_rejects_legacy_or_out_of_contract_stages(monkeypatch, bad_stage):
    monkeypatch.setattr(
        "forestseg.fusion.policy.run_stage0",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_stage0 should not run")),
    )
    monkeypatch.setattr(
        "forestseg.fusion.policy.run_stage1_grid",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_stage1_grid should not run")),
    )

    with pytest.raises(ValueError, match="stage must be 0 or 1"):
        choose_fusion(
            stage=bad_stage,
            prob_dl=np.zeros((1, 1), dtype=np.float32),
            prob_spec=np.zeros((1, 1), dtype=np.float32),
            prob_tex=np.zeros((1, 1), dtype=np.float32),
            shadow_score=np.zeros((1, 1), dtype=np.float32),
            fixed_params=FIXED_PARAMS,
            grid_params=GRID_PARAMS,
        )


@pytest.mark.parametrize("bad_stage", [-1, 2, 3])
def test_choose_fusion_by_validation_rejects_legacy_or_out_of_contract_stages(monkeypatch, bad_stage):
    monkeypatch.setattr(
        "forestseg.fusion.policy.sample_raster_at_points",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sample_raster_at_points should not run")),
    )

    with pytest.raises(ValueError, match="stage must be 0 or 1"):
        choose_fusion_by_validation(
            stage=bad_stage,
            prob_dl_path="dl.tif",
            prob_spec_path="spec.tif",
            prob_tex_path="tex.tif",
            val_points=[{"label": 1}, {"label": 0}],
            fixed_params=FIXED_PARAMS,
            grid_params=GRID_PARAMS,
        )


def test_choose_fusion_by_validation_falls_back_when_no_valid_points(monkeypatch):
    def fake_sample(*args, **kwargs):
        return [float("nan"), float("nan")]

    monkeypatch.setattr("forestseg.fusion.policy.sample_raster_at_points", fake_sample)
    val_points = [{"label": 1}, {"label": 0}]

    candidate, metrics, stage = choose_fusion_by_validation(
        stage=0,
        prob_dl_path="a.tif",
        prob_spec_path="b.tif",
        prob_tex_path="c.tif",
        val_points=val_points,
        fixed_params=FIXED_PARAMS,
        grid_params=GRID_PARAMS,
    )

    assert stage == "validation_fallback_fixed"
    assert candidate.threshold == FIXED_PARAMS["threshold"]
    assert metrics["threshold"] == FIXED_PARAMS["threshold"]
    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["confusion_matrix"] == {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    assert metrics["count"] == 0
    assert metrics["reward"] == 0.0
    assert metrics["valid_points_used"] == 0


def test_choose_fusion_by_validation_falls_back_when_no_valid_points_for_stage1(monkeypatch):
    def fake_sample(*args, **kwargs):
        return [float("nan"), float("nan")]

    monkeypatch.setattr("forestseg.fusion.policy.sample_raster_at_points", fake_sample)

    candidate, metrics, stage = choose_fusion_by_validation(
        stage=1,
        prob_dl_path="dl.tif",
        prob_spec_path="spec.tif",
        prob_tex_path="tex.tif",
        val_points=[{"label": 1}, {"label": 0}],
        fixed_params=FIXED_PARAMS,
        grid_params=GRID_PARAMS,
    )

    assert stage == "validation_fallback_fixed"
    assert candidate.lambda_spec == FIXED_PARAMS["lambda_spec"]
    assert candidate.lambda_tex == FIXED_PARAMS["lambda_tex"]
    assert candidate.threshold == FIXED_PARAMS["threshold"]
    assert candidate.min_area_m2 == FIXED_PARAMS["min_area_m2"]
    assert candidate.morph_kernel == FIXED_PARAMS["morph_kernel"]
    assert candidate.shadow_penalty == FIXED_PARAMS["shadow_penalty"]
    assert metrics["count"] == 0
    assert metrics["valid_points_used"] == 0


def test_choose_fusion_by_validation_rejects_empty_grid_dimension(monkeypatch):
    def fake_sample(path, points):
        return [0.9, 0.1]

    monkeypatch.setattr("forestseg.fusion.policy.sample_raster_at_points", fake_sample)
    val_points = [{"label": 1}, {"label": 0}]
    invalid_grid_params = copy.deepcopy(GRID_PARAMS)
    invalid_grid_params["threshold"] = []

    with pytest.raises(ValueError, match="grid_params.threshold"):
        choose_fusion_by_validation(
            stage=1,
            prob_dl_path="dl.tif",
            prob_spec_path="spec.tif",
            prob_tex_path="tex.tif",
            val_points=val_points,
            fixed_params=FIXED_PARAMS,
            grid_params=invalid_grid_params,
        )


def test_choose_fusion_by_validation_rejects_missing_grid_dimension(monkeypatch):
    def fake_sample(path, points):
        return [0.9, 0.1]

    monkeypatch.setattr("forestseg.fusion.policy.sample_raster_at_points", fake_sample)
    val_points = [{"label": 1}, {"label": 0}]
    invalid_grid_params = copy.deepcopy(GRID_PARAMS)
    invalid_grid_params.pop("threshold")

    with pytest.raises(ValueError, match="grid_params.threshold"):
        choose_fusion_by_validation(
            stage=1,
            prob_dl_path="dl.tif",
            prob_spec_path="spec.tif",
            prob_tex_path="tex.tif",
            val_points=val_points,
            fixed_params=FIXED_PARAMS,
            grid_params=invalid_grid_params,
        )


def test_choose_fusion_by_validation_rejects_scalar_grid_dimension(monkeypatch):
    def fake_sample(path, points):
        return [0.9, 0.1]

    monkeypatch.setattr("forestseg.fusion.policy.sample_raster_at_points", fake_sample)
    val_points = [{"label": 1}, {"label": 0}]
    invalid_grid_params = copy.deepcopy(GRID_PARAMS)
    invalid_grid_params["threshold"] = 0.5

    with pytest.raises(ValueError, match="grid_params.threshold"):
        choose_fusion_by_validation(
            stage=1,
            prob_dl_path="dl.tif",
            prob_spec_path="spec.tif",
            prob_tex_path="tex.tif",
            val_points=val_points,
            fixed_params=FIXED_PARAMS,
            grid_params=invalid_grid_params,
        )


def test_choose_fusion_by_validation_rejects_invalid_grid_when_no_valid_points(monkeypatch):
    def fake_sample(*args, **kwargs):
        return [float("nan"), float("nan")]

    monkeypatch.setattr("forestseg.fusion.policy.sample_raster_at_points", fake_sample)
    invalid_grid_params = copy.deepcopy(GRID_PARAMS)
    invalid_grid_params["threshold"] = []

    with pytest.raises(ValueError, match="grid_params.threshold"):
        choose_fusion_by_validation(
            stage=1,
            prob_dl_path="dl.tif",
            prob_spec_path="spec.tif",
            prob_tex_path="tex.tif",
            val_points=[{"label": 1}, {"label": 0}],
            fixed_params=FIXED_PARAMS,
            grid_params=invalid_grid_params,
        )


def test_choose_fusion_by_validation_uses_fixed_postprocess_params_for_validation_selection(monkeypatch):
    sample_values = {
        "dl.tif": [0.2, 0.2, 0.8, 0.8],
        "spec.tif": [0.9, 0.9, 0.1, 0.1],
        "tex.tif": [0.2, 0.2, 0.8, 0.8],
    }

    def fake_sample(path, points):
        return sample_values[path]

    monkeypatch.setattr("forestseg.fusion.policy.sample_raster_at_points", fake_sample)

    candidate, metrics, stage = choose_fusion_by_validation(
        stage=1,
        prob_dl_path="dl.tif",
        prob_spec_path="spec.tif",
        prob_tex_path="tex.tif",
        val_points=[{"label": 0}, {"label": 0}, {"label": 1}, {"label": 1}],
        fixed_params=FIXED_PARAMS,
        grid_params={
            "lambda_spec": [0.1, 0.9],
            "lambda_tex": [0.0],
            "threshold": [0.5],
            "min_area_m2": [100.0, 500.0],
            "morph_kernel": [1, 7],
            "shadow_penalty": [0.1, 0.9],
        },
    )

    assert stage == "validation_stage1"
    assert candidate.lambda_spec == 0.1
    assert candidate.min_area_m2 == FIXED_PARAMS["min_area_m2"]
    assert candidate.morph_kernel == FIXED_PARAMS["morph_kernel"]
    assert candidate.shadow_penalty == FIXED_PARAMS["shadow_penalty"]
    assert metrics["valid_points_used"] == 4


def test_choose_fusion_by_validation_falls_back_to_grid_postprocess_params_when_fixed_missing(monkeypatch):
    sample_values = {
        "dl.tif": [0.2, 0.2, 0.8, 0.8],
        "spec.tif": [0.9, 0.9, 0.1, 0.1],
        "tex.tif": [0.2, 0.2, 0.8, 0.8],
    }

    def fake_sample(path, points):
        return sample_values[path]

    monkeypatch.setattr("forestseg.fusion.policy.sample_raster_at_points", fake_sample)

    candidate, metrics, stage = choose_fusion_by_validation(
        stage=1,
        prob_dl_path="dl.tif",
        prob_spec_path="spec.tif",
        prob_tex_path="tex.tif",
        val_points=[{"label": 0}, {"label": 0}, {"label": 1}, {"label": 1}],
        fixed_params={
            "lambda_spec": 0.2,
            "lambda_tex": 0.2,
            "threshold": 0.5,
        },
        grid_params={
            "lambda_spec": [0.1, 0.9],
            "lambda_tex": [0.0],
            "threshold": [0.5],
            "min_area_m2": [100.0, 500.0],
            "morph_kernel": [1, 7],
            "shadow_penalty": [0.1, 0.9],
        },
    )

    assert stage == "validation_stage1"
    assert candidate.lambda_spec == 0.1
    assert candidate.min_area_m2 == 100.0
    assert candidate.morph_kernel == 1
    assert candidate.shadow_penalty == 0.1
    assert metrics["valid_points_used"] == 4
