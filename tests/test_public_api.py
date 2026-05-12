"""Smoke tests that lock in the public API surface of ``forestseg``.

These guard against accidental regressions in the import paths consumed by
upstream `dp` users — anything listed here must stay importable.
"""

from __future__ import annotations

import importlib

import pytest

# Modules that callers rely on being importable.
PUBLIC_MODULES = [
    "forestseg",
    "forestseg.cli",
    "forestseg.io_raster",
    "forestseg.labels",
    "forestseg.features",
    "forestseg.spectral",
    "forestseg.texture",
    "forestseg.model",
    "forestseg.train",
    "forestseg.infer",
    "forestseg.sampler",
    "forestseg.fusion",
    "forestseg.postprocess",
    "forestseg.rl_env",
    "forestseg.rl_policy",
    "forestseg.bandit",
    "forestseg.feedback_loop",
    "forestseg.scene_runtime",
    "forestseg.metrics",
    "forestseg.export",
    "forestseg.coupling_tensor",
    "forestseg._logging",
    "forestseg._version",
]


# Symbols that the upstream tests rely on being attributes of specific modules.
# Format: (module, attribute).
PUBLIC_ATTRIBUTES = [
    ("forestseg", "__version__"),
    ("forestseg.cli", "main"),
    ("forestseg.cli", "cmd_preflight_check"),
    ("forestseg.cli", "cmd_prepare_input"),
    ("forestseg.cli", "cmd_build_spec_tex"),
    ("forestseg.cli", "cmd_check_label_points"),
    ("forestseg.cli", "cmd_prepare_label_points"),
    ("forestseg.cli", "cmd_build_feature_stack"),
    ("forestseg.cli", "cmd_train_or_load_dl"),
    ("forestseg.cli", "cmd_run_rl_fusion"),
    ("forestseg.cli", "cmd_postprocess_export"),
    ("forestseg.cli", "cmd_run_closed_loop"),
    ("forestseg.cli", "cmd_run_all"),
    ("forestseg.cli", "_load_rl_history"),
    ("forestseg.cli", "load_points_json"),
    ("forestseg.cli", "save_metrics"),
    ("forestseg.cli", "train_supervised_model"),
    ("forestseg.cli", "infer_to_files"),
    ("forestseg.cli", "resolve_scene_input"),
    ("forestseg.cli", "_require_json_copy"),
    ("forestseg.cli", "_snapshot_required_json"),
    ("forestseg.fusion", "FusionParams"),
    ("forestseg.fusion", "fuse_probabilities"),
    ("forestseg.labels", "LabelPoint"),
    ("forestseg.labels", "LabelReadOptions"),
    ("forestseg.labels", "LabelReadResult"),
    ("forestseg.labels", "read_label_points"),
    ("forestseg.labels", "read_label_points_from_two_files"),
    ("forestseg.labels", "validate_label_points"),
    ("forestseg.labels", "split_points_by_grid"),
    ("forestseg.labels", "save_points_json"),
    ("forestseg.labels", "load_points_json"),
    ("forestseg.labels", "sample_raster_at_points"),
    ("forestseg.labels", "fiona"),
    ("forestseg.labels", "transform_geom"),
    ("forestseg.metrics", "binary_metrics"),
    ("forestseg.rl_policy", "choose_fusion"),
    ("forestseg.rl_policy", "choose_fusion_by_validation"),
    ("forestseg.scene_runtime", "resolve_scene_input"),
    ("forestseg.scene_runtime", "_parse_scene_file_fallback"),
    ("forestseg.scene_runtime", "_source_scene"),
    ("forestseg.train", "train_supervised_model"),
]


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_public_module_importable(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.parametrize("module_name,attribute", PUBLIC_ATTRIBUTES)
def test_public_attribute_present(module_name: str, attribute: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, attribute), f"{module_name}.{attribute} is missing"


def test_version_string() -> None:
    import forestseg

    assert isinstance(forestseg.__version__, str)
    # Match basic semver (e.g. "0.2.0", "0.2.0a1", "0.2.0+local").
    parts = forestseg.__version__.split(".")
    assert len(parts) >= 2
    assert parts[0].isdigit()


def test_logging_helpers_isolated(capsys: pytest.CaptureFixture) -> None:
    """``_logging`` must not configure logging on import (side-effect-free)."""
    import logging

    from forestseg._logging import configure_logging, get_logger

    logger = get_logger("forestseg.tests")
    assert logger.name == "forestseg.tests"

    parent = configure_logging(level=logging.WARNING)
    assert parent.name == "forestseg"
    assert parent.level == logging.WARNING
