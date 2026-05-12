# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- CI lint job: sort imports in `tests/test_public_api.py` so `ruff check` passes.
- CI test job: three `test_cmd_run_rl_fusion_*` tests no longer regress.
  - `_validate_rl_history_entry` now rejects empty-dict entries (`[{}]`) at
    load time rather than silently filling in legacy defaults; the empty
    dict carries no information and was always meant to be invalid.
  - The `_write_basic_rl_fusion_inputs` test helper now also seeds
    `input_prepared.tif` so the `cmd_run_rl_fusion` lazy-build branch is
    short-circuited the same way it is for the other intermediate inputs.

### Added

- PEP 561 `py.typed` marker so downstream consumers pick up type hints.
- CI now also runs on pushes to `refactor/**` branches, not just `main`.

### Changed

- Bump `ruff-pre-commit` to `v0.15.12` so the pinned hook understands
  the `RUF043` selector already used in `pyproject.toml`.
- `configs/pipeline.windows.yaml`: trim trailing blank line (end-of-file-fixer).
- Shrink `tool.mypy.disable_error_code` from 19 codes down to 3
  (`import-untyped`, `import-not-found`, `arg-type`). The remaining
  codes are the only ones that still fire on the tree; the easy ones
  (`var-annotated`, `no-redef`, `assignment`, `operator`) were
  addressed inline by annotating a handful of NumPy locals, resolving
  a shadowed name in `scene_runtime.py`, and narrowing `metrics[...]`
  reads in `rl_policy.py` before the `+` reductions. Mypy is
  meaningfully stricter now without touching public-facing types.
- Modularize `forestseg.cli` by extracting cohesive blocks into a set
  of small private sibling modules. Each new module has a
  module-level docstring and an explicit `__all__`. Every moved name
  is re-exported from `forestseg.cli` via an `import X as X`
  re-export, so the public attribute surface (`forestseg.cli.X`),
  `test_public_api.PUBLIC_ATTRIBUTES`, and the existing test imports
  are unchanged.
  - `forestseg._constants` — `WORK_FILES`, `REQUIRED_FUSION_GRID_KEYS`,
    `REQUIRED_FUSION_FIXED_PARAM_KEYS`, and the `wf()` path helper.
  - `forestseg._paths` — `load_cfg`, `ensure_work`, `_log_progress`,
    `_require_existing_path`, `_ensure_directory_writable`.
  - `forestseg._validators` — the nine pure scalar validators
    (`_validate_ratio` / `_validate_positive_*` / `_validate_choice`
    / `_validate_unit_interval` / `_validate_non_negative_*` /
    `_validate_stage_int` / `_validate_bool`).
  - `forestseg._fusion_config` — `_resolve_fusion_stage`,
    `_validate_fusion_config`, `_validate_fusion_param_value`,
    `_validate_fusion_params`,
    `_validate_fusion_param_candidates`.
  - `forestseg._bandit_config` — `_resolve_bandit_config`.
  - `forestseg._label_resolution` — `_resolve_label_paths`,
    `_resolve_label_mode`, `_require_labels_for_preflight`.
  - `forestseg._artifacts` — `_round_artifact_paths`,
    `RestoreOutputsError`, `_optional_restore_outcome`. The on-disk
    copy / snapshot / `_restore_outputs_atomically` helpers are kept
    in `forestseg.cli` so the test suite's
    `monkeypatch.setattr("forestseg.cli._require_json_copy", ...)`
    style hooks continue to reach the actual call sites.
  - `forestseg._feature_feedback` — `_feature_feedback_transforms`.
  - `forestseg._rl_history` — `rl_history.json` IO + validation
    (`_normalize_legacy_rl_history_entry`,
    `_validate_rl_history_entry`, `_load_rl_history`,
    `_history_entry`, `_rewrite_latest_rl_history_entry` plus the
    `LEGACY_RL_HISTORY_FUSION_PARAM_DEFAULTS` and
    `VALID_*_SELECTION_METRICS` constants).
- `cli.py` shrinks from ~1840 to ~1290 lines and is now focused on
  argparse plumbing and command handlers.
- Add `__all__` to `forestseg._logging`, `forestseg._validators`, and
  `forestseg._rl_history` for consistency with the new modules.

### Tests

- `tests/test_rl_history.py`: new direct unit-test module covering
  `_load_rl_history` and `_validate_rl_history_entry`
  (12 cases — happy path, malformed JSON, non-list payload, non-dict
  entry, empty-dict entry, unknown selection-metric downgrade, negative
  reward, mismatched top-level reward, legacy payload normalization).
  The existing integration coverage in `test_cli_preflight.py` is
  unchanged.

## [0.2.0] — 2025-05-12 — `dp-1` refactor

This release marks the migration from
[`jyhuang201900/dp`](https://github.com/jyhuang201900/dp) to the standardized
`dp-1` repository. The pipeline algorithms and CLI surface are unchanged; all
of the work in this release is structural / quality.

### Added

- **PEP 621 packaging**: full `pyproject.toml` with `hatchling` backend,
  optional `[torch]`, `[rl]`, `[test]`, `[dev]` extras, and a console-script
  entry point (`forestseg = forestseg.cli:main`).
- **`python -m forestseg`** entry point (`src/forestseg/__main__.py`).
- **Single source-of-truth version** in `src/forestseg/_version.py`,
  re-exported as `forestseg.__version__`.
- **Structured logging helpers** in `src/forestseg/_logging.py`
  (`configure_logging`, `get_logger`) — opt-in, non-breaking.
- **Tooling**:
  - `ruff` (lint + format) configured in `pyproject.toml`.
  - `mypy` configured with a lenient baseline.
  - `pytest` configured with marker / warning policy.
  - `pre-commit` hooks (ruff, hygiene checks).
  - GitHub Actions CI matrix on Python 3.10 / 3.11 / 3.12.
  - `Makefile` with `install / lint / format / typecheck / test / ci` targets.
- **Docs**:
  - English `README.md` and Chinese `README.zh-CN.md`.
  - `docs/architecture.md` — module dependency diagram & rationale.
  - `docs/pipeline.md` — end-to-end stage-by-stage walkthrough.
  - `docs/labeling_workflow.md` — QGIS labeling SOP (migrated from
    `标准的打标签流程.md`).
  - `CONTRIBUTING.md`.
- **Cross-platform config**: `configs/pipeline.example.yaml` with relative
  paths suitable for Linux / macOS / Windows; original Windows layout
  preserved as `configs/pipeline.windows.yaml`.
- **MIT `LICENSE`**, `.editorconfig`, `.gitignore`.

### Changed

- Source code is now installed from `src/forestseg/` (`src/` layout)
  rather than imported from a loose package root.
- Cleaned up a handful of redundant `int(round(...))` and explicit-`strict`
  `zip()` call sites flagged by `ruff` (no semantic change).
- Replaced ad-hoc dependency text files with `pyproject.toml` extras.

### Compatibility

- All public CLI commands, YAML keys, and Python import paths
  (`from forestseg.cli import …`, `from forestseg.labels import …`, etc.)
  are 100% backwards compatible.
- Full pytest suite passes with parity: **338 passed, 3 known-flaky**
  (identical to upstream `dp`).

[Unreleased]: https://github.com/jyhuang201900/dp-1/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jyhuang201900/dp-1/releases/tag/v0.2.0
