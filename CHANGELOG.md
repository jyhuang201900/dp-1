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
- Shrink `tool.mypy.disable_error_code` from 19 codes down to 4
  (`import-untyped`, `import-not-found`, `arg-type`, `operator`).
  The remaining codes are the only ones that still fire on the tree;
  the easy ones (`var-annotated`, `no-redef`, `assignment`) were
  addressed inline by annotating a handful of NumPy locals and
  resolving a shadowed name in `scene_runtime.py`. Mypy is meaningfully
  stricter now without touching public-facing types.

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
