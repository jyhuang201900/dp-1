# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
