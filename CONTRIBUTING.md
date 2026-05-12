# Contributing to dp-1 / `forestseg`

Thanks for your interest in improving this project! This document captures the
conventions we expect contributors (and CI) to follow.

## 1. Set up the dev environment

```bash
git clone https://github.com/jyhuang201900/dp-1.git
cd dp-1
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
make install-dev                         # editable install + dev + test + torch (CPU)
pre-commit install                       # enable git hooks
```

`make install-dev` is equivalent to:

```bash
pip install -e ".[dev,torch,rl]" --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Workflow

1. Open an issue if your change is non-trivial (>~30 lines, new feature, or
   user-visible behaviour change) before starting work.
2. Branch off `main`. Use a descriptive name, e.g.
   `feature/<short-slug>` or `fix/<short-slug>`.
3. Keep commits small and focused. Conventional Commit prefixes (`feat:`,
   `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:`) are encouraged but
   not enforced.
4. Run `make ci` locally before pushing.
5. Open a Pull Request against `main`. The CI workflow on GitHub Actions
   will run `ruff` + `pytest` on Python 3.10 / 3.11 / 3.12.

## 3. Coding standards

- **Formatting**: `ruff format` (line length 120, double quotes).
- **Linting**: `ruff check` must be clean.
- **Imports**: sorted by `ruff`'s isort profile. `forestseg.*` is the
  first-party prefix.
- **Typing**: prefer modern PEP 604 syntax (`int | None` over
  `Optional[int]`). `mypy src/forestseg` must remain green at the current
  lenient configuration.
- **Docstrings**: short imperative summaries; full reST/NumPy/Google docstrings
  are optional but encouraged for new public APIs.
- **Logging**: prefer `forestseg._logging.get_logger(__name__)` over
  bare `print` for new code. Legacy `print` calls are kept for now to
  preserve CLI output parity.

## 4. Tests

- Place unit tests under `tests/` mirroring the module layout when possible
  (`tests/test_<module>.py`).
- Run the suite with `make test` (or `pytest`).
- New behaviour must come with a regression test.
- The current baseline is **338 passed, 3 known-flaky** (the three failing
  tests rely on real raster fixtures absent from this repository; they also
  fail upstream and are not regressions). Do not increase the failure count.

## 5. Modifying the CLI

The CLI subcommand surface and the YAML keys consumed by each command are
considered public API. Adding a new flag is fine; renaming or removing flags
requires a major-version bump and a `CHANGELOG.md` entry.

When adding a new pipeline stage:

1. Add a `cmd_<stage>(cfg)` handler in `src/forestseg/cli.py`.
2. Wire it into `_build_parser` with descriptive `--help` text.
3. Add an entry to `scripts/` if you want a shell wrapper.
4. Document it in `README.md` and `docs/pipeline.md`.
5. Add unit tests covering the happy path + at least two failure modes.

## 6. Pull-request checklist

- [ ] `make ci` is green locally.
- [ ] New behaviour is covered by tests.
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`.
- [ ] Docs updated if a public flag, YAML key, or output artifact changed.
- [ ] PR description explains the problem, the solution, and any
      backwards-compat considerations.

## 7. Reporting bugs / requesting features

Use the GitHub issue tracker at
<https://github.com/jyhuang201900/dp-1/issues>. Please include:

- A minimal reproduction (config snippet + command).
- Expected vs. actual behaviour.
- Environment: OS, Python version, `pip freeze` of the relevant venv.
