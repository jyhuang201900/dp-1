# dp-1 / `forestseg`

> **KH-4 supervised forest extraction pipeline — refactored & standardized.**
> ![CI](https://github.com/jyhuang201900/dp-1/actions/workflows/ci.yml/badge.svg)
> 中文版本：see [README.zh-CN.md](README.zh-CN.md)

`dp-1` is a modular, end-to-end remote-sensing pipeline that turns raw KH-4
satellite imagery + a handful of manually-labelled points into a clean forest /
non-forest mask, with both raster (`.tif`) and vector (`.gpkg`) outputs.

It is the standardized successor of [`jyhuang201900/dp`](https://github.com/jyhuang201900/dp):
same algorithms, but with proper Python packaging, linting / typing / pre-commit
tooling, CI, cross-platform configuration, and structured documentation.

---

## Pipeline overview

```
raw raster + scene.sh + label .gpkg + pipeline.yaml
    │
    ▼
[preprocess]                normalize, nodata, percentile clip
    │
    ▼
[spectral prior] + [texture prior]   Otsu + quantile + local stats / LBP
    │
    ▼
[label QA]                  geometry + CRS + class-balance checks
    │
    ▼
[grid-aware split]          spatial train / val to limit leakage
    │
    ▼
[feature stack]             aligned multi-band TIFF (+ optional DEM)
    │
    ▼
[DL training & inference]   DeepLabV3+ (ResNet-34 by default), MC dropout
    │
    ▼
[fusion]                    DL ⊕ spectral ⊕ texture, grid / RL search
    │
    ▼
[postprocess]               morphology, small-component removal, shadow penalty
    │
    ▼
[export]                    final mask raster + clean vector polygons
```

Full design notes: [`docs/pipeline.md`](docs/pipeline.md) and
[`docs/architecture.md`](docs/architecture.md).
Labeling SOP: [`docs/labeling_workflow.md`](docs/labeling_workflow.md).

---

## Requirements

- Python **3.10 – 3.12**
- A C compiler + GDAL system libraries (provided by the `rasterio` / `fiona`
  wheels on Linux & macOS; on Windows install via conda or `osgeo4w`)
- Optional: NVIDIA GPU with CUDA ≥ 12.1 for training

---

## Installation

### Quick start (CPU)

```bash
git clone https://github.com/jyhuang201900/dp-1.git
cd dp-1
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e ".[torch,rl,test]" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

### GPU (CUDA 12.1)

```bash
pip install -e ".[torch,rl]" \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

### Developer (lint + format + typecheck + pre-commit)

```bash
pip install -e ".[dev,torch,rl]" --extra-index-url https://download.pytorch.org/whl/cpu
pre-commit install
```

A `Makefile` wraps the common workflows:

```bash
make help            # list targets
make install-dev     # editable + dev + test + torch (CPU)
make lint            # ruff check
make format          # ruff fix + format
make typecheck       # mypy
make test            # pytest
make ci              # lint + format-check + test (mirrors GitHub CI)
```

---

## Configuration

The pipeline is driven by three YAML files in [`configs/`](configs/):

| File | Purpose |
|---|---|
| [`pipeline.example.yaml`](configs/pipeline.example.yaml) | Default cross-platform config (relative paths under `./work`) |
| [`pipeline.windows.yaml`](configs/pipeline.windows.yaml) | Original `E:/CORONA/...` Windows layout, kept for backwards compatibility |
| [`model.yaml`](configs/model.yaml) | DeepLabV3+ architecture & training/inference knobs |
| [`rl.yaml`](configs/rl.yaml) | RL / grid-search stages for fusion parameter selection |

All keys are documented inline; copy `pipeline.example.yaml` to e.g.
`configs/pipeline.local.yaml` and edit paths to fit your environment.

---

## Running the pipeline

The CLI exposes one entry point per stage as well as a `run-all`:

```bash
# Step-by-step
forestseg prepare-input        --config configs/pipeline.example.yaml
forestseg build-spec-tex       --config configs/pipeline.example.yaml
forestseg check-label-points   --config configs/pipeline.example.yaml
forestseg prepare-label-points --config configs/pipeline.example.yaml
forestseg build-feature-stack  --config configs/pipeline.example.yaml
forestseg train-or-load-dl     --config configs/pipeline.example.yaml
forestseg run-rl-fusion        --config configs/pipeline.example.yaml
forestseg postprocess-export   --config configs/pipeline.example.yaml

# Closed-loop variant (RL feedback over multiple rounds)
forestseg run-closed-loop      --config configs/pipeline.example.yaml

# One-shot end-to-end
forestseg run-all              --config configs/pipeline.example.yaml
```

Equivalent shell wrappers live in [`scripts/`](scripts/) for legacy use:

```bash
PIPELINE_CONFIG=configs/pipeline.example.yaml ./scripts/06_run_all_coupled.sh
```

You can also invoke the package directly:

```bash
python -m forestseg run-all --config configs/pipeline.example.yaml
```

---

## Project layout

```
dp-1/
├── pyproject.toml             ← PEP 621 metadata + ruff/mypy/pytest config
├── Makefile                   ← common workflows
├── .pre-commit-config.yaml    ← ruff + hygiene hooks
├── .github/workflows/ci.yml   ← lint + multi-py test matrix
├── README.md / README.zh-CN.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE                    ← MIT
├── configs/                   ← pipeline / model / RL YAML
├── docs/                      ← architecture, pipeline, labeling SOP
├── scripts/                   ← bash wrappers + label fixup utility
├── src/forestseg/             ← installable package
│   ├── __init__.py            ← public API surface
│   ├── __main__.py            ← python -m forestseg
│   ├── _version.py            ← single source of truth for version
│   ├── _logging.py            ← structured logging helpers
│   ├── cli.py                 ← argparse CLI + command handlers
│   ├── io_raster.py           ← raster I/O + windowed iteration
│   ├── labels.py              ← vector label QA, split, sampling
│   ├── features.py            ← multi-source feature stack
│   ├── spectral.py            ← spectral prior
│   ├── texture.py             ← texture prior
│   ├── model.py               ← DeepLabV3+ wrapper + DiceLoss
│   ├── train.py               ← supervised training loop
│   ├── infer.py               ← sliding-window inference (MC dropout)
│   ├── sampler.py             ← point-centred tile sampler
│   ├── fusion.py              ← DL+spec+tex fusion, grid search
│   ├── postprocess.py         ← morphology + small-component removal
│   ├── rl_env.py              ← state representation
│   ├── rl_policy.py           ← RL/grid policy selection
│   ├── bandit.py              ← contextual bandit alternative
│   ├── feedback_loop.py       ← closed-loop feature feedback
│   ├── scene_runtime.py       ← scene.sh resolution
│   ├── metrics.py             ← binary metrics
│   ├── export.py              ← raster / vector export
│   └── coupling_tensor.py     ← coupling-loss warmup helper
└── tests/                     ← pytest suite (parity-tested vs upstream)
```

---

## Testing

```bash
make test
# or
pytest -ra
```

Baseline parity with upstream `dp`: **338 passed, 3 known-flaky** (input
fixture issues in `test_cli_preflight.py::test_cmd_run_rl_fusion_*`; these
fail in the upstream repository too).

CI runs lint + format-check + the full pytest suite on Python 3.10 / 3.11 /
3.12 via GitHub Actions.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). TL;DR:

1. Fork & branch off `main`.
2. `make install-dev && pre-commit install`.
3. `make ci` must be green before opening a PR.
4. Stick to ruff's defaults (line length 120) and add tests for behaviour
   changes.

---

## License

[MIT](LICENSE) © 2025 jyhuang201900.

This project is a clean-room refactor of
[`jyhuang201900/dp`](https://github.com/jyhuang201900/dp). All algorithms,
configurations, and tests retain their original semantics.
