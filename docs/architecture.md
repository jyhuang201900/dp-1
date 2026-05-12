# Architecture

`forestseg` is a deliberately flat package: every concern (I/O, priors, model,
fusion, RL, …) is exposed as a top-level sibling under `src/forestseg/`,
making it cheap to import and easy to reason about. The dependency graph is a
DAG; there are **no cyclic imports** and no module imports from `cli.py`.

```
                         ┌────────────┐
                         │  cli.py    │  ← argparse entry point + cmd_* handlers
                         └─────┬──────┘
                               │ orchestrates
        ┌────────────┬─────────┼──────────┬────────────┬──────────────┐
        ▼            ▼         ▼          ▼            ▼              ▼
   scene_runtime  labels    io_raster  features     model         fusion
        │            │         │          │            │              │
        │            ▼         ▼          ▼            ▼              ▼
        │        (validation,│      (spectral,    (train.py,      (postprocess,
        │         sampling,  │       texture)     infer.py,       coupling_tensor,
        │         splitting) │                    sampler.py)     feedback_loop)
        ▼            ▼       ▼                        │              │
   metrics.py    export.py                            ▼              ▼
                                                  rl_env ──→ rl_policy / bandit
```

## Module map

| Module                       | Lines | Responsibility |
|------------------------------|------:|---|
| `cli.py`                     | 1.7k  | argparse parser + `cmd_*` handlers (`prepare-input`, `build-spec-tex`, `check-label-points`, `prepare-label-points`, `build-feature-stack`, `train-or-load-dl`, `run-rl-fusion`, `postprocess-export`, `run-closed-loop`, `run-all`, `preflight-check`) |
| `scene_runtime.py`           | 130   | Resolve `scene.sh` → typed `SceneRuntime`; cross-platform path utilities. |
| `io_raster.py`               | 280   | `read_band1` / `write_band1`, sliding-window iteration, gaussian weighting, nodata constants. |
| `labels.py`                  | 710   | Read & validate `LabelPoint`s from one or two GPKG files, spatially split by grid, JSON serialize, sample rasters at points. |
| `features.py`                | 120   | Compose multi-source feature stack (raw raster + spectral + texture + optional DEM) into a single aligned BIGTIFF. |
| `spectral.py`                | 50    | Otsu + quantile + local-statistic spectral prior. |
| `texture.py`                 | 70    | Multi-scale LBP texture prior. |
| `model.py`                   | 40    | `build_model` (DeepLabV3+ via `segmentation-models-pytorch`), `DiceLoss`, MC-dropout helper. |
| `train.py`                   | 150   | Supervised training loop with metric-driven checkpointing. |
| `infer.py`                   | 100   | Sliding-window inference with MC-dropout-based uncertainty. |
| `sampler.py`                 | 90    | Balanced point sampler + tile reader for per-point training batches. |
| `fusion.py`                  | 140   | `FusionParams` + multi-source fusion + grid-search; stage 0 fixed / stage 1 grid. |
| `postprocess.py`             | 80    | Morphology + small-component pruning + shadow penalty. |
| `rl_env.py`                  | 60    | State (entropy map + prior summaries) for RL / bandit policies. |
| `rl_policy.py`               | 200   | Grid / RL-style fusion-parameter selection driven by validation metrics. |
| `bandit.py`                  | 220   | Contextual multi-armed bandit alternative. |
| `feedback_loop.py`           | 50    | Feature-stack updates between RL rounds. |
| `coupling_tensor.py`         | 20    | Optional auxiliary coupling loss warmup helper. |
| `metrics.py`                 | 35    | Binary precision / recall / F1 / IoU. |
| `export.py`                  | 95    | Final raster + vector export, RL parameter dump. |
| `_logging.py`                | 80    | Structured logging helpers (`configure_logging`, `get_logger`). |
| `_version.py`                | —     | Single-source-of-truth `__version__`. |
| `__init__.py`                | —     | Re-exports `__version__`, docstring overview. |
| `__main__.py`                | —     | `python -m forestseg` entry point. |

## Design principles

1. **Flat package, not deep packages.** A KH-4 segmentation pipeline is
   small enough that one module per concern is more discoverable than a
   tree of sub-packages. Each module is import-safe (no side effects).
2. **No hidden state.** Every stage reads its config dict, writes its
   outputs as files in `work_dir`, and returns. Stages can be re-run
   independently from the CLI.
3. **Test-first compatibility.** The test suite monkey-patches CLI handlers
   via the `forestseg.cli.<name>` namespace; the refactor preserves that
   exact attribute layout so all 338 upstream tests pass unchanged.
4. **Typed dataclasses for cross-stage handoffs**: `LabelPoint`,
   `LabelReadResult`, `FusionParams`, `RasterData`, `CoreHaloWindow`,
   `SceneRuntime` make the implicit "what flows between stages" explicit.
5. **Cross-platform paths.** All path manipulations go through `pathlib`
   and `scene_runtime`'s WSL/Windows conversion helpers; no `os.system` or
   shell pipes inside Python code.

## Backwards compatibility

`forestseg.cli`, `forestseg.labels`, `forestseg.fusion`, `forestseg.train`
etc. all preserve the same public attribute surface as upstream `dp`. The
test suite passes byte-for-byte.

The newly added modules (`_version`, `_logging`, `__main__`) are additive
and never shadow existing names.
