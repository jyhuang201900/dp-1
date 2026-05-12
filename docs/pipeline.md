# Pipeline reference

This document walks through every CLI stage of `forestseg`, the YAML keys it
consumes, the files it reads, and the artifacts it writes. All paths are
relative to `work_dir` / `output_dir` from `pipeline.yaml` unless noted.

## High-level diagram

```text
input image + scene.sh + labels + pipeline.yaml
        │
        ▼
   preflight-check ─────────────────────────────► early validation report
        │
        ▼
   prepare-input          ► <work>/inputs/<scene>.tif
        │
        ▼
   build-spec-tex         ► <work>/spec/<scene>_spec.tif
                          ► <work>/tex/<scene>_tex.tif
        │
        ▼
   check-label-points     ► <work>/labels/label_report.json
   prepare-label-points   ► <work>/labels/train.json , val.json
        │
        ▼
   build-feature-stack    ► <work>/features/<scene>_features.tif
        │
        ▼
   train-or-load-dl       ► <work>/checkpoints/best.pt
                          ► <work>/dl/<scene>_prob.tif
                          ► <work>/dl/<scene>_unc.tif
        │
        ▼
   run-rl-fusion          ► <work>/fusion/round_<n>/...
                          ► <work>/fusion/selected_params.json
                          ► <work>/fusion/rl_history.json
        │
        ▼
   postprocess-export     ► <output>/<scene>_mask.tif
                          ► <output>/<scene>_polygons.gpkg
```

## Stages

### `preflight-check`
- Validates the YAML config, label files, and CRS metadata before any heavy
  computation begins.
- Writes a structured `preflight_report.json` summarizing all warnings and
  blocking errors.
- Exits non-zero on a blocking error.

### `prepare-input`
- Reads `scene.sh` via [`scene_runtime`](../src/forestseg/scene_runtime.py).
- Locates the orthorectified TIFF (preferring `*_utm_v1.tif`, falling back to
  `*_utm_quick.tif` if `input.fallback_quick` is true).
- Applies the percentile clip from `preprocess.clip_percentiles` (default
  `[2, 98]`) and writes a normalized float32 TIFF to `<work>/inputs/`.

### `build-spec-tex`
- Reads the normalized input TIFF.
- Computes the **spectral prior** (`spectral.py`): convex combination of
  Otsu, quantile, and local-statistic responses weighted by
  `spectral.{otsu_weight, quantile_weight, local_weight}`.
- Computes the **texture prior** (`texture.py`): multi-scale LBP at the
  windows / radius / point counts from `texture.*`.
- Writes both as separate TIFFs aligned to the input.

### `check-label-points`
- Reads the positive and negative label files declared in
  `labels.{positive_path, negative_path}` using `labels.read_label_points_*`.
- Performs full QA: geometry type, CRS, attribute presence, class balance,
  and grid-split feasibility.
- Emits `label_report.json` plus a colour-coded console summary.

### `prepare-label-points`
- Repeats validation and, on success, materializes the train / val
  point JSON files used by the training loop.
- Spatial split uses `labels.grid_size` cells and `labels.split_ratio`.

### `build-feature-stack`
- Aligns the normalized image, spectral prior, texture prior, optional DEM
  (`dem.path`), and any extra rasters into a single multi-band BIGTIFF
  consumed by both training and inference.
- Bands are tagged via the GeoTIFF description metadata so the order is
  reproducible.

### `train-or-load-dl`
- If `dl.checkpoint_path` exists, loads it; otherwise trains a fresh model.
- Architecture: DeepLabV3+ with the `dl.encoder_name` backbone (default
  `resnet34`) and MC dropout enabled at inference.
- Selection metric is `dl.selection_metric` (one of `f1 / precision /
  recall / iou`).
- After training, runs sliding-window inference (`dl.tile_size`,
  `dl.stride`) to produce a probability map plus an MC-dropout-derived
  uncertainty map.

### `run-rl-fusion`
- Performs grid search (or, in stage 2+, an RL bandit) over the parameter
  candidates in `fusion.grid` to combine DL probability + spectral + texture
  into a final probability.
- Each round persists its parameters and metrics into
  `<work>/fusion/round_<n>/` and appends a row to
  `<work>/fusion/rl_history.json`.
- The best parameter set is written to `<work>/fusion/selected_params.json`.

### `run-closed-loop`
- Runs `train-or-load-dl` + `run-rl-fusion` for `rl_loop.rounds` iterations
  with early stopping (`rl_loop.patience`, `rl_loop.min_delta`).
- Between rounds, `feedback_loop.py` rewrites the feature stack so that
  later rounds incorporate the previous round's fused probability map.

### `postprocess-export`
- Applies opening/closing morphology with kernels `postprocess.opening_kernel`
  / `postprocess.closing_kernel`.
- Removes connected components smaller than the final
  `fusion.fixed_params.min_area_m2` (in m²; converted to pixels via the
  raster transform).
- Writes:
  - Final binary mask raster (uint8) to `<output>/<scene>_mask.tif`.
  - Vector polygons (forest class only) to `<output>/<scene>_polygons.gpkg`.
  - A `selected_params.json` snapshot next to the export.

### `run-all`
- Convenience wrapper that runs every stage above in sequence using the
  same config.
- Use this for first-time end-to-end smoke tests; for production prefer
  `run-closed-loop` because it includes the RL feedback rounds.

## Tips

- Each stage is idempotent — re-running it overwrites its outputs but never
  touches sibling stages.
- The CLI accepts `--dry-run` on every stage to print the config it would
  consume without doing any I/O.
- For reproducibility, pin `dl.seed`, `labels.split_seed`, and the
  fusion `selection_metric`.
- The structured logger can be tuned by setting
  `FORESTSEG_LOG_LEVEL=DEBUG` before launching the CLI.
