"""Label point reading, validation, splitting, and config resolution.

Re-exports from :mod:`forestseg.labels.points` so that
``from forestseg.labels import LabelPoint`` keeps working.
"""

from __future__ import annotations

# Third-party re-exports expected by upstream tests.
import fiona
from rasterio.warp import transform_geom

from .points import (
    LabelPoint,
    LabelReadOptions,
    LabelReadResult,
    export_points_preview,
    load_points_json,
    read_label_points,
    read_label_points_from_two_files,
    sample_raster_at_points,
    save_points_json,
    split_points_by_grid,
    validate_label_points,
    validate_label_points_from_two_files,
)

__all__ = [
    "LabelPoint",
    "LabelReadOptions",
    "LabelReadResult",
    "export_points_preview",
    "fiona",
    "load_points_json",
    "read_label_points",
    "read_label_points_from_two_files",
    "sample_raster_at_points",
    "save_points_json",
    "split_points_by_grid",
    "transform_geom",
    "validate_label_points",
    "validate_label_points_from_two_files",
]
