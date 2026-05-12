import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from forestseg.features import build_feature_stack

BASE_PROFILE = {
    "driver": "GTiff",
    "height": 16,
    "width": 16,
    "count": 1,
    "dtype": "float32",
    "crs": CRS.from_epsg(3857),
    "transform": from_origin(0.0, 16.0, 1.0, 1.0),
    "nodata": -9999.0,
}


def _write_raster(path, profile=None, value=1.0):
    raster_profile = dict(BASE_PROFILE)
    if profile:
        raster_profile.update(profile)
    arr = np.full((raster_profile["height"], raster_profile["width"]), value, dtype=np.float32)
    with rasterio.open(path, "w", **raster_profile) as dst:
        dst.write(arr, 1)
    return path


def test_build_feature_stack_rejects_empty_feature_paths(tmp_path):
    output_path = tmp_path / "stack.tif"

    with pytest.raises(ValueError, match="feature_paths"):
        build_feature_stack(
            feature_paths=[],
            output_path=str(output_path),
            feature_names=[],
        )


@pytest.mark.parametrize(
    ("profile_override", "message"),
    [
        ({"width": 3}, "特征栅格.*尺寸不一致"),
        ({"height": 3}, "特征栅格.*尺寸不一致"),
        ({"crs": CRS.from_epsg(4326)}, "特征栅格 CRS 不一致"),
        ({"transform": from_origin(10.0, 2.0, 1.0, 1.0)}, "特征栅格仿射变换不一致"),
    ],
)
def test_build_feature_stack_rejects_misaligned_inputs(tmp_path, profile_override, message):
    feature_a = _write_raster(tmp_path / "feature_a.tif", value=0.2)
    feature_b = _write_raster(tmp_path / "feature_b.tif", profile=profile_override, value=0.8)
    output_path = tmp_path / "stack.tif"

    with pytest.raises(ValueError, match=message):
        build_feature_stack(
            feature_paths=[str(feature_a), str(feature_b)],
            output_path=str(output_path),
            feature_names=["spec", "tex"],
        )


@pytest.mark.parametrize(
    ("profile_override", "message"),
    [
        ({"width": 32}, "特征栅格.*尺寸不一致"),
        ({"height": 32}, "特征栅格.*尺寸不一致"),
        ({"crs": CRS.from_epsg(4326)}, "特征栅格 CRS 不一致"),
        ({"transform": from_origin(10.0, 16.0, 1.0, 1.0)}, "特征栅格仿射变换不一致"),
    ],
)
def test_build_feature_stack_rejects_misaligned_dem(tmp_path, profile_override, message):
    feature_a = _write_raster(tmp_path / "feature_a.tif", value=0.2)
    dem_path = _write_raster(tmp_path / "dem_input.tif", profile=profile_override, value=42.0)
    output_path = tmp_path / "stack.tif"
    dem_output_path = tmp_path / "dem_normalized.tif"

    with pytest.raises(ValueError, match=message):
        build_feature_stack(
            feature_paths=[str(feature_a)],
            output_path=str(output_path),
            feature_names=["spec"],
            dem_path=str(dem_path),
            dem_output_path=str(dem_output_path),
        )

    meta_path = output_path.with_name(output_path.stem + "_meta.json")
    assert not dem_output_path.exists()
    assert not output_path.exists()
    assert not meta_path.exists()


def test_build_feature_stack_rejects_feature_name_count_mismatch(tmp_path):
    feature_a = _write_raster(tmp_path / "feature_a.tif", value=0.2)
    feature_b = _write_raster(tmp_path / "feature_b.tif", value=0.8)
    output_path = tmp_path / "stack.tif"

    with pytest.raises(ValueError, match="feature_names"):
        build_feature_stack(
            feature_paths=[str(feature_a), str(feature_b)],
            output_path=str(output_path),
            feature_names=["spec"],
        )


def test_build_feature_stack_writes_expected_bands(tmp_path):
    feature_a = _write_raster(tmp_path / "feature_a.tif", value=0.2)
    feature_b = _write_raster(tmp_path / "feature_b.tif", value=0.8)
    output_path = tmp_path / "stack.tif"

    meta = build_feature_stack(
        feature_paths=[str(feature_a), str(feature_b)],
        output_path=str(output_path),
        feature_names=["spec", "tex"],
    )

    with rasterio.open(output_path) as ds:
        assert ds.count == 2
        assert ds.descriptions == ("spec", "tex")
        assert ds.width == BASE_PROFILE["width"]
        assert ds.height == BASE_PROFILE["height"]
        np.testing.assert_allclose(ds.read(1), 0.2)
        np.testing.assert_allclose(ds.read(2), 0.8)

    assert meta["feature_names"] == ["spec", "tex"]
    assert meta["output_path"] == str(output_path)
