import pytest

from forestseg import labels as labels_module


class DummyCollection:
    _DEFAULT_CRS = object()

    def __init__(self, features, crs=_DEFAULT_CRS):
        self._features = features
        self.crs = {"init": "epsg:4326"} if crs is self._DEFAULT_CRS else crs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._features)


def test_read_label_points_from_two_files_forces_source_class_labels(monkeypatch, tmp_path):
    positive_path = tmp_path / "labels_positive.gpkg"
    negative_path = tmp_path / "labels_negative.gpkg"
    positive_path.write_text("placeholder", encoding="utf-8")
    negative_path.write_text("placeholder", encoding="utf-8")

    positive_features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": 0},
        }
    ]
    negative_features = [
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": 1},
        }
    ]

    def fake_open(path, layer=None):
        if path == str(positive_path):
            return DummyCollection(positive_features, crs=None)
        if path == str(negative_path):
            return DummyCollection(negative_features, crs=None)
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(labels_module.fiona, "open", fake_open)

    points = labels_module.read_label_points_from_two_files(
        positive_path=str(positive_path),
        negative_path=str(negative_path),
        class_field="class",
        positive_value="1",
        negative_value="0",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        positive_layer="labels",
        negative_layer="labels",
    )

    assert [pt.label for pt in points] == [1, 0]


def test_validate_label_points_from_two_files_reports_balanced_counts(monkeypatch, tmp_path):
    positive_path = tmp_path / "labels_positive.gpkg"
    negative_path = tmp_path / "labels_negative.gpkg"
    positive_path.write_text("placeholder", encoding="utf-8")
    negative_path.write_text("placeholder", encoding="utf-8")

    positive_features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": 1},
        }
    ]
    negative_features = [
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": 0},
        }
    ]

    def fake_open(path, layer=None):
        if path == str(positive_path):
            return DummyCollection(positive_features, crs=None)
        if path == str(negative_path):
            return DummyCollection(negative_features, crs=None)
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(labels_module.fiona, "open", fake_open)

    report = labels_module.validate_label_points_from_two_files(
        positive_path=str(positive_path),
        negative_path=str(negative_path),
        class_field="class",
        positive_value="1",
        negative_value="0",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        positive_layer="labels",
        negative_layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    assert report["valid_point_count"] == 2
    assert report["class_balance"] == {"positive": 1, "negative": 1}


def test_validate_label_points_reports_invalid_and_missing_label_blockers(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (100.0, 0.0)},
            "properties": {"class": "non_forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2100.0, 0.0)},
            "properties": {"class": "non_forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2200.0, 0.0)},
            "properties": {"class": "bad_label"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2300.0, 0.0)},
            "properties": {"class": None},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert report["valid_point_count"] == 4
    assert report["total_features"] == 6
    assert report["raw_label_counts"] == {"forest": 2, "non_forest": 2, "bad_label": 1}
    assert report["valid_label_counts"] == {"forest": 2, "non_forest": 2}
    assert report["class_balance"] == {"positive": 2, "negative": 2}
    assert report["missing_class_count"] == 1
    assert report["unexpected_label_counts"] == {"bad_label": 1}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [2, 2]
    assert split_positive_counts == [1, 1]
    assert split_negative_counts == [1, 1]
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert set(report["risk"]["warnings"]) == {"有效标签点过少，结果方差可能较大。"}
    assert set(report["risk"]["blocking"]) == {
        "存在未识别标签值，请先清洗标签字段后再运行。",
        "存在缺失 class 字段的要素，请补齐后再运行。",
    }


def test_read_label_points_accepts_numeric_labels_with_1_as_forest(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": 1},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": 2},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    points = labels_module.read_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="1",
        negative_value="0",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
    )

    assert [pt.label for pt in points] == [1, 0]
    assert [pt.properties["class"] for pt in points] == [1, 2]


def test_read_label_points_rejects_non_finite_numeric_labels(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": 1},
        },
        {
            "geometry": {"type": "Point", "coordinates": (1.0, 1.0)},
            "properties": {"class": "nan"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    with pytest.raises(ValueError, match="读取标签失败"):
        labels_module.read_label_points(
            path=str(label_path),
            class_field="class",
            positive_value="1",
            negative_value="0",
            target_crs=None,
            grid_size=1000.0,
            origin_x=0.0,
            origin_y=0.0,
            layer="labels",
        )


def test_validate_label_points_accepts_numeric_labels_with_1_as_forest(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": 1},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": 0},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2100.0, 0.0)},
            "properties": {"class": 2},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="1",
        negative_value="0",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    assert report["valid_point_count"] == 3
    assert report["raw_label_counts"] == {"1": 1, "0": 1, "2": 1}
    assert report["valid_label_counts"] == {"1": 1, "0": 1, "2": 1}
    assert report["class_balance"] == {"positive": 1, "negative": 2}
    assert report["unexpected_label_counts"] == {}

    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"label_type": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"label_type": "non_forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2100.0, 0.0)},
            "properties": {"label_type": None},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="label_type",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert report["missing_class_count"] == 1
    assert report["valid_point_count"] == 2
    assert report["total_features"] == 3
    assert report["raw_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["class_balance"] == {"positive": 1, "negative": 1}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [1, 1]
    assert split_positive_counts == [0, 1]
    assert split_negative_counts == [0, 1]
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert set(report["risk"]["warnings"]) == {"有效标签点过少，结果方差可能较大。"}
    assert set(report["risk"]["blocking"]) == {
        "存在缺失 label_type 字段的要素，请补齐后再运行。",
        "训练集缺少正类或负类，请调整网格或切分比例。",
        "验证集缺少正类或负类，请调整网格或切分比例。",
    }


def test_read_label_points_rejects_missing_source_crs_when_target_crs_requested(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    with pytest.raises(ValueError, match="标签文件缺少 CRS"):
        labels_module.read_label_points(
            path=str(label_path),
            class_field="class",
            positive_value="forest",
            negative_value="non_forest",
            target_crs=labels_module.CRS.from_epsg(3857),
            grid_size=1000.0,
            origin_x=0.0,
            origin_y=0.0,
            layer="labels",
        )


def test_validate_label_points_blocks_missing_source_crs_when_target_crs_requested(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": "non_forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=labels_module.CRS.from_epsg(3857),
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    assert report["crs_missing_for_target"] is True
    assert report["source_crs"] is None
    assert report["target_crs"] == "EPSG:3857"
    assert report["total_features"] == 2
    assert report["raw_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_label_counts"] == {}
    assert report["valid_point_count"] == 0
    assert report["class_balance"] == {"positive": 0, "negative": 0}
    assert report["split_preview"]["split_possible"] is False
    assert report["split_preview"]["grid_count"] == 0
    assert report["split_preview"]["train_grid_count"] == 0
    assert report["split_preview"]["val_grid_count"] == 0
    assert report["split_preview"]["train_points"] == 0
    assert report["split_preview"]["val_points"] == 0
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert set(report["risk"]["warnings"]) == set()
    assert set(report["risk"]["blocking"]) == {
        "标签图层缺少源 CRS，无法转换到目标坐标系 EPSG:3857。请先为标签图层补充正确坐标系后再运行。",
    }


def test_read_label_points_transforms_coordinates_when_target_crs_differs(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (10.0, 20.0)},
            "properties": {"class": "forest"},
        },
    ]
    transformed_geom = {"type": "Point", "coordinates": (2500.0, 3500.0)}
    transform_calls = []

    def fake_transform(src, dst, geom):
        transform_calls.append((src.to_string(), dst.to_string(), geom))
        return transformed_geom

    monkeypatch.setattr(
        labels_module.fiona,
        "open",
        lambda path, layer=None: DummyCollection(features, crs={"init": "epsg:4326"}),
    )
    monkeypatch.setattr(labels_module, "transform_geom", fake_transform)

    points = labels_module.read_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=labels_module.CRS.from_epsg(3857),
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
    )

    normalized_points = [
        {
            "x": pt.x,
            "y": pt.y,
            "label": pt.label,
            "grid_id": pt.grid_id,
        }
        for pt in points
    ]

    assert transform_calls == [
        (
            "EPSG:4326",
            "EPSG:3857",
            {"type": "Point", "coordinates": (10.0, 20.0)},
        )
    ]
    assert normalized_points == [
        {
            "x": 2500.0,
            "y": 3500.0,
            "label": 1,
            "grid_id": "2_3",
        }
    ]


def test_read_label_points_skips_transform_when_source_crs_matches_target(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (1200.0, 2300.0)},
            "properties": {"class": "forest"},
        },
    ]

    def fail_transform(src, dst, geom):
        raise AssertionError("transform_geom should not be called when CRS already matches")

    monkeypatch.setattr(
        labels_module.fiona,
        "open",
        lambda path, layer=None: DummyCollection(features, crs={"init": "epsg:3857"}),
    )
    monkeypatch.setattr(labels_module, "transform_geom", fail_transform)

    points = labels_module.read_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=labels_module.CRS.from_epsg(3857),
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
    )

    normalized_points = [
        {
            "x": pt.x,
            "y": pt.y,
            "label": pt.label,
            "grid_id": pt.grid_id,
        }
        for pt in points
    ]

    assert normalized_points == [
        {
            "x": 1200.0,
            "y": 2300.0,
            "label": 1,
            "grid_id": "1_2",
        }
    ]


def test_read_label_points_rejects_invalid_labels(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (1.0, 1.0)},
            "properties": {"class": "bad_label"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    with pytest.raises(ValueError, match="读取标签失败"):
        labels_module.read_label_points(
            path=str(label_path),
            class_field="class",
            positive_value="forest",
            negative_value="non_forest",
            target_crs=None,
            grid_size=1000.0,
            origin_x=0.0,
            origin_y=0.0,
            layer="labels",
        )

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    assert report["total_features"] == 2
    assert report["raw_label_counts"] == {"forest": 1, "bad_label": 1}
    assert report["valid_label_counts"] == {"forest": 1}
    assert report["valid_point_count"] == 1
    assert report["class_balance"] == {"positive": 1, "negative": 0}
    assert report["split_preview"]["split_possible"] is False
    assert report["split_preview"]["grid_count"] == 1
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 0
    assert report["split_preview"]["train_points"] == 1
    assert report["split_preview"]["val_points"] == 0
    assert report["split_preview"]["train_positive"] == 1
    assert report["split_preview"]["train_negative"] == 0
    assert report["split_preview"]["val_positive"] == 0
    assert report["split_preview"]["val_negative"] == 0
    assert report["unexpected_label_counts"] == {"bad_label": 1}
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert set(report["risk"]["warnings"]) == {"有效标签点过少，结果方差可能较大。"}
    assert set(report["risk"]["blocking"]) == {
        "存在未识别标签值，请先清洗标签字段后再运行。",
        "有效标签缺少正类或负类，无法形成二分类监督。",
        "无法按网格形成同时包含训练集和验证集的空间切分，请增加更多网格标签。",
        "训练集缺少正类或负类，请调整网格或切分比例。",
        "验证集缺少正类或负类，请调整网格或切分比例。",
    }


def test_validate_label_points_skips_transform_when_source_crs_matches_target(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (100.0, 100.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2500.0, 3500.0)},
            "properties": {"class": "non_forest"},
        },
    ]

    def fail_transform(src, dst, geom):
        raise AssertionError("transform_geom should not be called when CRS already matches")

    monkeypatch.setattr(
        labels_module.fiona,
        "listlayers",
        lambda path: ["labels"],
    )
    monkeypatch.setattr(
        labels_module.fiona,
        "open",
        lambda path, layer=None: DummyCollection(features, crs={"init": "epsg:3857"}),
    )
    monkeypatch.setattr(labels_module, "transform_geom", fail_transform)

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=labels_module.CRS.from_epsg(3857),
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert report["source_crs"] == "EPSG:3857"
    assert report["target_crs"] == "EPSG:3857"
    assert report["total_features"] == 2
    assert report["raw_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_point_count"] == 2
    assert report["class_balance"] == {"positive": 1, "negative": 1}
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [1, 1]
    assert report["split_preview"]["split_possible"] is True
    assert split_positive_counts == [0, 1]
    assert split_negative_counts == [0, 1]
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert "有效标签点过少，结果方差可能较大。" in report["risk"]["warnings"]
    assert "训练集缺少正类或负类，请调整网格或切分比例。" in report["risk"]["blocking"]
    assert "验证集缺少正类或负类，请调整网格或切分比例。" in report["risk"]["blocking"]


def test_validate_label_points_uses_transformed_coordinates_for_grid_split(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (10.0, 20.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (20.0, 30.0)},
            "properties": {"class": "non_forest"},
        },
    ]
    transformed_by_coord = {
        (10.0, 20.0): {"type": "Point", "coordinates": (100.0, 100.0)},
        (20.0, 30.0): {"type": "Point", "coordinates": (2500.0, 3500.0)},
    }
    transform_calls = []

    def fake_transform(src, dst, geom):
        transform_calls.append((src.to_string(), dst.to_string(), geom))
        return transformed_by_coord[tuple(geom["coordinates"])]

    monkeypatch.setattr(
        labels_module.fiona,
        "listlayers",
        lambda path: ["labels"],
    )
    monkeypatch.setattr(
        labels_module.fiona,
        "open",
        lambda path, layer=None: DummyCollection(features, crs={"init": "epsg:4326"}),
    )
    monkeypatch.setattr(labels_module, "transform_geom", fake_transform)

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=labels_module.CRS.from_epsg(3857),
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert transform_calls == [
        (
            "EPSG:4326",
            "EPSG:3857",
            {"type": "Point", "coordinates": (10.0, 20.0)},
        ),
        (
            "EPSG:4326",
            "EPSG:3857",
            {"type": "Point", "coordinates": (20.0, 30.0)},
        ),
    ]
    assert report["source_crs"] == "EPSG:4326"
    assert report["target_crs"] == "EPSG:3857"
    assert report["total_features"] == 2
    assert report["raw_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_point_count"] == 2
    assert report["class_balance"] == {"positive": 1, "negative": 1}
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [1, 1]
    assert report["split_preview"]["split_possible"] is True
    assert split_positive_counts == [0, 1]
    assert split_negative_counts == [0, 1]
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert "有效标签点过少，结果方差可能较大。" in report["risk"]["warnings"]
    assert "训练集缺少正类或负类，请调整网格或切分比例。" in report["risk"]["blocking"]
    assert "验证集缺少正类或负类，请调整网格或切分比例。" in report["risk"]["blocking"]


def test_read_label_points_returns_valid_points(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (100.0, 0.0)},
            "properties": {"class": "non_forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    points = labels_module.read_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
    )

    normalized_points = sorted(
        [
            {
                "x": pt.x,
                "y": pt.y,
                "label": pt.label,
                "grid_id": pt.grid_id,
                "properties": pt.properties,
            }
            for pt in points
        ],
        key=lambda pt: (pt["x"], pt["y"], pt["label"]),
    )

    assert normalized_points == [
        {
            "x": 0.0,
            "y": 0.0,
            "label": 1,
            "grid_id": "0_0",
            "properties": {"class": "forest"},
        },
        {
            "x": 100.0,
            "y": 0.0,
            "label": 0,
            "grid_id": "0_0",
            "properties": {"class": "non_forest"},
        },
    ]


def test_read_label_points_uses_first_available_layer_by_default(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    listlayers_calls = []
    layers_seen = []

    def fake_listlayers(path):
        listlayers_calls.append(path)
        return ["first_layer", "second_layer"]

    def fake_open(path, layer=None):
        layers_seen.append(layer)
        assert layer == "first_layer"
        return DummyCollection(
            [
                {
                    "geometry": {"type": "Point", "coordinates": (2200.0, 3100.0)},
                    "properties": {"class": "forest"},
                },
            ],
            crs=None,
        )

    monkeypatch.setattr(labels_module.fiona, "listlayers", fake_listlayers)
    monkeypatch.setattr(labels_module.fiona, "open", fake_open)

    points = labels_module.read_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer=None,
    )

    normalized_points = [
        {
            "x": pt.x,
            "y": pt.y,
            "label": pt.label,
            "grid_id": pt.grid_id,
        }
        for pt in points
    ]

    assert listlayers_calls == [str(label_path)]
    assert layers_seen == ["first_layer"]
    assert normalized_points == [
        {
            "x": 2200.0,
            "y": 3100.0,
            "label": 1,
            "grid_id": "2_3",
        }
    ]


def test_read_label_points_uses_explicit_layer_without_listing_layers(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")

    def fail_listlayers(path):
        raise AssertionError("fiona.listlayers should not be called when layer is explicit")

    def fake_open(path, layer=None):
        assert layer == "manual_layer"
        return DummyCollection(
            [
                {
                    "geometry": {"type": "Point", "coordinates": (2200.0, 3100.0)},
                    "properties": {"class": "forest"},
                },
            ],
            crs=None,
        )

    monkeypatch.setattr(labels_module.fiona, "listlayers", fail_listlayers)
    monkeypatch.setattr(labels_module.fiona, "open", fake_open)

    points = labels_module.read_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="manual_layer",
    )

    normalized_points = [
        {
            "x": pt.x,
            "y": pt.y,
            "label": pt.label,
            "grid_id": pt.grid_id,
        }
        for pt in points
    ]

    assert normalized_points == [
        {
            "x": 2200.0,
            "y": 3100.0,
            "label": 1,
            "grid_id": "2_3",
        }
    ]


def test_read_label_points_rejects_when_no_layers_are_available(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    listlayers_calls = []

    def fake_listlayers(path):
        listlayers_calls.append(path)
        return []

    def fail_open(path, layer=None):
        raise AssertionError("fiona.open should not be called when no layers are available")

    monkeypatch.setattr(labels_module.fiona, "listlayers", fake_listlayers)
    monkeypatch.setattr(labels_module.fiona, "open", fail_open)

    with pytest.raises(ValueError, match="标签文件没有可用图层"):
        labels_module.read_label_points(
            path=str(label_path),
            class_field="class",
            positive_value="forest",
            negative_value="non_forest",
            target_crs=None,
            grid_size=1000.0,
            origin_x=0.0,
            origin_y=0.0,
            layer=None,
        )

    assert listlayers_calls == [str(label_path)]


def test_read_label_points_normalizes_case_and_whitespace(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": " Forest "},
        },
        {
            "geometry": {"type": "Point", "coordinates": (1500.0, 0.0)},
            "properties": {"class": "NON_FOREST"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    points = labels_module.read_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
    )

    normalized_points = sorted(
        [
            {
                "x": pt.x,
                "y": pt.y,
                "label": pt.label,
            }
            for pt in points
        ],
        key=lambda pt: (pt["x"], pt["y"], pt["label"]),
    )

    assert normalized_points == [
        {
            "x": 0.0,
            "y": 0.0,
            "label": 1,
        },
        {
            "x": 1500.0,
            "y": 0.0,
            "label": 0,
        },
    ]


def test_read_label_points_rejects_colliding_normalized_label_values(tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")

    with pytest.raises(ValueError, match="正负类取值不能相同"):
        labels_module.read_label_points(
            path=str(label_path),
            positive_value="Forest",
            negative_value=" forest ",
        )


def test_validate_label_points_rejects_colliding_normalized_label_values(tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")

    with pytest.raises(ValueError, match="正负类取值不能相同"):
        labels_module.validate_label_points(
            path=str(label_path),
            positive_value="Forest",
            negative_value=" forest ",
        )


def test_validate_label_points_rejects_when_no_layers_are_available(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    listlayers_calls = []

    def fake_listlayers(path):
        listlayers_calls.append(path)
        return []

    def fail_open(path, layer=None):
        raise AssertionError("fiona.open should not be called when no layers are available")

    monkeypatch.setattr(labels_module.fiona, "listlayers", fake_listlayers)
    monkeypatch.setattr(labels_module.fiona, "open", fail_open)

    with pytest.raises(ValueError, match="标签文件没有可用图层"):
        labels_module.validate_label_points(
            path=str(label_path),
            class_field="class",
            positive_value="forest",
            negative_value="non_forest",
            target_crs=None,
            grid_size=1000.0,
            origin_x=0.0,
            origin_y=0.0,
            layer=None,
        )

    assert listlayers_calls == [str(label_path)]


def test_validate_label_points_uses_first_available_layer_by_default(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    listlayers_calls = []
    layers_seen = []

    def fake_listlayers(path):
        listlayers_calls.append(path)
        return ["first_layer", "second_layer"]

    def fake_open(path, layer=None):
        layers_seen.append(layer)
        assert layer == "first_layer"
        return DummyCollection(
            [
                {
                    "geometry": {"type": "Point", "coordinates": (2200.0, 3100.0)},
                    "properties": {"class": "forest"},
                },
                {
                    "geometry": {"type": "Point", "coordinates": (4200.0, 3100.0)},
                    "properties": {"class": "non_forest"},
                },
            ],
            crs=None,
        )

    monkeypatch.setattr(labels_module.fiona, "listlayers", fake_listlayers)
    monkeypatch.setattr(labels_module.fiona, "open", fake_open)

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer=None,
        train_ratio=0.7,
        seed=42,
    )

    assert listlayers_calls == [str(label_path)]
    assert layers_seen == ["first_layer"]
    assert report["selected_layer"] == "first_layer"
    assert report["available_layers"] == ["first_layer", "second_layer"]
    assert report["total_features"] == 2
    assert report["raw_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_point_count"] == 2
    assert report["class_balance"] == {"positive": 1, "negative": 1}


def test_validate_label_points_uses_explicit_layer_without_listing_layers(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")

    def fail_listlayers(path):
        raise AssertionError("fiona.listlayers should not be called when layer is explicit")

    def fake_open(path, layer=None):
        assert layer == "manual_layer"
        return DummyCollection(
            [
                {
                    "geometry": {"type": "Point", "coordinates": (2200.0, 3100.0)},
                    "properties": {"class": "forest"},
                },
                {
                    "geometry": {"type": "Point", "coordinates": (4200.0, 3100.0)},
                    "properties": {"class": "non_forest"},
                },
            ],
            crs=None,
        )

    monkeypatch.setattr(labels_module.fiona, "listlayers", fail_listlayers)
    monkeypatch.setattr(labels_module.fiona, "open", fake_open)

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="manual_layer",
        train_ratio=0.7,
        seed=42,
    )

    assert report["selected_layer"] == "manual_layer"
    assert report["available_layers"] == ["manual_layer"]
    assert report["total_features"] == 2
    assert report["valid_point_count"] == 2
    assert report["class_balance"] == {"positive": 1, "negative": 1}


def test_read_label_points_rejects_non_point_geometries(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]],
            },
            "properties": {"class": "non_forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    with pytest.raises(ValueError, match="读取标签失败"):
        labels_module.read_label_points(
            path=str(label_path),
            class_field="class",
            positive_value="forest",
            negative_value="non_forest",
            target_crs=None,
            grid_size=1000.0,
            origin_x=0.0,
            origin_y=0.0,
            layer="labels",
        )


def test_read_label_points_rejects_null_geometries(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": None,
            "properties": {"class": "non_forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    with pytest.raises(ValueError, match="读取标签失败"):
        labels_module.read_label_points(
            path=str(label_path),
            class_field="class",
            positive_value="forest",
            negative_value="non_forest",
            target_crs=None,
            grid_size=1000.0,
            origin_x=0.0,
            origin_y=0.0,
            layer="labels",
        )


def test_read_label_points_rejects_when_no_features_are_present(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = []

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    with pytest.raises(ValueError, match="没有有效点"):
        labels_module.read_label_points(
            path=str(label_path),
            class_field="class",
            positive_value="forest",
            negative_value="non_forest",
            target_crs=None,
            grid_size=1000.0,
            origin_x=0.0,
            origin_y=0.0,
            layer="labels",
        )


def test_validate_label_points_reports_empty_layer_as_blocking(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = []

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    assert report["total_features"] == 0
    assert report["geometry_counts"] == {}
    assert report["raw_label_counts"] == {}
    assert report["valid_label_counts"] == {}
    assert report["unexpected_label_counts"] == {}
    assert report["missing_class_count"] == 0
    assert report["valid_point_count"] == 0
    assert report["class_balance"] == {"positive": 0, "negative": 0}
    assert report["split_preview"]["split_possible"] is False
    assert report["split_preview"]["grid_count"] == 0
    assert report["split_preview"]["train_grid_count"] == 0
    assert report["split_preview"]["val_grid_count"] == 0
    assert report["split_preview"]["train_points"] == 0
    assert report["split_preview"]["val_points"] == 0
    assert report["split_preview"]["train_positive"] == 0
    assert report["split_preview"]["train_negative"] == 0
    assert report["split_preview"]["val_positive"] == 0
    assert report["split_preview"]["val_negative"] == 0
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert report["risk"]["warnings"] == ["有效标签点过少，结果方差可能较大。"]
    assert report["risk"]["blocking"] == [
        "有效标签缺少正类或负类，无法形成二分类监督。",
        "无法按网格形成同时包含训练集和验证集的空间切分，请增加更多网格标签。",
        "训练集缺少正类或负类，请调整网格或切分比例。",
        "验证集缺少正类或负类，请调整网格或切分比例。",
    ]


def test_validate_label_points_blocks_non_point_geometries(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": "non_forest"},
        },
        {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]],
            },
            "properties": {"class": "forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    assert report["total_features"] == 3
    assert report["geometry_counts"]["Point"] == 2
    assert report["geometry_counts"]["Polygon"] == 1
    assert report["raw_label_counts"] == {"forest": 2, "non_forest": 1}
    assert report["valid_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_point_count"] == 2
    assert report["class_balance"] == {"positive": 1, "negative": 1}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert "有效标签点过少，结果方差可能较大。" in report["risk"]["warnings"]
    assert "标签图层包含非 Point 几何，请移除或转换为点后再运行。" in report["risk"]["blocking"]


def test_validate_label_points_blocks_null_geometries(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": "non_forest"},
        },
        {
            "geometry": None,
            "properties": {"class": "forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    assert report["total_features"] == 3
    assert report["geometry_counts"]["Point"] == 2
    assert report["geometry_counts"]["None"] == 1
    assert report["raw_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_point_count"] == 2
    assert report["class_balance"] == {"positive": 1, "negative": 1}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert "有效标签点过少，结果方差可能较大。" in report["risk"]["warnings"]
    assert "标签图层包含非 Point 几何，请移除或转换为点后再运行。" in report["risk"]["blocking"]


def test_validate_label_points_blocks_when_no_valid_points_remain(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "bad_label"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (1.0, 1.0)},
            "properties": {"class": None},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert report["total_features"] == 2
    assert report["raw_label_counts"] == {"bad_label": 1}
    assert report["valid_label_counts"] == {}
    assert report["valid_point_count"] == 0
    assert report["class_balance"] == {"positive": 0, "negative": 0}
    assert report["unexpected_label_counts"] == {"bad_label": 1}
    assert report["missing_class_count"] == 1
    assert report["split_preview"]["split_possible"] is False
    assert report["split_preview"]["grid_count"] == 0
    assert report["split_preview"]["train_grid_count"] == 0
    assert report["split_preview"]["val_grid_count"] == 0
    assert split_point_counts == [0, 0]
    assert split_positive_counts == [0, 0]
    assert split_negative_counts == [0, 0]
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert set(report["risk"]["warnings"]) == {"有效标签点过少，结果方差可能较大。"}
    assert set(report["risk"]["blocking"]) == {
        "存在未识别标签值，请先清洗标签字段后再运行。",
        "存在缺失 class 字段的要素，请补齐后再运行。",
        "有效标签缺少正类或负类，无法形成二分类监督。",
        "无法按网格形成同时包含训练集和验证集的空间切分，请增加更多网格标签。",
        "训练集缺少正类或负类，请调整网格或切分比例。",
        "验证集缺少正类或负类，请调整网格或切分比例。",
    }


def test_validate_label_points_blocks_when_split_is_impossible(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (10.0, 10.0)},
            "properties": {"class": "non_forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    assert report["total_features"] == 2
    assert report["raw_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_label_counts"] == {"forest": 1, "non_forest": 1}
    assert report["valid_point_count"] == 2
    assert report["class_balance"] == {"positive": 1, "negative": 1}
    assert report["split_preview"]["split_possible"] is False
    assert report["split_preview"]["grid_count"] == 1
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 0
    assert report["split_preview"]["train_points"] == 2
    assert report["split_preview"]["val_points"] == 0
    assert report["split_preview"]["train_positive"] == 1
    assert report["split_preview"]["train_negative"] == 1
    assert report["split_preview"]["val_positive"] == 0
    assert report["split_preview"]["val_negative"] == 0
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert "有效标签点过少，结果方差可能较大。" in report["risk"]["warnings"]
    assert "无法按网格形成同时包含训练集和验证集的空间切分，请增加更多网格标签。" in report["risk"]["blocking"]
    assert "验证集缺少正类或负类，请调整网格或切分比例。" in report["risk"]["blocking"]


def test_validate_label_points_blocks_single_class_datasets(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": "forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert report["total_features"] == 2
    assert report["raw_label_counts"] == {"forest": 2}
    assert report["valid_label_counts"] == {"forest": 2}
    assert report["valid_point_count"] == 2
    assert report["class_balance"] == {"positive": 2, "negative": 0}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [1, 1]
    assert split_positive_counts == [1, 1]
    assert split_negative_counts == [0, 0]
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert "有效标签点过少，结果方差可能较大。" in report["risk"]["warnings"]
    assert "有效标签缺少正类或负类，无法形成二分类监督。" in report["risk"]["blocking"]
    assert "训练集缺少正类或负类，请调整网格或切分比例。" in report["risk"]["blocking"]
    assert "验证集缺少正类或负类，请调整网格或切分比例。" in report["risk"]["blocking"]


def test_validate_label_points_blocks_single_class_split_partitions(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (100.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": "non_forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2100.0, 0.0)},
            "properties": {"class": "non_forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )

    assert report["total_features"] == 4
    assert report["raw_label_counts"] == {"forest": 2, "non_forest": 2}
    assert report["valid_label_counts"] == {"forest": 2, "non_forest": 2}
    assert report["valid_point_count"] == 4
    assert report["class_balance"] == {"positive": 2, "negative": 2}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [2, 2]
    assert sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    ) == [0, 2]
    assert sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    ) == [0, 2]
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert set(report["risk"]["warnings"]) == {"有效标签点过少，结果方差可能较大。"}
    assert set(report["risk"]["blocking"]) == {
        "训练集缺少正类或负类，请调整网格或切分比例。",
        "验证集缺少正类或负类，请调整网格或切分比例。",
    }


def test_validate_label_points_adds_risk_warning_for_sparse_labels(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (100.0, 0.0)},
            "properties": {"class": "non_forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2100.0, 0.0)},
            "properties": {"class": "non_forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert report["total_features"] == 4
    assert report["raw_label_counts"] == {"forest": 2, "non_forest": 2}
    assert report["valid_label_counts"] == {"forest": 2, "non_forest": 2}
    assert report["valid_point_count"] == 4
    assert report["class_balance"] == {"positive": 2, "negative": 2}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [2, 2]
    assert split_positive_counts == [1, 1]
    assert split_negative_counts == [1, 1]
    assert report["risk"]["status"] == "warning"
    assert report["risk"]["can_run"] is True
    assert set(report["risk"]["warnings"]) == {
        "有效标签点过少，结果方差可能较大。",
        "训练集某一类样本过少，监督信号可能不稳定。",
        "验证集某一类样本过少，评估指标可能不稳定。",
    }
    assert report["risk"]["blocking"] == []


def test_validate_label_points_warns_when_training_class_count_is_too_small(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = []
    for idx in range(5):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (0.0, idx * 10.0)},
                "properties": {"class": "forest"},
            }
        )
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (0.0, 100.0 + idx * 10.0)},
                "properties": {"class": "non_forest"},
            }
        )
    features.append(
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": "forest"},
        }
    )
    features.append(
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 100.0)},
            "properties": {"class": "non_forest"},
        }
    )

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    sparse_split_warnings = {
        warning
        for warning in report["risk"]["warnings"]
        if warning
        in {
            "训练集某一类样本过少，监督信号可能不稳定。",
            "验证集某一类样本过少，评估指标可能不稳定。",
        }
    }

    assert report["total_features"] == 12
    assert report["raw_label_counts"] == {"forest": 6, "non_forest": 6}
    assert report["valid_label_counts"] == {"forest": 6, "non_forest": 6}
    assert report["valid_point_count"] == 12
    assert report["class_balance"] == {"positive": 6, "negative": 6}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [2, 10]
    assert split_positive_counts == [1, 5]
    assert split_negative_counts == [1, 5]
    assert report["risk"]["status"] == "warning"
    assert report["risk"]["can_run"] is True
    assert "有效标签点过少，结果方差可能较大。" in report["risk"]["warnings"]
    assert sparse_split_warnings in (
        {"训练集某一类样本过少，监督信号可能不稳定。"},
        {"验证集某一类样本过少，评估指标可能不稳定。"},
    )
    assert "有效标签类别分布严重失衡，少数类占比过低，模型可能偏向多数类。" not in report["risk"]["warnings"]
    assert "训练集类别分布严重失衡，少数类占比过低，训练可能偏向多数类。" not in report["risk"]["warnings"]
    assert "验证集类别分布严重失衡，少数类占比过低，评估可能偏向多数类。" not in report["risk"]["warnings"]
    assert report["risk"]["blocking"] == []


def test_validate_label_points_warns_when_class_balance_is_severely_imbalanced(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = []
    for x_offset in (0.0, 2000.0):
        for idx in range(3):
            features.append(
                {
                    "geometry": {"type": "Point", "coordinates": (x_offset + idx * 10.0, 0.0)},
                    "properties": {"class": "forest"},
                }
            )
        for idx in range(12):
            features.append(
                {
                    "geometry": {"type": "Point", "coordinates": (x_offset + idx * 10.0, 100.0)},
                    "properties": {"class": "non_forest"},
                }
            )

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert report["total_features"] == 30
    assert report["raw_label_counts"] == {"forest": 6, "non_forest": 24}
    assert report["valid_label_counts"] == {"forest": 6, "non_forest": 24}
    assert report["valid_point_count"] == 30
    assert report["class_balance"] == {"positive": 6, "negative": 24}
    assert report["split_preview"]["split_possible"] is True
    assert split_point_counts == [15, 15]
    assert split_positive_counts == [3, 3]
    assert split_negative_counts == [12, 12]
    assert report["risk"]["status"] == "warning"
    assert report["risk"]["can_run"] is True
    assert "有效标签类别分布严重失衡，少数类占比过低，模型可能偏向多数类。" in report["risk"]["warnings"]
    assert "训练集类别分布严重失衡，少数类占比过低，训练可能偏向多数类。" in report["risk"]["warnings"]
    assert "验证集类别分布严重失衡，少数类占比过低，评估可能偏向多数类。" in report["risk"]["warnings"]
    assert "有效标签点过少，结果方差可能较大。" not in report["risk"]["warnings"]
    assert "训练集某一类样本过少，监督信号可能不稳定。" not in report["risk"]["warnings"]
    assert "验证集某一类样本过少，评估指标可能不稳定。" not in report["risk"]["warnings"]
    assert report["risk"]["blocking"] == []


def test_validate_label_points_warns_when_splits_are_severely_imbalanced_even_if_overall_balance_is_ok(
    monkeypatch, tmp_path
):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = []
    for idx in range(3):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (idx * 10.0, 0.0)},
                "properties": {"class": "forest"},
            }
        )
    for idx in range(30):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (idx * 10.0, 100.0)},
                "properties": {"class": "non_forest"},
            }
        )
    for idx in range(30):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (2000.0 + idx * 10.0, 0.0)},
                "properties": {"class": "forest"},
            }
        )
    for idx in range(3):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (2000.0 + idx * 10.0, 100.0)},
                "properties": {"class": "non_forest"},
            }
        )

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_minority_counts = sorted(
        [
            min(report["split_preview"]["train_positive"], report["split_preview"]["train_negative"]),
            min(report["split_preview"]["val_positive"], report["split_preview"]["val_negative"]),
        ]
    )
    split_majority_counts = sorted(
        [
            max(report["split_preview"]["train_positive"], report["split_preview"]["train_negative"]),
            max(report["split_preview"]["val_positive"], report["split_preview"]["val_negative"]),
        ]
    )

    assert report["total_features"] == 66
    assert report["raw_label_counts"] == {"forest": 33, "non_forest": 33}
    assert report["valid_label_counts"] == {"forest": 33, "non_forest": 33}
    assert report["valid_point_count"] == 66
    assert report["class_balance"] == {"positive": 33, "negative": 33}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [33, 33]
    assert split_minority_counts == [3, 3]
    assert split_majority_counts == [30, 30]
    assert report["risk"]["status"] == "warning"
    assert report["risk"]["can_run"] is True
    assert "训练集类别分布严重失衡，少数类占比过低，训练可能偏向多数类。" in report["risk"]["warnings"]
    assert "验证集类别分布严重失衡，少数类占比过低，评估可能偏向多数类。" in report["risk"]["warnings"]
    assert "有效标签点过少，结果方差可能较大。" not in report["risk"]["warnings"]
    assert "有效标签类别分布严重失衡，少数类占比过低，模型可能偏向多数类。" not in report["risk"]["warnings"]
    assert "训练集某一类样本过少，监督信号可能不稳定。" not in report["risk"]["warnings"]
    assert "验证集某一类样本过少，评估指标可能不稳定。" not in report["risk"]["warnings"]
    assert report["risk"]["blocking"] == []


def test_validate_label_points_warns_for_split_skewed_two_grid_balanced_overall_case(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = []
    for idx in range(12):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (idx * 10.0, 0.0)},
                "properties": {"class": "forest"},
            }
        )
    for idx in range(3):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (idx * 10.0, 100.0)},
                "properties": {"class": "non_forest"},
            }
        )
    for idx in range(3):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (2000.0 + idx * 10.0, 0.0)},
                "properties": {"class": "forest"},
            }
        )
    for idx in range(12):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (2000.0 + idx * 10.0, 100.0)},
                "properties": {"class": "non_forest"},
            }
        )

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.5,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_minority_counts = sorted(
        [
            min(report["split_preview"]["train_positive"], report["split_preview"]["train_negative"]),
            min(report["split_preview"]["val_positive"], report["split_preview"]["val_negative"]),
        ]
    )
    split_majority_counts = sorted(
        [
            max(report["split_preview"]["train_positive"], report["split_preview"]["train_negative"]),
            max(report["split_preview"]["val_positive"], report["split_preview"]["val_negative"]),
        ]
    )

    assert report["total_features"] == 30
    assert report["raw_label_counts"] == {"forest": 15, "non_forest": 15}
    assert report["valid_label_counts"] == {"forest": 15, "non_forest": 15}
    assert report["valid_point_count"] == 30
    assert report["class_balance"] == {"positive": 15, "negative": 15}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [15, 15]
    assert split_minority_counts == [3, 3]
    assert split_majority_counts == [12, 12]
    assert report["risk"]["status"] == "warning"
    assert report["risk"]["can_run"] is True
    assert "训练集类别分布严重失衡，少数类占比过低，训练可能偏向多数类。" in report["risk"]["warnings"]
    assert "验证集类别分布严重失衡，少数类占比过低，评估可能偏向多数类。" in report["risk"]["warnings"]
    assert "有效标签点过少，结果方差可能较大。" not in report["risk"]["warnings"]
    assert "有效标签类别分布严重失衡，少数类占比过低，模型可能偏向多数类。" not in report["risk"]["warnings"]
    assert "训练集某一类样本过少，监督信号可能不稳定。" not in report["risk"]["warnings"]
    assert "验证集某一类样本过少，评估指标可能不稳定。" not in report["risk"]["warnings"]
    assert report["risk"]["blocking"] == []


def test_validate_label_points_does_not_add_split_severe_imbalance_warning_below_minority_threshold(
    monkeypatch, tmp_path
):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = []
    for idx in range(8):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (idx * 10.0, 0.0)},
                "properties": {"class": "forest"},
            }
        )
    for idx in range(2):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (idx * 10.0, 100.0)},
                "properties": {"class": "non_forest"},
            }
        )
    for idx in range(2):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (2000.0 + idx * 10.0, 0.0)},
                "properties": {"class": "forest"},
            }
        )
    for idx in range(8):
        features.append(
            {
                "geometry": {"type": "Point", "coordinates": (2000.0 + idx * 10.0, 100.0)},
                "properties": {"class": "non_forest"},
            }
        )

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.5,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_minority_counts = sorted(
        [
            min(report["split_preview"]["train_positive"], report["split_preview"]["train_negative"]),
            min(report["split_preview"]["val_positive"], report["split_preview"]["val_negative"]),
        ]
    )
    split_majority_counts = sorted(
        [
            max(report["split_preview"]["train_positive"], report["split_preview"]["train_negative"]),
            max(report["split_preview"]["val_positive"], report["split_preview"]["val_negative"]),
        ]
    )

    assert report["total_features"] == 20
    assert report["raw_label_counts"] == {"forest": 10, "non_forest": 10}
    assert report["valid_label_counts"] == {"forest": 10, "non_forest": 10}
    assert report["valid_point_count"] == 20
    assert report["class_balance"] == {"positive": 10, "negative": 10}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [10, 10]
    assert split_minority_counts == [2, 2]
    assert split_majority_counts == [8, 8]
    assert report["risk"]["status"] == "warning"
    assert report["risk"]["can_run"] is True
    assert "训练集某一类样本过少，监督信号可能不稳定。" in report["risk"]["warnings"]
    assert "验证集某一类样本过少，评估指标可能不稳定。" in report["risk"]["warnings"]
    assert "有效标签点过少，结果方差可能较大。" not in report["risk"]["warnings"]
    assert "有效标签类别分布严重失衡，少数类占比过低，模型可能偏向多数类。" not in report["risk"]["warnings"]
    assert "训练集类别分布严重失衡，少数类占比过低，训练可能偏向多数类。" not in report["risk"]["warnings"]
    assert "验证集类别分布严重失衡，少数类占比过低，评估可能偏向多数类。" not in report["risk"]["warnings"]
    assert report["risk"]["blocking"] == []


def test_validate_label_points_does_not_warn_when_class_imbalance_stays_below_threshold(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = []
    for x_offset in (0.0, 2000.0):
        for idx in range(3):
            features.append(
                {
                    "geometry": {"type": "Point", "coordinates": (x_offset + idx * 10.0, 0.0)},
                    "properties": {"class": "forest"},
                }
            )
        for idx in range(9):
            features.append(
                {
                    "geometry": {"type": "Point", "coordinates": (x_offset + idx * 10.0, 100.0)},
                    "properties": {"class": "non_forest"},
                }
            )

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert report["total_features"] == 24
    assert report["raw_label_counts"] == {"forest": 6, "non_forest": 18}
    assert report["valid_label_counts"] == {"forest": 6, "non_forest": 18}
    assert report["valid_point_count"] == 24
    assert report["class_balance"] == {"positive": 6, "negative": 18}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [12, 12]
    assert split_positive_counts == [3, 3]
    assert split_negative_counts == [9, 9]
    assert report["risk"]["status"] == "ok"
    assert report["risk"]["can_run"] is True
    assert report["risk"]["warnings"] == []
    assert report["risk"]["blocking"] == []


def test_validate_label_points_is_ok_when_label_distribution_is_balanced(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = []
    for x_offset in (0.0, 2000.0):
        for idx in range(5):
            features.append(
                {
                    "geometry": {"type": "Point", "coordinates": (x_offset + idx * 10.0, 0.0)},
                    "properties": {"class": "forest"},
                }
            )
            features.append(
                {
                    "geometry": {"type": "Point", "coordinates": (x_offset + idx * 10.0, 100.0)},
                    "properties": {"class": "non_forest"},
                }
            )

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert report["total_features"] == 20
    assert report["raw_label_counts"] == {"forest": 10, "non_forest": 10}
    assert report["valid_label_counts"] == {"forest": 10, "non_forest": 10}
    assert report["valid_point_count"] == 20
    assert report["class_balance"] == {"positive": 10, "negative": 10}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [10, 10]
    assert split_positive_counts == [5, 5]
    assert split_negative_counts == [5, 5]
    assert report["risk"]["status"] == "ok"
    assert report["risk"]["can_run"] is True
    assert report["risk"]["warnings"] == []
    assert report["risk"]["blocking"] == []


def test_validate_label_points_keeps_warnings_when_status_is_block(monkeypatch, tmp_path):
    label_path = tmp_path / "labels.gpkg"
    label_path.write_text("placeholder", encoding="utf-8")
    features = [
        {
            "geometry": {"type": "Point", "coordinates": (0.0, 0.0)},
            "properties": {"class": "forest"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (2000.0, 0.0)},
            "properties": {"class": "forest"},
        },
    ]

    monkeypatch.setattr(labels_module.fiona, "listlayers", lambda path: ["labels"])
    monkeypatch.setattr(labels_module.fiona, "open", lambda path, layer=None: DummyCollection(features, crs=None))

    report = labels_module.validate_label_points(
        path=str(label_path),
        class_field="class",
        positive_value="forest",
        negative_value="non_forest",
        target_crs=None,
        grid_size=1000.0,
        origin_x=0.0,
        origin_y=0.0,
        layer="labels",
        train_ratio=0.7,
        seed=42,
    )

    split_point_counts = sorted(
        [
            report["split_preview"]["train_points"],
            report["split_preview"]["val_points"],
        ]
    )
    split_positive_counts = sorted(
        [
            report["split_preview"]["train_positive"],
            report["split_preview"]["val_positive"],
        ]
    )
    split_negative_counts = sorted(
        [
            report["split_preview"]["train_negative"],
            report["split_preview"]["val_negative"],
        ]
    )

    assert report["total_features"] == 2
    assert report["raw_label_counts"] == {"forest": 2}
    assert report["valid_label_counts"] == {"forest": 2}
    assert report["valid_point_count"] == 2
    assert report["class_balance"] == {"positive": 2, "negative": 0}
    assert report["split_preview"]["split_possible"] is True
    assert report["split_preview"]["grid_count"] == 2
    assert report["split_preview"]["train_grid_count"] == 1
    assert report["split_preview"]["val_grid_count"] == 1
    assert split_point_counts == [1, 1]
    assert split_positive_counts == [1, 1]
    assert split_negative_counts == [0, 0]
    assert report["risk"]["status"] == "block"
    assert report["risk"]["can_run"] is False
    assert set(report["risk"]["warnings"]) == {"有效标签点过少，结果方差可能较大。"}
    assert set(report["risk"]["blocking"]) == {
        "有效标签缺少正类或负类，无法形成二分类监督。",
        "训练集缺少正类或负类，请调整网格或切分比例。",
        "验证集缺少正类或负类，请调整网格或切分比例。",
    }


def test_split_points_by_grid_reports_impossible_for_single_grid():
    points = [
        labels_module.LabelPoint(x=0.0, y=0.0, label=1, grid_id="0_0", properties={}),
        labels_module.LabelPoint(x=1.0, y=1.0, label=0, grid_id="0_0", properties={}),
    ]

    train_points, val_points, meta = labels_module.split_points_by_grid(points, train_ratio=0.7, seed=42)

    assert {pt.grid_id for pt in train_points}.isdisjoint({pt.grid_id for pt in val_points})
    assert meta["split_possible"] is False
    assert meta["grid_count"] == 1
    assert meta["train_grid_count"] == 1
    assert meta["val_grid_count"] == 0
    assert len(train_points) == 2
    assert len(val_points) == 0


def test_split_points_by_grid_is_reproducible_for_seed():
    points = [
        labels_module.LabelPoint(x=0.0, y=0.0, label=1, grid_id="0_0", properties={}),
        labels_module.LabelPoint(x=1.0, y=1.0, label=0, grid_id="0_0", properties={}),
        labels_module.LabelPoint(x=1000.0, y=0.0, label=1, grid_id="1_0", properties={}),
        labels_module.LabelPoint(x=1001.0, y=1.0, label=0, grid_id="1_0", properties={}),
        labels_module.LabelPoint(x=2000.0, y=0.0, label=1, grid_id="2_0", properties={}),
        labels_module.LabelPoint(x=2001.0, y=1.0, label=0, grid_id="2_0", properties={}),
        labels_module.LabelPoint(x=3000.0, y=0.0, label=1, grid_id="3_0", properties={}),
        labels_module.LabelPoint(x=3001.0, y=1.0, label=0, grid_id="3_0", properties={}),
    ]

    train_points_first, val_points_first, meta_first = labels_module.split_points_by_grid(
        points, train_ratio=0.7, seed=42
    )
    train_points_second, val_points_second, meta_second = labels_module.split_points_by_grid(
        points, train_ratio=0.7, seed=42
    )

    train_grid_ids_first = {pt.grid_id for pt in train_points_first}
    val_grid_ids_first = {pt.grid_id for pt in val_points_first}
    train_grid_ids_second = {pt.grid_id for pt in train_points_second}
    val_grid_ids_second = {pt.grid_id for pt in val_points_second}

    split_assignments = {
        (
            frozenset(pt.grid_id for pt in train_points),
            frozenset(pt.grid_id for pt in val_points),
        )
        for seed in range(10)
        for train_points, val_points, _ in [labels_module.split_points_by_grid(points, train_ratio=0.7, seed=seed)]
    }

    assert train_grid_ids_first.isdisjoint(val_grid_ids_first)
    assert len(train_grid_ids_first) == 3
    assert len(val_grid_ids_first) == 1
    assert train_grid_ids_first == train_grid_ids_second
    assert val_grid_ids_first == val_grid_ids_second
    assert meta_first == meta_second
    assert meta_first["split_possible"] is True
    assert meta_first["grid_count"] == 4
    assert meta_first["train_grid_count"] == 3
    assert meta_first["val_grid_count"] == 1
    assert meta_first["train_points"] == 6
    assert meta_first["val_points"] == 2

    assert len(split_assignments) > 1


def test_split_points_by_grid_clamps_extreme_train_ratios_to_keep_both_sides():
    points = [
        labels_module.LabelPoint(x=0.0, y=0.0, label=1, grid_id="0_0", properties={}),
        labels_module.LabelPoint(x=1000.0, y=0.0, label=0, grid_id="1_0", properties={}),
        labels_module.LabelPoint(x=2000.0, y=0.0, label=1, grid_id="2_0", properties={}),
    ]

    train_points_low, val_points_low, meta_low = labels_module.split_points_by_grid(points, train_ratio=0.0, seed=42)
    train_points_high, val_points_high, meta_high = labels_module.split_points_by_grid(points, train_ratio=1.0, seed=42)

    assert {pt.grid_id for pt in train_points_low}.isdisjoint({pt.grid_id for pt in val_points_low})
    assert {pt.grid_id for pt in train_points_high}.isdisjoint({pt.grid_id for pt in val_points_high})

    assert meta_low["split_possible"] is True
    assert meta_low["grid_count"] == 3
    assert meta_low["train_grid_count"] == 1
    assert meta_low["val_grid_count"] == 2
    assert meta_low["train_points"] == 1
    assert meta_low["val_points"] == 2

    assert meta_high["split_possible"] is True
    assert meta_high["grid_count"] == 3
    assert meta_high["train_grid_count"] == 2
    assert meta_high["val_grid_count"] == 1
    assert meta_high["train_points"] == 2
    assert meta_high["val_points"] == 1


def test_save_and_load_points_json_round_trip(tmp_path):
    output_path = tmp_path / "nested" / "points.json"
    points = [
        labels_module.LabelPoint(
            x=12.5,
            y=34.5,
            label=1,
            grid_id="0_0",
            properties={"note": "森林", "score": 0.75},
        ),
        labels_module.LabelPoint(
            x=56.0,
            y=78.0,
            label=0,
            grid_id="1_2",
            properties={"note": "non_forest", "source": "manual"},
        ),
    ]
    meta = {"selected_layer": "labels", "target_crs": "EPSG:3857"}

    returned_path = labels_module.save_points_json(str(output_path), points, meta=meta)
    loaded_points, loaded_meta = labels_module.load_points_json(str(output_path))

    assert returned_path == str(output_path)
    assert output_path.exists()
    assert loaded_meta == meta
    assert loaded_points == [
        {
            "x": 12.5,
            "y": 34.5,
            "label": 1,
            "grid_id": "0_0",
            "properties": {"note": "森林", "score": 0.75},
        },
        {
            "x": 56.0,
            "y": 78.0,
            "label": 0,
            "grid_id": "1_2",
            "properties": {"note": "non_forest", "source": "manual"},
        },
    ]


def test_load_points_json_defaults_missing_sections_to_empty_collections(tmp_path):
    points_path = tmp_path / "points.json"
    points_path.write_text("{}", encoding="utf-8")

    loaded_points, loaded_meta = labels_module.load_points_json(str(points_path))

    assert loaded_points == []
    assert loaded_meta == {}


def test_save_points_json_defaults_meta_to_empty_object(tmp_path):
    output_path = tmp_path / "points.json"
    points = [
        labels_module.LabelPoint(
            x=1.0,
            y=2.0,
            label=1,
            grid_id="0_0",
            properties={"source": "manual"},
        )
    ]

    labels_module.save_points_json(str(output_path), points)
    loaded_points, loaded_meta = labels_module.load_points_json(str(output_path))

    assert loaded_meta == {}
    assert loaded_points == [
        {
            "x": 1.0,
            "y": 2.0,
            "label": 1,
            "grid_id": "0_0",
            "properties": {"source": "manual"},
        }
    ]


def test_save_points_json_creates_parent_directory_before_writing(tmp_path):
    output_path = tmp_path / "nested" / "points.json"

    returned_path = labels_module.save_points_json(str(output_path), [], meta={"source": "test"})
    loaded_points, loaded_meta = labels_module.load_points_json(str(output_path))

    assert returned_path == str(output_path)
    assert output_path.exists()
    assert loaded_points == []
    assert loaded_meta == {"source": "test"}


def test_sample_raster_at_points_returns_first_band_values_in_input_order(monkeypatch):
    points = [
        {"x": 10, "y": 20},
        {"x": 30.5, "y": 40.25},
    ]
    opened_paths = []
    sampled_calls = []

    class DummyRasterDataset:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def sample(self, coords, indexes=1):
            sampled_calls.append((list(coords), indexes))
            return iter(([1], [2.5]))

    def fake_open(path):
        opened_paths.append(path)
        return DummyRasterDataset()

    monkeypatch.setattr(labels_module.rasterio, "open", fake_open)

    values = labels_module.sample_raster_at_points("dummy.tif", points)

    assert opened_paths == ["dummy.tif"]
    assert sampled_calls == [([(10.0, 20.0), (30.5, 40.25)], 1)]
    assert values == [1.0, 2.5]


def test_sample_raster_at_points_coerces_coordinate_inputs_and_sampled_values_to_float(monkeypatch):
    points = [
        {"x": 1, "y": "2.5"},
        {"x": "3.75", "y": 4},
    ]
    sampled_calls = []

    class DummyRasterDataset:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def sample(self, coords, indexes=1):
            sampled_calls.append((list(coords), indexes))
            return iter(([7], [8]))

    monkeypatch.setattr(labels_module.rasterio, "open", lambda path: DummyRasterDataset())

    values = labels_module.sample_raster_at_points("dummy.tif", points)

    assert sampled_calls == [([(1.0, 2.5), (3.75, 4.0)], 1)]
    assert values == [7.0, 8.0]


def test_sample_raster_at_points_uses_first_value_from_each_sample_vector(monkeypatch):
    class DummyRasterDataset:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def sample(self, coords, indexes=1):
            return iter(([1.5, 99.0], [2.25, 88.0]))

    monkeypatch.setattr(labels_module.rasterio, "open", lambda path: DummyRasterDataset())

    values = labels_module.sample_raster_at_points("dummy.tif", [{"x": 1, "y": 2}, {"x": 3, "y": 4}])

    assert values == [1.5, 2.25]


def test_sample_raster_at_points_returns_empty_list_for_empty_points(monkeypatch):
    sampled_calls = []

    class DummyRasterDataset:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def sample(self, coords, indexes=1):
            sampled_calls.append((list(coords), indexes))
            return iter(())

    monkeypatch.setattr(labels_module.rasterio, "open", lambda path: DummyRasterDataset())

    values = labels_module.sample_raster_at_points("dummy.tif", [])

    assert sampled_calls == [([], 1)]
    assert values == []


def test_raster_xy_to_rowcol_returns_dataset_index_as_ints():
    index_calls = []

    class DummyRasterDataset:
        def index(self, x, y):
            index_calls.append((x, y))
            return 12.0, 34.0

    row, col = labels_module.raster_xy_to_rowcol(DummyRasterDataset(), 10.5, 20.25)

    assert index_calls == [(10.5, 20.25)]
    assert row == 12
    assert col == 34


def test_export_points_preview_replaces_existing_file_and_writes_points(monkeypatch, tmp_path):
    output_path = tmp_path / "preview.gpkg"
    output_path.write_text("old", encoding="utf-8")
    points = [
        labels_module.LabelPoint(x=10.5, y=20.25, label=1, grid_id="0_0", properties={"ignored": True}),
        labels_module.LabelPoint(x=30.0, y=40.0, label=0, grid_id="1_1", properties={}),
    ]
    crs = labels_module.CRS.from_epsg(3857)
    removed_paths = []
    open_calls = []
    written_features = []

    class DummySink:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, feature):
            written_features.append(feature)

    def fake_remove(path):
        removed_paths.append(path)

    def fake_open(path, mode, driver, crs_wkt, schema, layer):
        open_calls.append(
            {
                "path": path,
                "mode": mode,
                "driver": driver,
                "crs_wkt": crs_wkt,
                "schema": schema,
                "layer": layer,
            }
        )
        return DummySink()

    monkeypatch.setattr(labels_module.os, "remove", fake_remove)
    monkeypatch.setattr(labels_module.fiona, "open", fake_open)

    returned_path = labels_module.export_points_preview(str(output_path), points, crs, layer="preview")

    assert returned_path == str(output_path)
    assert removed_paths == [str(output_path)]
    assert open_calls == [
        {
            "path": str(output_path),
            "mode": "w",
            "driver": "GPKG",
            "crs_wkt": crs.to_wkt(),
            "schema": {"geometry": "Point", "properties": {"label": "int", "grid_id": "str"}},
            "layer": "preview",
        }
    ]
    assert written_features == [
        {
            "geometry": {"type": "Point", "coordinates": (10.5, 20.25)},
            "properties": {"label": 1, "grid_id": "0_0"},
        },
        {
            "geometry": {"type": "Point", "coordinates": (30.0, 40.0)},
            "properties": {"label": 0, "grid_id": "1_1"},
        },
    ]


def test_export_points_preview_uses_none_crs_wkt_and_default_layer_for_new_file(monkeypatch, tmp_path):
    output_path = tmp_path / "preview.gpkg"
    open_calls = []
    written_features = []

    class DummySink:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, feature):
            written_features.append(feature)

    def fail_remove(path):
        raise AssertionError("os.remove should not be called when preview file does not exist")

    def fake_open(path, mode, driver, crs_wkt, schema, layer):
        open_calls.append(
            {
                "path": path,
                "mode": mode,
                "driver": driver,
                "crs_wkt": crs_wkt,
                "schema": schema,
                "layer": layer,
            }
        )
        return DummySink()

    monkeypatch.setattr(labels_module.os, "remove", fail_remove)
    monkeypatch.setattr(labels_module.fiona, "open", fake_open)

    returned_path = labels_module.export_points_preview(
        str(output_path),
        [labels_module.LabelPoint(x=1.0, y=2.0, label=1, grid_id="0_0", properties={})],
        None,
    )

    assert returned_path == str(output_path)
    assert open_calls == [
        {
            "path": str(output_path),
            "mode": "w",
            "driver": "GPKG",
            "crs_wkt": None,
            "schema": {"geometry": "Point", "properties": {"label": "int", "grid_id": "str"}},
            "layer": "labels",
        }
    ]
    assert written_features == [
        {
            "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
            "properties": {"label": 1, "grid_id": "0_0"},
        }
    ]


def test_export_points_preview_coerces_label_and_grid_id_for_output(monkeypatch, tmp_path):
    output_path = tmp_path / "preview.gpkg"
    written_features = []

    class DummySink:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, feature):
            written_features.append(feature)

    monkeypatch.setattr(labels_module.fiona, "open", lambda *args, **kwargs: DummySink())

    labels_module.export_points_preview(
        str(output_path),
        [labels_module.LabelPoint(x=1.5, y=2.5, label=True, grid_id=7, properties={"ignored": "value"})],
        None,
    )

    assert written_features == [
        {
            "geometry": {"type": "Point", "coordinates": (1.5, 2.5)},
            "properties": {"label": 1, "grid_id": "7"},
        }
    ]


def test_export_points_preview_writes_no_features_for_empty_points(monkeypatch, tmp_path):
    output_path = tmp_path / "preview.gpkg"
    write_calls = []

    class DummySink:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, feature):
            write_calls.append(feature)

    monkeypatch.setattr(labels_module.fiona, "open", lambda *args, **kwargs: DummySink())

    returned_path = labels_module.export_points_preview(str(output_path), [], None)

    assert returned_path == str(output_path)
    assert write_calls == []


def test_export_points_preview_creates_parent_directory_before_open(monkeypatch, tmp_path):
    output_path = tmp_path / "nested" / "preview.gpkg"
    mkdir_calls = []
    open_calls = []

    class DummySink:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, feature):
            return None

    def fake_makedirs(path, exist_ok=False):
        mkdir_calls.append((path, exist_ok))

    def fake_open(path, mode, driver, crs_wkt, schema, layer):
        open_calls.append(path)
        return DummySink()

    monkeypatch.setattr(labels_module.os, "makedirs", fake_makedirs)
    monkeypatch.setattr(labels_module.fiona, "open", fake_open)

    labels_module.export_points_preview(str(output_path), [], None)

    assert mkdir_calls == [(str(output_path.parent), True)]
    assert open_calls == [str(output_path)]


def test_read_label_points_requires_existing_file(tmp_path):
    missing = tmp_path / "missing.gpkg"
    with pytest.raises(FileNotFoundError, match="标签文件不存在"):
        labels_module.read_label_points(str(missing))


def test_validate_label_points_requires_existing_file(tmp_path):
    missing = tmp_path / "missing.gpkg"
    with pytest.raises(FileNotFoundError):
        labels_module.validate_label_points(str(missing))
