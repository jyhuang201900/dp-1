from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fiona
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import Point, mapping, shape

VALID_GEOMETRY_TYPE = "Point"
MIN_SPLIT_CLASS_WARNING_COUNT = 3
SEVERE_CLASS_IMBALANCE_RATIO = 4.0


@dataclass(frozen=True)
class LabelReadOptions:
    path: str
    class_field: str = "class"
    positive_value: str = "1"
    negative_value: str = "0"
    target_crs: CRS | None = None
    grid_size: float = 1000.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    layer: str | None = None


def _try_parse_numeric_label(value: Any) -> float | None:
    try:
        numeric_value = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


@dataclass
class LabelPoint:
    x: float
    y: float
    label: int
    grid_id: str
    properties: dict[str, Any]


@dataclass
class LabelReadResult:
    available_layers: list[str]
    selected_layer: str
    source_crs: CRS | None
    crs_missing_for_target: bool
    geometry_counts: Counter[str]
    raw_label_counts: Counter[str]
    valid_label_counts: Counter[str]
    unexpected_label_counts: Counter[str]
    missing_class_count: int
    total_features: int
    valid_points: list[LabelPoint]


def _normalize_label(value: Any, positive_value: str, negative_value: str) -> int:
    s = str(value).strip().lower()
    if s == str(positive_value).strip().lower():
        return 1
    if s == str(negative_value).strip().lower():
        return 0
    raise ValueError(f"Unsupported label value: {value}")


def _label_value_conflict_error(positive_value: str, negative_value: str) -> ValueError:
    return ValueError(f"标签正负类取值不能相同：positive_value={positive_value}, negative_value={negative_value}。")


def _label_read_error(path: str, class_field: str, positive_value: str, negative_value: str) -> ValueError:
    return ValueError(
        f"读取标签失败：{path}。请检查图层名是否正确、几何是否为点、字段 {class_field} 是否存在，且森林标签为 {positive_value}、非森林标签使用其他数值（例如 {negative_value}）。"
    )


def _label_empty_error(path: str, class_field: str, positive_value: str, negative_value: str) -> ValueError:
    return ValueError(
        f"标签文件中没有有效点：{path}。请确认图层包含 Point 几何，并且字段 {class_field} 使用 {positive_value} 表示森林、其他数值（例如 {negative_value}）表示非森林。"
    )


def _label_crs_missing_blocking_message(target_crs: CRS) -> str:
    return f"标签图层缺少源 CRS，无法转换到目标坐标系 {target_crs.to_string()}。请先为标签图层补充正确坐标系后再运行。"


def _label_crs_missing_error(path: str, target_crs: CRS) -> ValueError:
    return ValueError(
        f"标签文件缺少 CRS：{path}。当前需要将标签转换到 {target_crs.to_string()}，请先为标签图层补充正确坐标系后再运行。"
    )


def _read_label_dataset(options: LabelReadOptions) -> LabelReadResult:
    if not os.path.exists(options.path):
        raise FileNotFoundError(
            f"标签文件不存在：{options.path}。请先准备 GPKG 点标签文件，并确认 pipeline.yaml 中 labels.path 配置正确。"
        )

    normalized_positive = str(options.positive_value).strip().lower()
    normalized_negative = str(options.negative_value).strip().lower()
    if normalized_positive == normalized_negative:
        raise _label_value_conflict_error(str(options.positive_value), str(options.negative_value))

    available_layers: list[str]
    if options.layer is not None:
        available_layers = [options.layer]
        selected_layer = options.layer
    else:
        available_layers = list(fiona.listlayers(options.path))
        selected_layer = available_layers[0] if available_layers else None
    if selected_layer is None:
        raise ValueError(f"标签文件没有可用图层：{options.path}。")

    normalized_positive = str(options.positive_value).strip().lower()
    normalized_negative = str(options.negative_value).strip().lower()

    expected_labels = {
        normalized_positive: 1,
        normalized_negative: 0,
    }
    geometry_counts: Counter[str] = Counter()
    raw_label_counts: Counter[str] = Counter()
    valid_label_counts: Counter[str] = Counter()
    unexpected_label_counts: Counter[str] = Counter()
    missing_class_count = 0
    total_features = 0
    valid_points: list[LabelPoint] = []

    crs_missing_for_target = False

    try:
        with fiona.open(options.path, layer=selected_layer) as src:
            src_crs = CRS.from_user_input(src.crs) if src.crs else None
            if options.target_crs is not None and src_crs is None:
                crs_missing_for_target = True
            for feat in src:
                total_features += 1
                geom = feat.get("geometry")
                geom_type = str((geom or {}).get("type") or "None")
                geometry_counts[geom_type] += 1
                if geom is None:
                    continue
                props = dict(feat.get("properties") or {})
                raw_value = props.get(options.class_field)
                if raw_value is None or str(raw_value).strip() == "":
                    missing_class_count += 1
                    continue
                raw_label = str(raw_value).strip()
                raw_label_counts[raw_label] += 1
                normalized = raw_label.lower()
                numeric_label = _try_parse_numeric_label(raw_value)
                positive_numeric = _try_parse_numeric_label(options.positive_value)
                negative_numeric = _try_parse_numeric_label(options.negative_value)
                if numeric_label is not None and positive_numeric is not None and negative_numeric is not None:
                    label_value = 1 if numeric_label == positive_numeric else 0
                else:
                    label_value = expected_labels.get(normalized)
                if label_value is None:
                    unexpected_label_counts[raw_label] += 1
                    continue
                transformed_geom = geom
                if options.target_crs is not None:
                    if src_crs is None:
                        continue
                    if src_crs != options.target_crs:
                        transformed_geom = transform_geom(src_crs, options.target_crs, geom)
                g = shape(transformed_geom)
                if g.geom_type != VALID_GEOMETRY_TYPE:
                    continue
                gid = _grid_id(g.x, g.y, options.origin_x, options.origin_y, options.grid_size)
                valid_points.append(
                    LabelPoint(x=float(g.x), y=float(g.y), label=label_value, grid_id=gid, properties=props)
                )
                valid_label_counts[raw_label] += 1
    except Exception as exc:
        raise _label_read_error(
            options.path, options.class_field, options.positive_value, options.negative_value
        ) from exc

    return LabelReadResult(
        available_layers=available_layers,
        selected_layer=selected_layer,
        source_crs=src_crs if "src_crs" in locals() else None,
        crs_missing_for_target=crs_missing_for_target,
        geometry_counts=geometry_counts,
        raw_label_counts=raw_label_counts,
        valid_label_counts=valid_label_counts,
        unexpected_label_counts=unexpected_label_counts,
        missing_class_count=missing_class_count,
        total_features=total_features,
        valid_points=valid_points,
    )


def _grid_id(x: float, y: float, origin_x: float, origin_y: float, grid_size: float) -> str:
    gx = math.floor((x - origin_x) / grid_size)
    gy = math.floor((y - origin_y) / grid_size)
    return f"{gx}_{gy}"


def read_label_points(
    path: str,
    class_field: str = "class",
    positive_value: str = "1",
    negative_value: str = "0",
    target_crs: CRS | None = None,
    grid_size: float = 1000.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    layer: str | None = None,
) -> list[LabelPoint]:
    result = _read_label_dataset(
        LabelReadOptions(
            path=path,
            class_field=class_field,
            positive_value=positive_value,
            negative_value=negative_value,
            target_crs=target_crs,
            grid_size=grid_size,
            origin_x=origin_x,
            origin_y=origin_y,
            layer=layer,
        )
    )
    if result.crs_missing_for_target and target_crs is not None:
        raise _label_crs_missing_error(path, target_crs)
    geometry_counts = dict(result.geometry_counts)
    non_point_feature_count = sum(
        int(count) for geom_type, count in geometry_counts.items() if geom_type != VALID_GEOMETRY_TYPE
    )
    if result.unexpected_label_counts or result.missing_class_count > 0 or non_point_feature_count > 0:
        raise _label_read_error(path, class_field, positive_value, negative_value)
    if not result.valid_points:
        raise _label_empty_error(path, class_field, positive_value, negative_value)
    return result.valid_points


def _label_dual_read_error(
    positive_path: str,
    negative_path: str,
    class_field: str,
    positive_value: str,
    negative_value: str,
) -> ValueError:
    return ValueError(
        "读取双标签失败："
        f"positive_path={positive_path}, negative_path={negative_path}。"
        f"请检查图层名是否正确、几何是否为点、字段 {class_field} 是否存在，"
        f"且森林标签为 {positive_value}、非森林标签使用其他数值（例如 {negative_value}）。"
    )


def _force_label(points: list[LabelPoint], label: int) -> list[LabelPoint]:
    return [
        LabelPoint(
            x=pt.x,
            y=pt.y,
            label=label,
            grid_id=pt.grid_id,
            properties=dict(pt.properties),
        )
        for pt in points
    ]


def _merge_label_results(positive_result: LabelReadResult, negative_result: LabelReadResult) -> LabelReadResult:
    available_layers = [
        *[f"positive:{name}" for name in positive_result.available_layers],
        *[f"negative:{name}" for name in negative_result.available_layers],
    ]
    selected_layer = f"positive:{positive_result.selected_layer}|negative:{negative_result.selected_layer}"
    source_crs = positive_result.source_crs
    if (
        positive_result.source_crs is None
        or negative_result.source_crs is None
        or positive_result.source_crs != negative_result.source_crs
    ):
        source_crs = None

    merged_points = [
        *_force_label(positive_result.valid_points, 1),
        *_force_label(negative_result.valid_points, 0),
    ]

    return LabelReadResult(
        available_layers=available_layers,
        selected_layer=selected_layer,
        source_crs=source_crs,
        crs_missing_for_target=positive_result.crs_missing_for_target or negative_result.crs_missing_for_target,
        geometry_counts=positive_result.geometry_counts + negative_result.geometry_counts,
        raw_label_counts=positive_result.raw_label_counts + negative_result.raw_label_counts,
        valid_label_counts=positive_result.valid_label_counts + negative_result.valid_label_counts,
        unexpected_label_counts=positive_result.unexpected_label_counts + negative_result.unexpected_label_counts,
        missing_class_count=positive_result.missing_class_count + negative_result.missing_class_count,
        total_features=positive_result.total_features + negative_result.total_features,
        valid_points=merged_points,
    )


def read_label_points_from_two_files(
    positive_path: str,
    negative_path: str,
    class_field: str = "class",
    positive_value: str = "1",
    negative_value: str = "0",
    target_crs: CRS | None = None,
    grid_size: float = 1000.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    positive_layer: str | None = None,
    negative_layer: str | None = None,
) -> list[LabelPoint]:
    try:
        positive_result = _read_label_dataset(
            LabelReadOptions(
                path=positive_path,
                class_field=class_field,
                positive_value=positive_value,
                negative_value=negative_value,
                target_crs=target_crs,
                grid_size=grid_size,
                origin_x=origin_x,
                origin_y=origin_y,
                layer=positive_layer,
            )
        )
        negative_result = _read_label_dataset(
            LabelReadOptions(
                path=negative_path,
                class_field=class_field,
                positive_value=positive_value,
                negative_value=negative_value,
                target_crs=target_crs,
                grid_size=grid_size,
                origin_x=origin_x,
                origin_y=origin_y,
                layer=negative_layer,
            )
        )
    except Exception as exc:
        raise _label_dual_read_error(positive_path, negative_path, class_field, positive_value, negative_value) from exc

    result = _merge_label_results(positive_result, negative_result)
    if result.crs_missing_for_target and target_crs is not None:
        path_hint = positive_path if positive_result.crs_missing_for_target else negative_path
        raise _label_crs_missing_error(path_hint, target_crs)

    geometry_counts = dict(result.geometry_counts)
    non_point_feature_count = sum(
        int(count) for geom_type, count in geometry_counts.items() if geom_type != VALID_GEOMETRY_TYPE
    )
    if result.unexpected_label_counts or result.missing_class_count > 0 or non_point_feature_count > 0:
        raise _label_dual_read_error(positive_path, negative_path, class_field, positive_value, negative_value)
    if not result.valid_points:
        raise _label_empty_error(
            f"{positive_path} | {negative_path}",
            class_field,
            positive_value,
            negative_value,
        )
    return result.valid_points


def _build_validation_report(
    result: LabelReadResult,
    *,
    path: str,
    class_field: str,
    positive_value: str,
    negative_value: str,
    target_crs: CRS | None,
    train_ratio: float,
    seed: int,
) -> dict[str, Any]:
    train_points, val_points, split_meta = split_points_by_grid(result.valid_points, train_ratio=train_ratio, seed=seed)
    class_balance = {
        "positive": sum(1 for pt in result.valid_points if pt.label == 1),
        "negative": sum(1 for pt in result.valid_points if pt.label == 0),
    }
    split_preview = {
        **split_meta,
        "train_positive": sum(1 for pt in train_points if pt.label == 1),
        "train_negative": sum(1 for pt in train_points if pt.label == 0),
        "val_positive": sum(1 for pt in val_points if pt.label == 1),
        "val_negative": sum(1 for pt in val_points if pt.label == 0),
    }
    split_preview["split_possible"] = bool(train_points) and bool(val_points)
    report = {
        "path": path,
        "selected_layer": result.selected_layer,
        "available_layers": result.available_layers,
        "source_crs": result.source_crs.to_string() if result.source_crs else None,
        "target_crs": target_crs.to_string() if target_crs else None,
        "class_field": class_field,
        "positive_value": positive_value,
        "negative_value": negative_value,
        "crs_missing_for_target": result.crs_missing_for_target,
        "total_features": result.total_features,
        "geometry_counts": dict(result.geometry_counts),
        "raw_label_counts": dict(result.raw_label_counts),
        "valid_label_counts": dict(result.valid_label_counts),
        "unexpected_label_counts": dict(result.unexpected_label_counts),
        "missing_class_count": result.missing_class_count,
        "valid_point_count": len(result.valid_points),
        "class_balance": class_balance,
        "split_preview": split_preview,
    }
    risk = _assess_label_report_risk(report)
    return {**report, "risk": risk}


def validate_label_points_from_two_files(
    positive_path: str,
    negative_path: str,
    class_field: str = "class",
    positive_value: str = "1",
    negative_value: str = "0",
    target_crs: CRS | None = None,
    grid_size: float = 1000.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    positive_layer: str | None = None,
    negative_layer: str | None = None,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> dict[str, Any]:
    positive_result = _read_label_dataset(
        LabelReadOptions(
            path=positive_path,
            class_field=class_field,
            positive_value=positive_value,
            negative_value=negative_value,
            target_crs=target_crs,
            grid_size=grid_size,
            origin_x=origin_x,
            origin_y=origin_y,
            layer=positive_layer,
        )
    )
    negative_result = _read_label_dataset(
        LabelReadOptions(
            path=negative_path,
            class_field=class_field,
            positive_value=positive_value,
            negative_value=negative_value,
            target_crs=target_crs,
            grid_size=grid_size,
            origin_x=origin_x,
            origin_y=origin_y,
            layer=negative_layer,
        )
    )
    result = _merge_label_results(positive_result, negative_result)
    return _build_validation_report(
        result,
        path=f"{Path(positive_path).name} + {Path(negative_path).name}",
        class_field=class_field,
        positive_value=positive_value,
        negative_value=negative_value,
        target_crs=target_crs,
        train_ratio=train_ratio,
        seed=seed,
    )


def validate_label_points(
    path: str,
    class_field: str = "class",
    positive_value: str = "1",
    negative_value: str = "0",
    target_crs: CRS | None = None,
    grid_size: float = 1000.0,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    layer: str | None = None,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> dict[str, Any]:
    result = _read_label_dataset(
        LabelReadOptions(
            path=path,
            class_field=class_field,
            positive_value=positive_value,
            negative_value=negative_value,
            target_crs=target_crs,
            grid_size=grid_size,
            origin_x=origin_x,
            origin_y=origin_y,
            layer=layer,
        )
    )

    train_points, val_points, split_meta = split_points_by_grid(result.valid_points, train_ratio=train_ratio, seed=seed)
    class_balance = {
        "positive": sum(1 for pt in result.valid_points if pt.label == 1),
        "negative": sum(1 for pt in result.valid_points if pt.label == 0),
    }
    split_preview = {
        **split_meta,
        "train_positive": sum(1 for pt in train_points if pt.label == 1),
        "train_negative": sum(1 for pt in train_points if pt.label == 0),
        "val_positive": sum(1 for pt in val_points if pt.label == 1),
        "val_negative": sum(1 for pt in val_points if pt.label == 0),
    }
    split_preview["split_possible"] = bool(train_points) and bool(val_points)
    report = {
        "path": path,
        "selected_layer": result.selected_layer,
        "available_layers": result.available_layers,
        "source_crs": result.source_crs.to_string() if result.source_crs else None,
        "target_crs": target_crs.to_string() if target_crs else None,
        "crs_missing_for_target": result.crs_missing_for_target,
        "class_field": class_field,
        "positive_value": positive_value,
        "negative_value": negative_value,
        "total_features": result.total_features,
        "geometry_counts": dict(result.geometry_counts),
        "raw_label_counts": dict(result.raw_label_counts),
        "valid_label_counts": dict(result.valid_label_counts),
        "unexpected_label_counts": dict(result.unexpected_label_counts),
        "missing_class_count": result.missing_class_count,
        "valid_point_count": len(result.valid_points),
        "class_balance": class_balance,
        "split_preview": split_preview,
    }
    risk = _assess_label_report_risk(report)
    return {**report, "risk": risk}


def _assess_label_report_risk(report: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    blocking: list[str] = []

    class_field = str(report.get("class_field") or "class")
    target_crs_value = str(report.get("target_crs") or "")
    crs_missing_for_target = bool(report.get("crs_missing_for_target", False))
    valid_point_count = int(report.get("valid_point_count", 0))
    class_balance = dict(report.get("class_balance") or {})
    split_preview = dict(report.get("split_preview") or {})
    unexpected_label_counts = dict(report.get("unexpected_label_counts") or {})
    missing_class_count = int(report.get("missing_class_count", 0))
    geometry_counts = dict(report.get("geometry_counts") or {})
    non_point_feature_count = sum(
        int(count) for geom_type, count in geometry_counts.items() if geom_type != VALID_GEOMETRY_TYPE
    )

    positive_count = int(class_balance.get("positive", 0))
    negative_count = int(class_balance.get("negative", 0))
    train_positive = int(split_preview.get("train_positive", 0))
    train_negative = int(split_preview.get("train_negative", 0))
    val_positive = int(split_preview.get("val_positive", 0))
    val_negative = int(split_preview.get("val_negative", 0))

    if unexpected_label_counts:
        blocking.append("存在未识别标签值，请先清洗标签字段后再运行。")
    if missing_class_count > 0:
        blocking.append(f"存在缺失 {class_field} 字段的要素，请补齐后再运行。")
    if crs_missing_for_target and target_crs_value:
        blocking.append(_label_crs_missing_blocking_message(CRS.from_user_input(target_crs_value)))
    if non_point_feature_count > 0:
        blocking.append("标签图层包含非 Point 几何，请移除或转换为点后再运行。")

    if crs_missing_for_target:
        status = "block" if blocking else "warning" if warnings else "ok"
        return {
            "status": status,
            "can_run": not blocking,
            "warnings": warnings,
            "blocking": blocking,
        }

    if valid_point_count < 20:
        warnings.append("有效标签点过少，结果方差可能较大。")

    if positive_count == 0 or negative_count == 0:
        blocking.append("有效标签缺少正类或负类，无法形成二分类监督。")
    if not bool(split_preview.get("split_possible", True)):
        blocking.append("无法按网格形成同时包含训练集和验证集的空间切分，请增加更多网格标签。")
    if train_positive == 0 or train_negative == 0:
        blocking.append("训练集缺少正类或负类，请调整网格或切分比例。")
    if val_positive == 0 or val_negative == 0:
        blocking.append("验证集缺少正类或负类，请调整网格或切分比例。")

    if not blocking:
        minority_count = min(positive_count, negative_count)
        majority_count = max(positive_count, negative_count)
        if valid_point_count >= 20 and minority_count > 0:
            imbalance_ratio = majority_count / minority_count
            if imbalance_ratio >= SEVERE_CLASS_IMBALANCE_RATIO:
                warnings.append("有效标签类别分布严重失衡，少数类占比过低，模型可能偏向多数类。")

        train_minority = min(train_positive, train_negative)
        train_majority = max(train_positive, train_negative)
        if train_minority >= MIN_SPLIT_CLASS_WARNING_COUNT:
            train_imbalance_ratio = train_majority / train_minority
            if train_imbalance_ratio >= SEVERE_CLASS_IMBALANCE_RATIO:
                warnings.append("训练集类别分布严重失衡，少数类占比过低，训练可能偏向多数类。")
        if train_minority < MIN_SPLIT_CLASS_WARNING_COUNT:
            warnings.append("训练集某一类样本过少，监督信号可能不稳定。")

        val_minority = min(val_positive, val_negative)
        val_majority = max(val_positive, val_negative)
        if val_minority >= MIN_SPLIT_CLASS_WARNING_COUNT:
            val_imbalance_ratio = val_majority / val_minority
            if val_imbalance_ratio >= SEVERE_CLASS_IMBALANCE_RATIO:
                warnings.append("验证集类别分布严重失衡，少数类占比过低，评估可能偏向多数类。")
        if val_minority < MIN_SPLIT_CLASS_WARNING_COUNT:
            warnings.append("验证集某一类样本过少，评估指标可能不稳定。")

    status = "block" if blocking else "warning" if warnings else "ok"
    return {
        "status": status,
        "can_run": not blocking,
        "warnings": warnings,
        "blocking": blocking,
    }


def split_points_by_grid(
    points: list[LabelPoint],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[LabelPoint], list[LabelPoint], dict[str, Any]]:
    groups: dict[str, list[LabelPoint]] = defaultdict(list)
    for pt in points:
        groups[pt.grid_id].append(pt)

    grid_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(grid_ids)
    split_at = max(1, min(len(grid_ids) - 1, round(len(grid_ids) * train_ratio))) if len(grid_ids) > 1 else 1
    train_ids = set(grid_ids[:split_at])

    train_points: list[LabelPoint] = []
    val_points: list[LabelPoint] = []
    for gid, pts in groups.items():
        if gid in train_ids:
            train_points.extend(pts)
        else:
            val_points.extend(pts)

    train_grid_ids = {pt.grid_id for pt in train_points}
    val_grid_ids = {pt.grid_id for pt in val_points}
    split_possible = bool(train_points) and bool(val_points)

    meta = {
        "train_ratio": train_ratio,
        "seed": seed,
        "grid_count": len(grid_ids),
        "train_grid_count": len(train_grid_ids),
        "val_grid_count": len(val_grid_ids),
        "train_points": len(train_points),
        "val_points": len(val_points),
        "split_possible": split_possible,
    }
    return train_points, val_points, meta


def save_points_json(path: str, points: list[LabelPoint], meta: dict[str, Any] | None = None) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "meta": meta or {},
        "points": [
            {"x": pt.x, "y": pt.y, "label": pt.label, "grid_id": pt.grid_id, "properties": pt.properties}
            for pt in points
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_points_json(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    return list(obj.get("points", [])), dict(obj.get("meta", {}))


def export_points_preview(path: str, points: list[LabelPoint], crs: CRS, layer: str = "labels") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    schema = {"geometry": "Point", "properties": {"label": "int", "grid_id": "str"}}
    with fiona.open(
        path, mode="w", driver="GPKG", crs_wkt=crs.to_wkt() if crs else None, schema=schema, layer=layer
    ) as sink:
        for pt in points:
            sink.write(
                {
                    "geometry": mapping(Point(pt.x, pt.y)),
                    "properties": {"label": int(pt.label), "grid_id": str(pt.grid_id)},
                }
            )
    return path


def raster_xy_to_rowcol(dataset: rasterio.DatasetReader, x: float, y: float) -> tuple[int, int]:
    row, col = dataset.index(x, y)
    return int(row), int(col)


def sample_raster_at_points(raster_path: str, points: list[dict[str, Any]]) -> list[float]:
    coords = [(float(p["x"]), float(p["y"])) for p in points]
    with rasterio.open(raster_path) as ds:
        values = []
        for val in ds.sample(coords, indexes=1):
            values.append(float(val[0]))
    return values
