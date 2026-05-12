#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from pathlib import Path


def _open_gpkg_readwrite(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {path}")
    if path.suffix.lower() != ".gpkg":
        raise ValueError(f"Expected .gpkg file: {path}")
    uri = f"file:{path.as_posix()}?mode=rw"
    conn = sqlite3.connect(uri, uri=True)
    conn.create_function("ST_IsEmpty", 1, lambda _geom: 0)
    conn.create_function("ST_MinX", 1, lambda _geom: 0.0)
    conn.create_function("ST_MinY", 1, lambda _geom: 0.0)
    conn.create_function("ST_MaxX", 1, lambda _geom: 0.0)
    conn.create_function("ST_MaxY", 1, lambda _geom: 0.0)
    return conn


def _update_class(path: Path, value: int, layer: str = "labels", field: str = "class") -> int:
    conn = _open_gpkg_readwrite(path)
    try:
        cur = conn.cursor()
        tables = {row[0] for row in cur.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'")}
        if layer not in tables:
            raise ValueError(f"Layer not found in {path}: {layer}")

        columns = {row[1] for row in cur.execute(f'PRAGMA table_info("{layer}")')}
        if field not in columns:
            raise ValueError(f"Field not found in {path}: {field}")

        cur.execute(f'UPDATE "{layer}" SET "{field}" = ?', (value,))
        count = cur.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


def main() -> None:
    base = Path("E:/CORONA/DeepLearning/labels")
    jobs = [
        (base / "label_points.gpkg", 1),
        (base / "label_points_0.gpkg", 0),
    ]
    for path, value in jobs:
        updated = _update_class(path=path, value=value)
        print(f"{path} => class={value}, updated={updated}")


if __name__ == "__main__":
    main()
