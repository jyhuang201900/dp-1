from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass


@dataclass
class SceneRuntime:
    scene_id: str
    cropped_raster: str
    generated_export_dir: str
    input_tif: str


def _windows_to_wsl_path(path: str) -> str:
    p = path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        if rest.startswith("/"):
            rest = rest[1:]
        return f"/mnt/{drive}/{rest}"
    return p


def _wsl_to_windows_path(path: str) -> str:
    p = path.replace("\\", "/")
    prefix = "/mnt/"
    if p.startswith(prefix) and len(p) > len(prefix) + 1:
        drive = p[len(prefix)]
        rest = p[len(prefix) + 1 :]
        return f"{drive.upper()}:/{rest}"
    return path


def _parse_scene_file_fallback(scene_sh: str) -> dict[str, str]:
    with open(scene_sh, encoding="utf-8") as f:
        text = f.read()

    def _pick(name: str) -> str | None:
        m = re.search(rf'^\s*{name}="([^"]*)"', text, flags=re.MULTILINE)
        return m.group(1) if m else None

    scene_id = _pick("SCENE_ID") or ""
    seg = _pick("SEGMENT_SUFFIX") or ""

    cropped_raw = _pick("CROPPED_RASTER") or ""
    if cropped_raw and ("${" not in cropped_raw):
        cropped = cropped_raw
    else:
        cropped = f"{scene_id}_{seg}_stitched_calc" if scene_id and seg else ""

    export_dir_raw = _pick("GENERATED_EXPORT_DIR") or ""
    if export_dir_raw and ("${" not in export_dir_raw):
        export_dir = export_dir_raw.replace("\\", "/")
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(scene_sh))).replace("\\", "/")
        export_dir = f"{project_root}/input/generated/export"

    return {
        "SCENE_ID": scene_id,
        "CROPPED_RASTER": cropped,
        "GENERATED_EXPORT_DIR": export_dir,
    }


def _source_scene(scene_sh: str) -> dict[str, str]:
    scene_sh_wsl = _windows_to_wsl_path(scene_sh)
    cmd = (
        "set -euo pipefail; "
        f"source '{scene_sh_wsl}'; "
        "if declare -F resolve_scene_runtime >/dev/null; then resolve_scene_runtime >/dev/null 2>&1; fi; "
        "echo SCENE_ID=$SCENE_ID; "
        "echo CROPPED_RASTER=$CROPPED_RASTER; "
        "echo GENERATED_EXPORT_DIR=$GENERATED_EXPORT_DIR;"
    )

    try:
        out = subprocess.check_output(
            ["bash", "-lc", cmd],
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        fallback_vals = _parse_scene_file_fallback(scene_sh)
        fallback_vals["GENERATED_EXPORT_DIR"] = _wsl_to_windows_path(fallback_vals.get("GENERATED_EXPORT_DIR", ""))
        return fallback_vals
    except subprocess.CalledProcessError as exc:
        output = (exc.output or "").strip()
        raise RuntimeError(f"加载 scene.sh 失败：{scene_sh}。{output}") from exc

    vals: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()

    if not vals.get("CROPPED_RASTER") or not vals.get("GENERATED_EXPORT_DIR"):
        vals = _parse_scene_file_fallback(scene_sh)

    vals["GENERATED_EXPORT_DIR"] = _wsl_to_windows_path(vals.get("GENERATED_EXPORT_DIR", ""))
    return vals


def resolve_scene_input(scene_sh: str, prefer_v1: bool = True, fallback_quick: bool = True) -> SceneRuntime:
    vals = _source_scene(scene_sh)
    scene_id = vals.get("SCENE_ID", "unknown")
    cropped = vals["CROPPED_RASTER"]
    export_dir = vals["GENERATED_EXPORT_DIR"]

    v1 = os.path.join(export_dir, f"{cropped}_utm_v1.tif")
    quick = os.path.join(export_dir, f"{cropped}_utm_quick.tif")

    candidates: list[str] = []
    if prefer_v1:
        candidates.append(v1)
    if fallback_quick:
        candidates.append(quick)

    for p in candidates:
        if os.path.exists(p):
            return SceneRuntime(
                scene_id=scene_id,
                cropped_raster=cropped,
                generated_export_dir=export_dir,
                input_tif=p,
            )

    raise FileNotFoundError(f"No orthorectified tif found. Checked: {candidates}")
