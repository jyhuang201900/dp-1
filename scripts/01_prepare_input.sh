#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${PIPELINE_CONFIG:-$ROOT_DIR/configs/pipeline.yaml}"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
python -m forestseg.cli prepare-input --config "$CONFIG_PATH" "$@"
