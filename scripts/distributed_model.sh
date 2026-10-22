#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_EXECUTABLE="${FLUXVLA_PYTHON:-python}"
ENTRY_POINT="$SCRIPT_DIR/distributed_model.py"
DEFAULT_PORT=29500
source "$SCRIPT_DIR/distributed_launch.sh"
