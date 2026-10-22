#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_EXECUTABLE="${FLUXTHEMIS_PYTHON:-python}"
ENTRY_POINT="$SCRIPT_DIR/distributed_eval.py"
DEFAULT_PORT=29501
source "$SCRIPT_DIR/distributed_launch.sh"
