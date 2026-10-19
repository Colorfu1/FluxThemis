#!/usr/bin/env bash
# Use the caller's prepared environment; never activate or install one.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${FLUXTHEMIS_PYTHON:-python}" "$SCRIPT_DIR/run_evaluation.py" "$@"
