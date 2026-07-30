#!/usr/bin/env bash
# Validate one or both installed FluxThemis runtimes without modifying them.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${FLUXTHEMIS_CONDA_ENV:-fluxthemis}"
ROS2_ENV_NAME="${FLUXTHEMIS_ROS2_CONDA_ENV:-fluxthemis-ros2}"
ROS_MODE="auto"
SMOKE_LEVEL="imports"
CONFIG_PATH=""
SKIP_PIP_CHECK=0

usage() {
  cat <<'EOF'
Usage: bash scripts/check_env.sh [options]

Checks FluxThemis, LIBERO and the generated ROS service types in clean Conda
shells. This command is read-only: it does not install packages, start a ROS
graph, or connect to a robot.

Options:
  --env-name NAME          Primary environment (default: fluxthemis).
  --ros auto|1|2|both|none Runtime(s) to check (default: auto).
  --ros2-env-name NAME     ROS 2 environment used by `--ros both`
                           (default: fluxthemis-ros2).
  --smoke-level LEVEL      LIBERO check: libero, imports, or none
                           (default: imports; primary environment only).
  --config PATH            Also build this FluxVLA config with
                           `fluxthemis-eval --dry-run`.
  --skip-pip-check         Skip `python -m pip check`.
  -h, --help               Show this help.

Examples:
  # Check only the profile selected by fluxthemis's Python version.
  bash scripts/check_env.sh --ros auto

  # Check an optional Python 3.12 / ROS 2 environment independently.
  bash scripts/check_env.sh --env-name fluxthemis-ros2 --ros auto \
    --smoke-level none

  bash scripts/check_env.sh --ros 1 --smoke-level libero \
    --config /root/projects/FluxVLA/configs/pi05/example.py
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || die '--env-name requires a value'
      ENV_NAME="$2"
      shift 2
      ;;
    --ros)
      [[ $# -ge 2 ]] || die '--ros requires a value'
      ROS_MODE="$2"
      shift 2
      ;;
    --ros2-env-name)
      [[ $# -ge 2 ]] || die '--ros2-env-name requires a value'
      ROS2_ENV_NAME="$2"
      shift 2
      ;;
    --smoke-level)
      [[ $# -ge 2 ]] || die '--smoke-level requires a value'
      SMOKE_LEVEL="$2"
      shift 2
      ;;
    --config)
      [[ $# -ge 2 ]] || die '--config requires a value'
      CONFIG_PATH="$2"
      shift 2
      ;;
    --skip-pip-check)
      SKIP_PIP_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown option: $1" ;;
  esac
done

case "${ROS_MODE}" in
  auto|1|2|both|none) ;;
  *) die '--ros must be auto, 1, 2, both, or none' ;;
esac
case "${SMOKE_LEVEL}" in
  libero|imports|none) ;;
  *) die '--smoke-level must be libero, imports, or none' ;;
esac

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
[[ -n "${CONDA_BIN}" && -x "${CONDA_BIN}" ]] || \
  die 'conda is not available on PATH'
CONDA_BASE="$("${CONDA_BIN}" info --base)"
[[ -r "${CONDA_BASE}/etc/profile.d/conda.sh" ]] || \
  die "Conda activation script is unavailable under ${CONDA_BASE}"

CLEAN_ENV=(
  env
  -u AMENT_PREFIX_PATH
  -u CATKIN_PREFIX_PATH
  -u CMAKE_PREFIX_PATH
  -u COLCON_PREFIX_PATH
  -u CONDA_DEFAULT_ENV
  -u CONDA_PREFIX
  -u CONDA_PROMPT_MODIFIER
  -u CONDA_SHLVL
  -u FLUXTHEMIS_ROS1_WORKSPACE
  -u FLUXTHEMIS_ROS2_DISTRO
  -u FLUXTHEMIS_ROS2_WORKSPACE
  -u LD_LIBRARY_PATH
  -u LIBERO_CONFIG_PATH
  -u MUJOCO_GL
  -u PKG_CONFIG_PATH
  -u PYOPENGL_PLATFORM
  -u PYTHONHOME
  -u PYTHONPATH
  -u ROS_DISTRO
  -u ROS_ETC_DIR
  -u ROS_HOSTNAME
  -u ROS_IP
  -u ROS_MASTER_URI
  -u ROS_PACKAGE_PATH
  -u ROS_PYTHON_VERSION
  -u ROS_ROOT
  -u ROS_VERSION
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
)

env_python_version() {
  local name="$1"
  "${CLEAN_ENV[@]}" "${CONDA_BIN}" run -n "${name}" python -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' \
    2>/dev/null || true
}

PRIMARY_PYTHON="$(env_python_version "${ENV_NAME}")"
[[ -n "${PRIMARY_PYTHON}" ]] || \
  die "Conda environment ${ENV_NAME@Q} does not exist"

if [[ "${ROS_MODE}" == "auto" ]]; then
  case "${PRIMARY_PYTHON}" in
    3.10) ROS_MODE=1 ;;
    3.12) ROS_MODE=2 ;;
    *)
      die "auto ROS selection supports Python 3.10 or 3.12; found ${PRIMARY_PYTHON}"
      ;;
  esac
  echo "Auto-selected ROS ${ROS_MODE} for Python ${PRIMARY_PYTHON}."
fi

case "${ROS_MODE}" in
  1)
    [[ "${PRIMARY_PYTHON}" == "3.10" ]] || \
      die "ROS 1 check requires ${ENV_NAME@Q} to use Python 3.10"
    ;;
  2)
    [[ "${PRIMARY_PYTHON}" == "3.12" ]] || \
      die "ROS 2 Jazzy check requires ${ENV_NAME@Q} to use Python 3.12"
    ;;
  both)
    [[ "${PRIMARY_PYTHON}" == "3.10" ]] || \
      die "dual check requires ${ENV_NAME@Q} to use Python 3.10"
    ROS2_PYTHON="$(env_python_version "${ROS2_ENV_NAME}")"
    [[ "${ROS2_PYTHON}" == "3.12" ]] || \
      die "dual check requires ${ROS2_ENV_NAME@Q} to use Python 3.12"
    ;;
  none)
    case "${PRIMARY_PYTHON}" in
      3.10|3.12) ;;
      *) die "FluxThemis supports Python 3.10 or 3.12; found ${PRIMARY_PYTHON}" ;;
    esac
    ;;
esac

if [[ -n "${CONFIG_PATH}" ]]; then
  [[ -f "${CONFIG_PATH}" ]] || die "FluxVLA config does not exist: ${CONFIG_PATH}"
  CONFIG_PATH="$(realpath "${CONFIG_PATH}")"
fi

run_environment_check() {
  local target_env="$1"
  local ros_version="$2"
  local libero_level="$3"
  local config_path="$4"
  local checker_libero="${libero_level}"
  local pip_option=""

  [[ "${checker_libero}" != "libero" ]] || checker_libero="imports"
  [[ "${SKIP_PIP_CHECK}" == "0" ]] || pip_option="--skip-pip-check"

  echo
  echo "Checking ${target_env} (ROS ${ros_version})..."
  "${CLEAN_ENV[@]}" /usr/bin/bash --noprofile --norc -c '
set -euo pipefail
source "$1/etc/profile.d/conda.sh"
conda activate "$2"
checker_args=(--ros "$4" --libero "$5")
if [[ -n "$6" ]]; then
  checker_args+=("$6")
fi
python "$3/check_env.py" "${checker_args[@]}"
if [[ "$7" == "libero" ]]; then
  python "$3/libero_smoke.py" --suite libero_10 --task-id 0 \
    --resolution 64 --steps 1
fi
if [[ -n "$8" ]]; then
  fluxthemis-eval --config "$8" --dry-run
fi
' -- "${CONDA_BASE}" "${target_env}" "${SCRIPT_DIR}" \
    "${ros_version}" "${checker_libero}" "${pip_option}" \
    "${libero_level}" "${config_path}"
}

case "${ROS_MODE}" in
  1) run_environment_check "${ENV_NAME}" 1 "${SMOKE_LEVEL}" \
       "${CONFIG_PATH}" ;;
  2) run_environment_check "${ENV_NAME}" 2 "${SMOKE_LEVEL}" \
       "${CONFIG_PATH}" ;;
  both)
    run_environment_check "${ENV_NAME}" 1 "${SMOKE_LEVEL}" \
      "${CONFIG_PATH}"
    run_environment_check "${ROS2_ENV_NAME}" 2 none ''
    ;;
  none) run_environment_check "${ENV_NAME}" none "${SMOKE_LEVEL}" \
          "${CONFIG_PATH}" ;;
esac

echo
echo 'All requested FluxThemis environment checks passed.'
