#!/usr/bin/env bash
# Install ROS 2 Jazzy and build the FluxThemis service interface.

set -euo pipefail

DEFAULT_APT_SOURCE_VERSION="1.2.0"
DEFAULT_APT_SOURCE_SHA256="0804d9b13db770eb87019be414cd78378835228ad5fa801fc88758596dd8f7e5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MESSAGE_PACKAGE="${PROJECT_ROOT}/ros2/fluxthemis_msgs"

ROS_DISTRO="${FLUXTHEMIS_ROS_DISTRO:-jazzy}"
WORKSPACE="${FLUXTHEMIS_ROS2_WORKSPACE:-/root/fluxthemis_ros2_ws}"
APT_SOURCE_VERSION="${FLUXTHEMIS_ROS_APT_SOURCE_VERSION:-${DEFAULT_APT_SOURCE_VERSION}}"
APT_SOURCE_SHA256="${FLUXTHEMIS_ROS_APT_SOURCE_SHA256:-}"
RUN_ROSDEP=1
DRY_RUN=0
APT_SOURCE_TMP=""

usage() {
  cat <<'EOF'
Usage: bash scripts/install_ros2_jazzy.sh [options]

Installs the official ROS 2 apt source and the ROS 2 runtime needed by
FluxThemis, then symlinks and builds ros2/fluxthemis_msgs. The supported host
is Ubuntu 24.04 (noble) with system Python 3.12. This script never creates,
deletes, or modifies a Conda environment.

Options:
  --workspace PATH             Colcon workspace (default:
                               /root/fluxthemis_ros2_ws).
  --ros-distro NAME            ROS distribution (default: jazzy).
  --apt-source-version VERSION Official ros2-apt-source release version
                               (default: 1.2.0).
  --apt-source-sha256 SHA256   Expected noble .deb checksum. Required for a
                               non-default apt-source version.
  --skip-rosdep                Skip rosdep update/install before colcon.
  --dry-run                    Print mutating commands without running them.
  -h, --help                   Show this help.

Environment variables:
  FLUXTHEMIS_ROS_DISTRO
  FLUXTHEMIS_ROS2_WORKSPACE
  FLUXTHEMIS_ROS_APT_SOURCE_VERSION
  FLUXTHEMIS_ROS_APT_SOURCE_SHA256

Examples:
  bash scripts/install_ros2_jazzy.sh
  bash scripts/install_ros2_jazzy.sh --workspace /data/ros2_ws
  bash scripts/install_ros2_jazzy.sh --dry-run

After installation:
  # Bash
  source /opt/ros/jazzy/setup.bash
  source /root/fluxthemis_ros2_ws/install/setup.bash
  # Zsh
  source /opt/ros/jazzy/setup.zsh
  source /root/fluxthemis_ros2_ws/install/setup.zsh
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

print_command() {
  printf '+' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

run() {
  print_command "$@"
  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

run_privileged() {
  if [[ "${EUID}" -eq 0 ]]; then
    run "$@"
  elif command -v sudo >/dev/null 2>&1; then
    run sudo "$@"
  else
    die 'root privileges are required for apt, and sudo is unavailable'
  fi
}

cleanup() {
  if [[ -n "${APT_SOURCE_TMP}" && -d "${APT_SOURCE_TMP}" ]]; then
    rm -f -- "${APT_SOURCE_TMP}/ros2-apt-source.deb"
    rmdir -- "${APT_SOURCE_TMP}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      [[ $# -ge 2 ]] || die '--workspace requires a value'
      WORKSPACE="$2"
      shift 2
      ;;
    --ros-distro)
      [[ $# -ge 2 ]] || die '--ros-distro requires a value'
      ROS_DISTRO="$2"
      shift 2
      ;;
    --apt-source-version)
      [[ $# -ge 2 ]] || die '--apt-source-version requires a value'
      APT_SOURCE_VERSION="$2"
      shift 2
      ;;
    --apt-source-sha256)
      [[ $# -ge 2 ]] || die '--apt-source-sha256 requires a value'
      APT_SOURCE_SHA256="$2"
      shift 2
      ;;
    --skip-rosdep)
      RUN_ROSDEP=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ "${ROS_DISTRO}" =~ ^[a-z][a-z0-9_-]*$ ]] || \
  die "invalid ROS distribution name: ${ROS_DISTRO@Q}"
[[ "${APT_SOURCE_VERSION}" =~ ^[0-9][0-9A-Za-z.+~-]*$ ]] || \
  die "invalid apt-source version: ${APT_SOURCE_VERSION@Q}"

if [[ -z "${APT_SOURCE_SHA256}" ]]; then
  if [[ "${APT_SOURCE_VERSION}" == "${DEFAULT_APT_SOURCE_VERSION}" ]]; then
    APT_SOURCE_SHA256="${DEFAULT_APT_SOURCE_SHA256}"
  else
    die '--apt-source-sha256 is required with a non-default version'
  fi
fi
[[ "${APT_SOURCE_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]] || \
  die '--apt-source-sha256 must contain exactly 64 hexadecimal characters'
APT_SOURCE_SHA256="${APT_SOURCE_SHA256,,}"

[[ -r /etc/os-release ]] || die '/etc/os-release is unavailable'
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" && \
   "${UBUNTU_CODENAME:-}" == "noble" ]] || \
  die 'this installer supports Ubuntu 24.04 (noble) only'
[[ -x /usr/bin/python3 ]] || die 'system Python is unavailable at /usr/bin/python3'
SYSTEM_PYTHON_VERSION="$(/usr/bin/python3 -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${SYSTEM_PYTHON_VERSION}" == "3.12" ]] || \
  die "system Python 3.12 is required; found ${SYSTEM_PYTHON_VERSION}"

[[ -f "${MESSAGE_PACKAGE}/package.xml" && \
   -f "${MESSAGE_PACKAGE}/CMakeLists.txt" && \
   -f "${MESSAGE_PACKAGE}/srv/PredictAction.srv" && \
   -f "${MESSAGE_PACKAGE}/srv/ReportEvaluation.srv" ]] || \
  die "invalid FluxThemis message package: ${MESSAGE_PACKAGE}"

WORKSPACE="$(realpath -m "${WORKSPACE}")"
WORKSPACE_LINK="${WORKSPACE}/src/fluxthemis_msgs"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
ROS_PACKAGES=(
  "ros-${ROS_DISTRO}-ros-base"
  ros-dev-tools
  "ros-${ROS_DISTRO}-cv-bridge"
)

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "note: ignoring active Conda environment ${CONDA_PREFIX@Q}; system Python 3.12 will be used."
fi

install_apt_source() {
  local installed_version=""
  local deb_url
  local deb_path
  local actual_sha256

  if dpkg-query -W -f='${Status}' ros2-apt-source 2>/dev/null | \
    grep -q '^install ok installed$'; then
    installed_version="$(dpkg-query -W -f='${Version}' ros2-apt-source)"
  fi
  if [[ -n "${installed_version}" ]] && \
    { [[ "${installed_version}" == "${APT_SOURCE_VERSION}"* ]] || \
      dpkg --compare-versions "${installed_version}" ge \
        "${APT_SOURCE_VERSION}"; }; then
    echo "Keeping ros2-apt-source ${installed_version}."
    return
  fi

  deb_url="https://github.com/ros-infrastructure/ros-apt-source/releases/download/${APT_SOURCE_VERSION}/ros2-apt-source_${APT_SOURCE_VERSION}.noble_all.deb"
  if [[ "${DRY_RUN}" == "1" ]]; then
    deb_path="/tmp/ros2-apt-source_${APT_SOURCE_VERSION}.noble_all.deb"
    print_command curl --fail --location --retry 3 \
      --output "${deb_path}" "${deb_url}"
    echo "+ verify sha256 ${APT_SOURCE_SHA256} ${deb_path}" >&2
    run_privileged env DEBIAN_FRONTEND=noninteractive \
      /usr/bin/apt-get install -y "${deb_path}"
    return
  fi

  APT_SOURCE_TMP="$(mktemp -d /tmp/fluxthemis-ros2.XXXXXX)"
  deb_path="${APT_SOURCE_TMP}/ros2-apt-source.deb"
  run /usr/bin/curl --fail --location --retry 3 \
    --output "${deb_path}" "${deb_url}"
  actual_sha256="$(sha256sum "${deb_path}" | awk '{print $1}')"
  [[ "${actual_sha256}" == "${APT_SOURCE_SHA256}" ]] || \
    die "ros2-apt-source checksum mismatch: expected ${APT_SOURCE_SHA256}, got ${actual_sha256}"
  run_privileged env DEBIAN_FRONTEND=noninteractive \
    /usr/bin/apt-get install -y "${deb_path}"
}

ensure_workspace_link() {
  local current_target
  local resolved_target
  local resolved_source

  resolved_source="$(realpath -m "${MESSAGE_PACKAGE}")"
  if [[ -L "${WORKSPACE_LINK}" ]]; then
    current_target="$(readlink "${WORKSPACE_LINK}")"
    if [[ "${current_target}" == /* ]]; then
      resolved_target="$(realpath -m "${current_target}")"
    else
      resolved_target="$(realpath -m \
        "$(dirname "${WORKSPACE_LINK}")/${current_target}")"
    fi
    [[ "${resolved_target}" == "${resolved_source}" ]] || \
      die "${WORKSPACE_LINK} points to ${resolved_target}; refusing to replace it"
    echo "Keeping interface symlink ${WORKSPACE_LINK}."
    return
  fi
  [[ ! -e "${WORKSPACE_LINK}" ]] || \
    die "${WORKSPACE_LINK} already exists and will not be overwritten"
  run mkdir -p "${WORKSPACE}/src"
  run ln -s "${MESSAGE_PACKAGE}" "${WORKSPACE_LINK}"
}

run_clean_ros_shell() {
  local command_text="$1"
  shift
  print_command env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER -u PYTHONHOME -u PYTHONPATH \
    -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
    -u CATKIN_PREFIX_PATH -u ROS_PACKAGE_PATH -u ROS_ROOT \
    -u ROS_MASTER_URI \
    -u LD_LIBRARY_PATH -u PKG_CONFIG_PATH -u ROS_DISTRO -u ROS_VERSION \
    -u ROS_PYTHON_VERSION \
    PATH=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin \
    /usr/bin/bash --noprofile --norc -c "${command_text}" -- "$@"
  if [[ "${DRY_RUN}" == "0" ]]; then
    env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
      -u CONDA_PROMPT_MODIFIER -u PYTHONHOME -u PYTHONPATH \
      -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
      -u CATKIN_PREFIX_PATH -u ROS_PACKAGE_PATH -u ROS_ROOT \
      -u ROS_MASTER_URI \
      -u LD_LIBRARY_PATH -u PKG_CONFIG_PATH -u ROS_DISTRO -u ROS_VERSION \
      -u ROS_PYTHON_VERSION \
      PATH=/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin \
      /usr/bin/bash --noprofile --norc -c "${command_text}" -- "$@"
  fi
}

run_privileged env DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get update
run_privileged env DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get install -y \
  ca-certificates curl software-properties-common
run_privileged /usr/bin/add-apt-repository -y -n universe
install_apt_source
run_privileged env DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get update
run_privileged env DEBIAN_FRONTEND=noninteractive /usr/bin/apt-get install -y \
  "${ROS_PACKAGES[@]}"

ensure_workspace_link

if [[ "${RUN_ROSDEP}" == "1" ]]; then
  if [[ ! -r /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    run_privileged /usr/bin/rosdep init
  fi
  run_clean_ros_shell '
set -euo pipefail
source "$1"
/usr/bin/rosdep update --rosdistro "$2"
/usr/bin/rosdep install --from-paths "$3" --ignore-src -r -y --rosdistro "$2"
' "${ROS_SETUP}" "${ROS_DISTRO}" "${WORKSPACE_LINK}"
fi

run_clean_ros_shell '
set -euo pipefail
source "$1"
cd "$2"
/usr/bin/colcon build --symlink-install --packages-select fluxthemis_msgs \
  --cmake-force-configure \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 \
               -DPYTHON_EXECUTABLE=/usr/bin/python3
' "${ROS_SETUP}" "${WORKSPACE}"

run_clean_ros_shell '
set -euo pipefail
source "$1"
source "$2/install/setup.bash"
/usr/bin/python3 -c "import rclpy, cv_bridge; from fluxthemis_msgs.srv import PredictAction, ReportEvaluation; print(PredictAction, ReportEvaluation)"
ros2 interface show fluxthemis_msgs/srv/PredictAction >/dev/null
ros2 interface show fluxthemis_msgs/srv/ReportEvaluation >/dev/null
' "${ROS_SETUP}" "${WORKSPACE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo 'Dry run complete; no packages, links, or build outputs were changed.'
  exit 0
fi

cat <<EOF

ROS 2 interface installation complete.
  ROS distro:  ${ROS_DISTRO}
  workspace:   ${WORKSPACE}
  interfaces:  fluxthemis_msgs/srv/PredictAction, fluxthemis_msgs/srv/ReportEvaluation

Activate it with:
  # Bash
  source ${ROS_SETUP}
  source ${WORKSPACE}/install/setup.bash
  # Zsh
  source /opt/ros/${ROS_DISTRO}/setup.zsh
  source ${WORKSPACE}/install/setup.zsh
EOF
