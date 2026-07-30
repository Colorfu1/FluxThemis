#!/usr/bin/env bash
# Activate the Python 3.10 ROS 1 overlay built by install_ros1_noetic_py310.sh.
# This file is meant to be sourced, either directly or by a Conda hook.

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "error: activate a Conda environment before sourcing activate_ros1.sh" >&2
  return 1 2>/dev/null || exit 1
fi

_fluxthemis_ros1_python="$("${CONDA_PREFIX}/bin/python" -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${_fluxthemis_ros1_python}" != "3.10" ]]; then
  echo "error: ROS 1 Noetic overlay requires Python 3.10; found ${_fluxthemis_ros1_python}" >&2
  unset _fluxthemis_ros1_python
  return 1 2>/dev/null || exit 1
fi

_fluxthemis_ros1_workspace="${FLUXTHEMIS_ROS1_WORKSPACE:-}"
_fluxthemis_ros1_config="${CONDA_PREFIX}/etc/fluxthemis/ros1-workspace"
if [[ -z "${_fluxthemis_ros1_workspace}" && -r "${_fluxthemis_ros1_config}" ]]; then
  IFS= read -r _fluxthemis_ros1_workspace < "${_fluxthemis_ros1_config}"
fi
_fluxthemis_ros1_workspace="${_fluxthemis_ros1_workspace:-/root/fluxthemis_ros1_ws}"
_fluxthemis_ros1_setup="${_fluxthemis_ros1_workspace}/install_isolated/setup.sh"

if [[ ! -r "${_fluxthemis_ros1_setup}" ]]; then
  echo "error: ROS 1 overlay is not installed at ${_fluxthemis_ros1_workspace}" >&2
  echo "       run scripts/install_ros1_noetic_py310.sh first" >&2
  unset _fluxthemis_ros1_python _fluxthemis_ros1_workspace \
    _fluxthemis_ros1_config \
    _fluxthemis_ros1_setup
  return 1 2>/dev/null || exit 1
fi

# Never put ROS 1/cp310 and ROS 2/cp312 Python packages in one interpreter.
# Use a clean process when switching middleware environments.
if [[ -n "${ROS_VERSION:-}" && "${ROS_VERSION}" != "1" ]]; then
  echo "error: ROS ${ROS_VERSION} is already sourced in this shell" >&2
  echo "       start a clean shell before activating the ROS 1 environment" >&2
  unset _fluxthemis_ros1_python _fluxthemis_ros1_workspace \
    _fluxthemis_ros1_config \
    _fluxthemis_ros1_setup
  return 1 2>/dev/null || exit 1
fi

# Preserve scalar ROS settings that existed before this Conda environment.
# The paired deactivation hook restores them instead of leaking Noetic values
# into the next shell/environment.
_fluxthemis_backup_ros_variable() {
  local variable_name="$1"
  local old_value=''
  local was_set=''
  eval "old_value=\${${variable_name}-}"
  eval "was_set=\${${variable_name}+x}"
  export "_FLUXTHEMIS_ROS_OLD_${variable_name}=${old_value}"
  export "_FLUXTHEMIS_ROS_OLD_${variable_name}_SET=${was_set}"
}
if [[ "${_FLUXTHEMIS_ROS_BACKUP_ACTIVE:-0}" != "1" ]]; then
  for _fluxthemis_ros1_variable in ROS_DISTRO ROS_ETC_DIR ROS_PYTHON_VERSION \
      ROS_ROOT ROS_VERSION ROS_MASTER_URI ROS_HOSTNAME ROS_IP ROS_DOMAIN_ID \
      ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE RMW_IMPLEMENTATION; do
    _fluxthemis_backup_ros_variable "${_fluxthemis_ros1_variable}"
  done
  export _FLUXTHEMIS_ROS_BACKUP_ACTIVE=1
fi
unset _fluxthemis_ros1_variable
unset -f _fluxthemis_backup_ros_variable 2>/dev/null || true

# rospack and other source-built executables link against Conda's
# libpython3.10.so, so the Conda library directory must precede system paths.
case ":${LD_LIBRARY_PATH:-}:" in
  *":${CONDA_PREFIX}/lib:"*) ;;
  *) export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

# setup.sh is shell-neutral and works when this hook is sourced from Bash or
# Zsh. shellcheck cannot resolve a runtime-selected workspace path.
# shellcheck disable=SC1090
_fluxthemis_ros1_had_nounset=0
case "$-" in
  *u*)
    _fluxthemis_ros1_had_nounset=1
    set +u
    ;;
esac
if source "${_fluxthemis_ros1_setup}"; then
  _fluxthemis_ros1_source_status=0
else
  _fluxthemis_ros1_source_status=$?
fi
if [[ "${_fluxthemis_ros1_had_nounset}" == "1" ]]; then
  set -u
fi
if [[ "${_fluxthemis_ros1_source_status}" != "0" ]]; then
  echo "error: failed to source ${_fluxthemis_ros1_setup}" >&2
  unset _fluxthemis_ros1_python _fluxthemis_ros1_workspace \
    _fluxthemis_ros1_config \
    _fluxthemis_ros1_setup _fluxthemis_ros1_had_nounset \
    _fluxthemis_ros1_source_status
  return 1 2>/dev/null || exit 1
fi

export FLUXTHEMIS_ROS1_WORKSPACE="${_fluxthemis_ros1_workspace}"
unset _fluxthemis_ros1_python _fluxthemis_ros1_workspace \
  _fluxthemis_ros1_config \
  _fluxthemis_ros1_setup _fluxthemis_ros1_had_nounset \
  _fluxthemis_ros1_source_status
