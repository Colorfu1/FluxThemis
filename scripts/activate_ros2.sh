#!/usr/bin/env bash
# Activate the Jazzy overlay built by install_ros2_jazzy.sh.

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo 'error: activate a Conda environment before sourcing activate_ros2.sh' >&2
  return 1 2>/dev/null || exit 1
fi

_fluxthemis_ros2_python="$("${CONDA_PREFIX}/bin/python" -c \
  'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${_fluxthemis_ros2_python}" != "3.12" ]]; then
  echo "error: ROS 2 Jazzy requires Python 3.12; found ${_fluxthemis_ros2_python}" >&2
  unset _fluxthemis_ros2_python
  return 1 2>/dev/null || exit 1
fi

if [[ -n "${ROS_VERSION:-}" && "${ROS_VERSION}" != "2" ]]; then
  echo "error: ROS ${ROS_VERSION} is already sourced in this shell" >&2
  echo '       start a clean shell before activating the ROS 2 environment' >&2
  unset _fluxthemis_ros2_python
  return 1 2>/dev/null || exit 1
fi

_fluxthemis_ros2_distro="${FLUXTHEMIS_ROS2_DISTRO:-jazzy}"
_fluxthemis_ros2_workspace="${FLUXTHEMIS_ROS2_WORKSPACE:-}"
_fluxthemis_ros2_config="${CONDA_PREFIX}/etc/fluxthemis/ros2-workspace"
if [[ -z "${_fluxthemis_ros2_workspace}" && \
      -r "${_fluxthemis_ros2_config}" ]]; then
  IFS= read -r _fluxthemis_ros2_workspace < "${_fluxthemis_ros2_config}"
fi
_fluxthemis_ros2_workspace="${_fluxthemis_ros2_workspace:-/root/fluxthemis_ros2_ws}"
_fluxthemis_ros2_system_setup="/opt/ros/${_fluxthemis_ros2_distro}/setup.sh"
_fluxthemis_ros2_overlay_setup="${_fluxthemis_ros2_workspace}/install/setup.sh"

if [[ ! -r "${_fluxthemis_ros2_system_setup}" || \
      ! -r "${_fluxthemis_ros2_overlay_setup}" ]]; then
  echo "error: ROS 2 ${_fluxthemis_ros2_distro} overlay is unavailable" >&2
  echo '       run scripts/install_ros2_jazzy.sh first' >&2
  unset _fluxthemis_ros2_python _fluxthemis_ros2_distro \
    _fluxthemis_ros2_workspace _fluxthemis_ros2_config \
    _fluxthemis_ros2_system_setup _fluxthemis_ros2_overlay_setup
  return 1 2>/dev/null || exit 1
fi

# Preserve scalar ROS settings that existed before this Conda environment.
# The paired deactivation hook restores them instead of leaking Jazzy values
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
  for _fluxthemis_ros2_variable in ROS_DISTRO ROS_ETC_DIR ROS_PYTHON_VERSION \
      ROS_ROOT ROS_VERSION ROS_MASTER_URI ROS_HOSTNAME ROS_IP ROS_DOMAIN_ID \
      ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE RMW_IMPLEMENTATION; do
    _fluxthemis_backup_ros_variable "${_fluxthemis_ros2_variable}"
  done
  export _FLUXTHEMIS_ROS_BACKUP_ACTIVE=1
fi
unset _fluxthemis_ros2_variable
unset -f _fluxthemis_backup_ros_variable 2>/dev/null || true

_fluxthemis_ros2_had_nounset=0
case "$-" in
  *u*)
    _fluxthemis_ros2_had_nounset=1
    set +u
    ;;
esac
# shellcheck disable=SC1090
if source "${_fluxthemis_ros2_system_setup}" && \
    source "${_fluxthemis_ros2_overlay_setup}"; then
  _fluxthemis_ros2_source_status=0
else
  _fluxthemis_ros2_source_status=$?
fi
if [[ "${_fluxthemis_ros2_had_nounset}" == "1" ]]; then
  set -u
fi
if [[ "${_fluxthemis_ros2_source_status}" != "0" ]]; then
  echo 'error: failed to source the ROS 2 Jazzy environment' >&2
  unset _fluxthemis_ros2_python _fluxthemis_ros2_distro \
    _fluxthemis_ros2_workspace _fluxthemis_ros2_config \
    _fluxthemis_ros2_system_setup _fluxthemis_ros2_overlay_setup \
    _fluxthemis_ros2_had_nounset _fluxthemis_ros2_source_status
  return 1 2>/dev/null || exit 1
fi

export FLUXTHEMIS_ROS2_DISTRO="${_fluxthemis_ros2_distro}"
export FLUXTHEMIS_ROS2_WORKSPACE="${_fluxthemis_ros2_workspace}"
unset _fluxthemis_ros2_python _fluxthemis_ros2_distro \
  _fluxthemis_ros2_workspace _fluxthemis_ros2_config \
  _fluxthemis_ros2_system_setup _fluxthemis_ros2_overlay_setup \
  _fluxthemis_ros2_had_nounset _fluxthemis_ros2_source_status
