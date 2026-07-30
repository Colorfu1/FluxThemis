#!/usr/bin/env bash
# Remove FluxThemis ROS overlay paths when leaving a Conda environment.

_fluxthemis_remove_path_prefix() {
  local variable_name="$1"
  local removed_prefix="$2"
  local current_value=''
  local rebuilt=''

  eval "current_value=\${${variable_name}:-}"
  rebuilt="$("${CONDA_PREFIX}/bin/python" -c '
import os
import sys

value, prefix = sys.argv[1:]
prefix = prefix.rstrip(os.sep)
kept = [
    entry for entry in value.split(os.pathsep)
    if entry and entry != prefix and not entry.startswith(prefix + os.sep)
]
print(os.pathsep.join(kept), end="")
' "${current_value}" "${removed_prefix}")"

  if [[ -n "${rebuilt}" ]]; then
    export "${variable_name}=${rebuilt}"
  else
    unset "${variable_name}"
  fi
}

if [[ "${ROS_VERSION:-}" == "1" ]]; then
  _fluxthemis_ros_overlay="${FLUXTHEMIS_ROS1_WORKSPACE:-/root/fluxthemis_ros1_ws}/install_isolated"
  for _fluxthemis_ros_variable in PATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH \
      LD_LIBRARY_PATH PYTHONPATH PKG_CONFIG_PATH; do
    _fluxthemis_remove_path_prefix "${_fluxthemis_ros_variable}" \
      "${_fluxthemis_ros_overlay}"
  done
  _fluxthemis_remove_path_prefix LD_LIBRARY_PATH "${CONDA_PREFIX}/lib"
  unset FLUXTHEMIS_ROS1_WORKSPACE
elif [[ "${ROS_VERSION:-}" == "2" ]]; then
  _fluxthemis_ros_workspace="${FLUXTHEMIS_ROS2_WORKSPACE:-/root/fluxthemis_ros2_ws}"
  _fluxthemis_ros_overlay="${_fluxthemis_ros_workspace}/install"
  _fluxthemis_ros_system="/opt/ros/${FLUXTHEMIS_ROS2_DISTRO:-jazzy}"
  for _fluxthemis_ros_prefix in \
      "${_fluxthemis_ros_overlay}" "${_fluxthemis_ros_system}"; do
    for _fluxthemis_ros_variable in PATH CMAKE_PREFIX_PATH \
        AMENT_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH PYTHONPATH \
        PKG_CONFIG_PATH; do
      _fluxthemis_remove_path_prefix "${_fluxthemis_ros_variable}" \
        "${_fluxthemis_ros_prefix}"
    done
  done
  unset FLUXTHEMIS_ROS2_DISTRO FLUXTHEMIS_ROS2_WORKSPACE
fi

_fluxthemis_restore_ros_variable() {
  local variable_name="$1"
  local old_value=''
  local was_set=''
  eval "old_value=\${_FLUXTHEMIS_ROS_OLD_${variable_name}-}"
  eval "was_set=\${_FLUXTHEMIS_ROS_OLD_${variable_name}_SET-}"
  if [[ "${was_set}" == "x" ]]; then
    export "${variable_name}=${old_value}"
  else
    unset "${variable_name}"
  fi
  unset "_FLUXTHEMIS_ROS_OLD_${variable_name}"
  unset "_FLUXTHEMIS_ROS_OLD_${variable_name}_SET"
}

if [[ "${_FLUXTHEMIS_ROS_BACKUP_ACTIVE:-0}" == "1" ]]; then
  for _fluxthemis_ros_variable in ROS_DISTRO ROS_ETC_DIR ROS_PYTHON_VERSION \
      ROS_ROOT ROS_VERSION ROS_MASTER_URI ROS_HOSTNAME ROS_IP ROS_DOMAIN_ID \
      ROS_LOCALHOST_ONLY ROS_AUTOMATIC_DISCOVERY_RANGE RMW_IMPLEMENTATION; do
    _fluxthemis_restore_ros_variable "${_fluxthemis_ros_variable}"
  done
else
  unset ROS_DISTRO ROS_ETC_DIR ROS_PYTHON_VERSION ROS_ROOT ROS_VERSION \
    ROS_MASTER_URI ROS_HOSTNAME ROS_IP ROS_AUTOMATIC_DISCOVERY_RANGE
fi
unset _FLUXTHEMIS_ROS_BACKUP_ACTIVE
unset _fluxthemis_ros_overlay _fluxthemis_ros_workspace \
  _fluxthemis_ros_system _fluxthemis_ros_prefix \
  _fluxthemis_ros_variable
unset -f _fluxthemis_restore_ros_variable 2>/dev/null || true
unset -f _fluxthemis_remove_path_prefix 2>/dev/null || true
