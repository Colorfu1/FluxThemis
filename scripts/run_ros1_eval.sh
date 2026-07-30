#!/usr/bin/env bash
# One-command ROS 1 evaluation launcher for FluxThemis. The FluxVLA model
# server remains a separate GPU process; this script owns the environment side.

set -Eeuo pipefail

ENV_NAME="${FLUXTHEMIS_CONDA_ENV:-fluxthemis}"
FLUXVLA_ROOT="${FLUXVLA_ROOT:-/root/projects/FluxVLA}"
CONFIG="${FLUXTHEMIS_CONFIG:-${FLUXVLA_ROOT}/configs/pi05/pi05_paligemma_libero_10_full_finetune.py}"
MASTER_TIMEOUT="${FLUXTHEMIS_MASTER_TIMEOUT:-30}"
SERVICE_TIMEOUT="${FLUXTHEMIS_SERVICE_TIMEOUT:-900}"
PARALLEL_WORKERS="${FLUXTHEMIS_PARALLEL_WORKERS:-}"
SIMULATOR_GPUS="${FLUXTHEMIS_SIM_GPU_IDS:-}"
SMOKE=0
KEEP_MASTER=1
EVAL_ARGS=()

SERVICE_NAME=''
MASTER_URI=''
STATE_ROOT=''
MASTER_STATE_DIR=''
ROSCORE_LOG=''
ROSCORE_PID_FILE=''
ROSCORE_START_FILE=''
ROSCORE_URI_FILE=''
ROSCORE_PID=''
ROSCORE_START_TIME=''
ACTIVE_CHILD_PID=''
EVAL_LOCK_FD=''

usage() {
  cat <<'EOF'
Usage: bash scripts/run_ros1_eval.sh [launcher options] [eval options]

Activates the FluxThemis ROS 1 environment, starts or reuses roscore, waits
for the FluxVLA action and optional evaluation-reporting services, validates
their types, and runs fluxthemis-eval. Eval options such as --run-name and
runner-only --cfg-options are passed through unchanged.

Launcher options:
  --config PATH             FluxVLA config. Defaults to the PI0.5 LIBERO-10
                            config under FLUXVLA_ROOT.
  --env-name NAME           Conda environment (default: fluxthemis).
  --master-timeout SECONDS  roscore startup timeout (default: 30).
  --service-timeout SECONDS FluxVLA service wait timeout (default: 900).
  --workers COUNT           Spawn COUNT parallel simulator rollout workers.
                            Environment: FLUXTHEMIS_PARALLEL_WORKERS.
  --sim-gpus IDS            Comma-separated physical simulator/render GPU ids.
                            Environment: FLUXTHEMIS_SIM_GPU_IDS.
  --smoke                   Run task 0 for one episode.
  --shutdown-master         Stop a launcher-owned master after a successful
                            eval. By default it stays available so the loaded
                            FluxVLA server can be reused.
  -h, --help                Show this help.

Examples:
  bash scripts/run_ros1_eval.sh --smoke --run-name pi05_smoke
  bash scripts/run_ros1_eval.sh --run-name pi05_full
  bash scripts/run_ros1_eval.sh --workers 2 --sim-gpus 2,3 \
    --run-name pi05_parallel
  bash scripts/run_ros1_eval.sh --config /path/to/FluxVLA/configs/model.py \
    --cfg-options themis.runner.episodes_per_task=3

Run this launcher first, then start the FluxVLA model server in another
terminal. The launcher waits while its checkpoint is loading.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_value() {
  [[ $# -ge 2 ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      require_value "$@"
      CONFIG="$2"
      shift 2
      ;;
    --env-name)
      require_value "$@"
      ENV_NAME="$2"
      shift 2
      ;;
    --master-timeout)
      require_value "$@"
      MASTER_TIMEOUT="$2"
      shift 2
      ;;
    --service-timeout)
      require_value "$@"
      SERVICE_TIMEOUT="$2"
      shift 2
      ;;
    --workers)
      require_value "$@"
      PARALLEL_WORKERS="$2"
      shift 2
      ;;
    --sim-gpus)
      require_value "$@"
      SIMULATOR_GPUS="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --shutdown-master)
      KEEP_MASTER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EVAL_ARGS+=("$@")
      break
      ;;
    *)
      EVAL_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ "${MASTER_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || \
  die '--master-timeout must be a positive integer'
[[ "${SERVICE_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || \
  die '--service-timeout must be a positive integer'
if [[ -n "${PARALLEL_WORKERS}" ]]; then
  [[ "${PARALLEL_WORKERS}" =~ ^[1-9][0-9]*$ ]] || \
    die '--workers must be a positive integer'
  EVAL_ARGS+=(--parallel-workers "${PARALLEL_WORKERS}")
fi
if [[ -n "${SIMULATOR_GPUS}" ]]; then
  [[ "${SIMULATOR_GPUS}" != *, && "${SIMULATOR_GPUS}" != ,* \
    && "${SIMULATOR_GPUS}" != *,,* ]] || \
    die '--sim-gpus must be a comma-separated list of GPU ids'
  EVAL_ARGS+=(--simulator-gpus "${SIMULATOR_GPUS}")
fi

if [[ "${SMOKE}" == "1" ]]; then
  for argument in "${EVAL_ARGS[@]}"; do
    [[ "${argument}" != '--cfg-options' ]] || \
      die '--smoke cannot be combined with --cfg-options'
  done
  EVAL_ARGS+=(
    --cfg-options
    'themis.runner.task_ids=["0"]'
    'themis.runner.episodes_per_task=1'
  )
fi

CONDA_BIN="$(command -v conda || true)"
if [[ -z "${CONDA_BIN}" && -x /root/miniconda3/bin/conda ]]; then
  CONDA_BIN=/root/miniconda3/bin/conda
fi
[[ -n "${CONDA_BIN}" ]] || die 'conda is not available'
CONDA_BASE="$("${CONDA_BIN}" info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# Re-run the target hook even when this environment was already active.
conda activate "${ENV_NAME}"

[[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == '3.10' ]] || \
  die 'ROS 1 evaluation requires a Python 3.10 Conda environment'
[[ "${ROS_VERSION:-}" == '1' ]] || \
  die "${ENV_NAME@Q} does not have an active ROS 1 hook; reactivate it or rerun scripts/install.sh"
command -v roscore >/dev/null || die 'roscore is unavailable'
command -v rosservice >/dev/null || die 'rosservice is unavailable'
command -v fluxthemis-eval >/dev/null || die 'fluxthemis-eval is unavailable'
command -v flock >/dev/null || die 'flock is unavailable'
command -v setsid >/dev/null || die 'setsid is unavailable'
command -v timeout >/dev/null || die 'timeout is unavailable'

# Resolve the config through the same FluxThemis parser used by the real eval.
# Only runner overrides are accepted: transport and server settings remain a
# single source of truth shared with the already-started FluxVLA process.
CONFIG_INFO="$(python - "${CONFIG}" "${EVAL_ARGS[@]}" <<'PY'
import sys
from pathlib import Path

from fluxthemis.cli.eval import (ensure_run_artifacts_available, load_config,
                                 parse_args)

config_path = str(Path(sys.argv[1]).expanduser().resolve(strict=True))
args = parse_args(['--config', config_path, *sys.argv[2:]])
for key in (args.cfg_options or {}):
    if not key.startswith('themis.runner.'):
        raise SystemExit(
            'run_ros1_eval.sh only accepts themis.runner.* cfg overrides; '
            f'got {key!r}')
cfg, runner_cfg = load_config(args)
try:
    ensure_run_artifacts_available(runner_cfg, dry_run=args.dry_run)
except (FileExistsError, ValueError) as exc:
    raise SystemExit(str(exc)) from exc
client_type = runner_cfg.get('model_client', {}).get('type')
client_name = (client_type.rsplit('.', 1)[-1]
               if isinstance(client_type, str)
               else getattr(client_type, '__name__', ''))
if client_name != 'FluxVLAROSModelClient':
    raise SystemExit(
        'run_ros1_eval.sh requires themis.runner.model_client.type='
        'FluxVLAROSModelClient')
service_name = cfg.themis.transport.service_name
if not isinstance(service_name, str) or not service_name.startswith('/'):
    raise SystemExit('themis.transport.service_name must be an absolute name')
report_service_name = cfg.themis.transport.get('report_service_name', '')
if (not isinstance(report_service_name, str)
        or (report_service_name and not report_service_name.startswith('/'))):
    raise SystemExit(
        'themis.transport.report_service_name must be an absolute name')
print(config_path)
print(service_name)
print(report_service_name)
PY
)"
CONFIG="$(sed -n '1p' <<<"${CONFIG_INFO}")"
SERVICE_NAME="$(sed -n '2p' <<<"${CONFIG_INFO}")"
REPORT_SERVICE_NAME="$(sed -n '3p' <<<"${CONFIG_INFO}")"
[[ -n "${CONFIG}" && -n "${SERVICE_NAME}" ]] || \
  die 'failed to resolve FluxVLA config and ROS service name'

# Canonicalize equivalent master URIs so state and locks cannot drift between
# http://localhost:11311 and http://localhost:11311/.
MASTER_URI="$(python - <<'PY'
import os
from urllib.parse import urlsplit

raw = os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')
parsed = urlsplit(raw)
if parsed.scheme != 'http' or not parsed.hostname:
    raise SystemExit(f'ROS_MASTER_URI must be an http URI, got {raw!r}')
port = parsed.port or 11311
host = parsed.hostname.lower()
if ':' in host:
    host = f'[{host}]'
print(f'http://{host}:{port}')
PY
)"
export ROS_MASTER_URI="${MASTER_URI}"

read -r MASTER_KEY EVAL_KEY < <(python - "${MASTER_URI}" "${SERVICE_NAME}" <<'PY'
import hashlib
import sys

master_uri, service_name = sys.argv[1:]
print(hashlib.sha256(master_uri.encode()).hexdigest()[:16],
      hashlib.sha256(f'{master_uri}\0{service_name}'.encode()).hexdigest()[:16])
PY
)
STATE_ROOT="${FLUXTHEMIS_ROS1_STATE_DIR:-${CONDA_PREFIX}/var/run/fluxthemis}"
install -d -m 0700 "${STATE_ROOT}"
MASTER_STATE_DIR="${STATE_ROOT}/master-${MASTER_KEY}"
install -d -m 0700 "${MASTER_STATE_DIR}"
ROSCORE_LOG="${MASTER_STATE_DIR}/roscore.log"
ROSCORE_PID_FILE="${MASTER_STATE_DIR}/roscore.pid"
ROSCORE_START_FILE="${MASTER_STATE_DIR}/roscore.start"
ROSCORE_URI_FILE="${MASTER_STATE_DIR}/roscore.uri"

EVAL_LOCK_FILE="${STATE_ROOT}/eval-${EVAL_KEY}.lock"
exec {EVAL_LOCK_FD}>"${EVAL_LOCK_FILE}"
flock -n "${EVAL_LOCK_FD}" || \
  die "another FluxThemis evaluation already owns ${SERVICE_NAME}"

process_start_time() {
  local pid="$1"
  [[ -r "/proc/${pid}/stat" ]] || return 1
  awk '{print $22}' "/proc/${pid}/stat"
}

session_pids() {
  local session_id="$1"
  ps -eo pid=,sid=,stat= | \
    awk -v sid="${session_id}" '$2 == sid && $3 !~ /^Z/ {print $1}'
}

session_alive() {
  [[ -n "$(session_pids "$1")" ]]
}

terminate_session() {
  local session_id="$1"
  local first_signal="${2:-INT}"
  local signal_name=''
  local pid_list=''
  local _=''

  session_alive "${session_id}" || return 0
  for signal_name in "${first_signal}" TERM KILL; do
    pid_list="$(session_pids "${session_id}")"
    [[ -n "${pid_list}" ]] || return 0
    # shellcheck disable=SC2086
    kill -s "${signal_name}" ${pid_list} 2>/dev/null || true
    for _ in $(seq 1 50); do
      session_alive "${session_id}" || return 0
      sleep 0.1
    done
  done
  ! session_alive "${session_id}"
}

master_ready() {
  python - <<'PY' >/dev/null 2>&1
import os
import socket
import xmlrpc.client

socket.setdefaulttimeout(1.0)
master = xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI'])
code, _, _ = master.getUri('/fluxthemis_eval_launcher')
raise SystemExit(0 if code == 1 else 1)
PY
}

tracked_master_record() {
  local pid=''
  local recorded_start=''
  local current_start=''
  local recorded_uri=''
  local pgid=''

  [[ -r "${ROSCORE_PID_FILE}" && -r "${ROSCORE_START_FILE}" \
    && -r "${ROSCORE_URI_FILE}" ]] || return 1
  IFS= read -r pid < "${ROSCORE_PID_FILE}"
  IFS= read -r recorded_start < "${ROSCORE_START_FILE}"
  IFS= read -r recorded_uri < "${ROSCORE_URI_FILE}"
  [[ "${pid}" =~ ^[1-9][0-9]*$ && "${recorded_start}" =~ ^[0-9]+$ ]] || \
    return 1
  [[ "${recorded_uri}" == "${MASTER_URI}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  current_start="$(process_start_time "${pid}" || true)"
  [[ "${current_start}" == "${recorded_start}" ]] || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/${pid}/cmdline" | \
    grep -Eq '(^|/)roscore([[:space:]]|$)' || return 1
  pgid="$(ps -o pgid= -p "${pid}" | tr -d '[:space:]')"
  [[ "${pgid}" == "${pid}" ]] || return 1
  printf '%s %s\n' "${pid}" "${recorded_start}"
}

remove_master_state() {
  rm -f -- "${ROSCORE_PID_FILE}" "${ROSCORE_START_FILE}" \
    "${ROSCORE_URI_FILE}"
}

stop_owned_roscore() {
  local current_start=''
  [[ -n "${ROSCORE_PID}" ]] || return 0
  if ! kill -0 "${ROSCORE_PID}" 2>/dev/null; then
    remove_master_state
    ROSCORE_PID=''
    ROSCORE_START_TIME=''
    return 0
  fi
  current_start="$(process_start_time "${ROSCORE_PID}" || true)"
  if [[ "${current_start}" != "${ROSCORE_START_TIME}" ]]; then
    echo '[FluxThemis] Refusing to stop roscore: recorded PID was reused.' >&2
    return 1
  fi
  if ! terminate_session "${ROSCORE_PID}" INT; then
    echo "[FluxThemis] Failed to stop roscore session ${ROSCORE_PID}." >&2
    return 1
  fi
  wait "${ROSCORE_PID}" 2>/dev/null || true
  remove_master_state
  ROSCORE_PID=''
  ROSCORE_START_TIME=''
}

stop_active_child() {
  local first_signal="${1:-INT}"
  [[ -n "${ACTIVE_CHILD_PID}" ]] || return 0
  terminate_session "${ACTIVE_CHILD_PID}" "${first_signal}" || true
  wait "${ACTIVE_CHILD_PID}" 2>/dev/null || true
  ACTIVE_CHILD_PID=''
}

cleanup() {
  local status=$?
  set +e
  trap - EXIT INT TERM HUP
  stop_active_child TERM
  if [[ "${status}" == "0" && "${KEEP_MASTER}" == "1" \
    && -n "${ROSCORE_PID}" ]]; then
    echo "[FluxThemis] Keeping roscore PID ${ROSCORE_PID} for subsequent evaluations."
    echo "[FluxThemis] roscore log: ${ROSCORE_LOG}"
    ROSCORE_PID=''
    ROSCORE_START_TIME=''
  fi
  stop_owned_roscore || status=1
  exit "${status}"
}

handle_signal() {
  local signal_name="$1"
  local status="$2"
  trap - INT TERM HUP
  set +e
  stop_active_child "${signal_name}"
  exit "${status}"
}

trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP
trap cleanup EXIT

TRACKED_RECORD="$(tracked_master_record || true)"
if [[ -n "${TRACKED_RECORD}" ]]; then
  read -r ROSCORE_PID ROSCORE_START_TIME <<<"${TRACKED_RECORD}"
else
  remove_master_state
fi

if master_ready; then
  echo "[FluxThemis] Reusing ROS master at ${MASTER_URI}."
  if [[ -z "${ROSCORE_PID}" ]]; then
    echo '[FluxThemis] The existing master is external and will not be stopped.'
  fi
else
  if [[ -n "${ROSCORE_PID}" ]]; then
    echo '[FluxThemis] Replacing an unresponsive launcher-owned roscore.' >&2
    stop_owned_roscore
  fi

  echo "[FluxThemis] Starting roscore at ${MASTER_URI}."
  (
    exec {EVAL_LOCK_FD}>&-
    exec setsid roscore
  ) >"${ROSCORE_LOG}" 2>&1 &
  ROSCORE_PID=$!
  ROSCORE_START_TIME="$(process_start_time "${ROSCORE_PID}")"
  printf '%s\n' "${ROSCORE_PID}" > "${ROSCORE_PID_FILE}.tmp.$$"
  mv "${ROSCORE_PID_FILE}.tmp.$$" "${ROSCORE_PID_FILE}"
  printf '%s\n' "${ROSCORE_START_TIME}" > "${ROSCORE_START_FILE}.tmp.$$"
  mv "${ROSCORE_START_FILE}.tmp.$$" "${ROSCORE_START_FILE}"
  printf '%s\n' "${MASTER_URI}" > "${ROSCORE_URI_FILE}.tmp.$$"
  mv "${ROSCORE_URI_FILE}.tmp.$$" "${ROSCORE_URI_FILE}"

  master_deadline=$((SECONDS + MASTER_TIMEOUT))
  while ! master_ready; do
    if ! kill -0 "${ROSCORE_PID}" 2>/dev/null; then
      echo '[FluxThemis] roscore exited before becoming ready:' >&2
      sed -n '1,200p' "${ROSCORE_LOG}" >&2
      exit 1
    fi
    if (( SECONDS >= master_deadline )); then
      echo "[FluxThemis] roscore did not become ready in ${MASTER_TIMEOUT}s:" >&2
      sed -n '1,200p' "${ROSCORE_LOG}" >&2
      exit 1
    fi
    sleep 0.2
  done
fi

echo "[FluxThemis] Waiting up to ${SERVICE_TIMEOUT}s for ${SERVICE_NAME}."
service_deadline=$((SECONDS + SERVICE_TIMEOUT))
SERVICE_TYPE=''
while true; do
  set +e
  SERVICE_TYPE="$(timeout 2 rosservice type "${SERVICE_NAME}" 2>/dev/null)"
  service_status=$?
  set -e
  if [[ "${service_status}" == "0" ]]; then
    break
  fi
  if (( SECONDS >= service_deadline )); then
    die "FluxVLA service ${SERVICE_NAME} was not ready in ${SERVICE_TIMEOUT}s"
  fi
  sleep 0.2
done
[[ "${SERVICE_TYPE}" == 'fluxthemis_msgs/PredictAction' ]] || \
  die "${SERVICE_NAME} has type ${SERVICE_TYPE@Q}; expected fluxthemis_msgs/PredictAction"

if [[ -n "${REPORT_SERVICE_NAME}" ]]; then
  echo "[FluxThemis] Waiting up to ${SERVICE_TIMEOUT}s for ${REPORT_SERVICE_NAME}."
  report_deadline=$((SECONDS + SERVICE_TIMEOUT))
  REPORT_SERVICE_TYPE=''
  while true; do
    set +e
    REPORT_SERVICE_TYPE="$(timeout 2 rosservice type "${REPORT_SERVICE_NAME}" 2>/dev/null)"
    report_service_status=$?
    set -e
    if [[ "${report_service_status}" == "0" ]]; then
      break
    fi
    if (( SECONDS >= report_deadline )); then
      die "FluxVLA reporting service ${REPORT_SERVICE_NAME} was not ready in ${SERVICE_TIMEOUT}s"
    fi
    sleep 0.2
  done
  [[ "${REPORT_SERVICE_TYPE}" == 'fluxthemis_msgs/ReportEvaluation' ]] || \
    die "${REPORT_SERVICE_NAME} has type ${REPORT_SERVICE_TYPE@Q}; expected fluxthemis_msgs/ReportEvaluation"
fi

echo "[FluxThemis] FluxVLA services ready; starting evaluation with ${CONFIG}."
(
  exec {EVAL_LOCK_FD}>&-
  exec setsid fluxthemis-eval --config "${CONFIG}" "${EVAL_ARGS[@]}"
) &
ACTIVE_CHILD_PID=$!
set +e
wait "${ACTIVE_CHILD_PID}"
EVAL_STATUS=$?
set -e
ACTIVE_CHILD_PID=''
exit "${EVAL_STATUS}"
