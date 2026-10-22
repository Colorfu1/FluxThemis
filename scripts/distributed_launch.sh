#!/usr/bin/env bash
# GPU/replica options are local to each node. Other options reach Python intact.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
WORKERS=1
GPU_COUNT=""
GPU_IDS=""
FORWARD=()
while (($#)); do
    case "$1" in
        --gpus|--gpu-ids|--workers-per-gpu)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            KEY="$1"; VALUE="$2"; shift 2 ;;
        --gpus=*|--gpu-ids=*|--workers-per-gpu=*)
            KEY="${1%%=*}"; VALUE="${1#*=}"; shift ;;
        -h|--help)
            echo 'Launcher options: --gpus N, --gpu-ids ID,ID, --workers-per-gpu N (default: 1)'
            exec "$PYTHON_EXECUTABLE" "$ENTRY_POINT" "${FORWARD[@]}" "$@" ;;
        *) FORWARD+=("$1"); shift; continue ;;
    esac
    case "$KEY" in
        --gpus) GPU_COUNT="$VALUE" ;;
        --gpu-ids) GPU_IDS="$VALUE" ;;
        --workers-per-gpu) WORKERS="$VALUE" ;;
    esac
done
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo 'workers-per-gpu must be positive' >&2; exit 2; }
if [[ -n "$GPU_COUNT" && ! "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo 'gpus must be positive' >&2; exit 2
fi
# Dry-run performs one CPU config check, never torchrun or GPU discovery.
for ARG in "${FORWARD[@]}"; do
    if [[ "$ARG" == --dry-run ]]; then
        exec "$PYTHON_EXECUTABLE" "$ENTRY_POINT" --workers-per-gpu "$WORKERS" "${FORWARD[@]}"
    fi
done
ALLOCATED="${CUDA_VISIBLE_DEVICES-all}"
if [[ -n "$GPU_IDS" ]]; then
    VISIBLE="$GPU_IDS"
else
    VISIBLE="$ALLOCATED"
fi
if [[ "$VISIBLE" == all ]]; then
    GPU_COUNT="${GPU_COUNT:-${MLP_WORKER_GPU:-}}"
    if [[ -z "$GPU_COUNT" ]]; then
        GPU_COUNT="$("$PYTHON_EXECUTABLE" -c 'import torch; print(torch.cuda.device_count())')"
    fi
    [[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo 'No GPUs available' >&2; exit 2; }
    VISIBLE="$(seq -s, 0 "$((GPU_COUNT - 1))")"
fi
IFS=, read -r -a DEVICES <<< "$VISIBLE"
[[ -n "$VISIBLE" && "$VISIBLE" != *, && "$VISIBLE" != ,* && "$VISIBLE" != *,,* ]] || { echo 'Empty GPU IDs' >&2; exit 2; }
if [[ -n "$GPU_IDS" ]]; then
    GPU_COUNT="${GPU_COUNT:-${#DEVICES[@]}}"
else
    GPU_COUNT="${GPU_COUNT:-${MLP_WORKER_GPU:-${#DEVICES[@]}}}"
fi
[[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo 'gpus must be positive' >&2; exit 2; }
if ((GPU_COUNT > ${#DEVICES[@]})) || { [[ -n "$GPU_IDS" ]] && ((GPU_COUNT != ${#DEVICES[@]})); }; then
    echo 'GPU count disagrees with GPU IDs/visible allocation' >&2; exit 2
fi
DEVICES=("${DEVICES[@]:0:GPU_COUNT}")
SEEN=,
for GPU in "${DEVICES[@]}"; do
    [[ "$GPU" =~ ^[0-9]+$ || "$GPU" == GPU-* ]] || { echo "Invalid GPU: $GPU" >&2; exit 2; }
    [[ "$SEEN" != *",$GPU,"* ]] || { echo 'Duplicate GPU IDs' >&2; exit 2; }
    if [[ "$ALLOCATED" != all && ",$ALLOCATED," != *",$GPU,"* ]]; then
        echo "GPU $GPU is outside CUDA_VISIBLE_DEVICES" >&2; exit 2
    fi
    SEEN+="$GPU,"
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${DEVICES[*]}")"
EXPECTED_NPROC=$((GPU_COUNT * WORKERS))
if [[ -n "${NPROC_PER_NODE:-}" && "$NPROC_PER_NODE" != gpu && "$NPROC_PER_NODE" != "$EXPECTED_NPROC" ]]; then
    echo "NPROC_PER_NODE=$NPROC_PER_NODE conflicts with GPUs * workers-per-gpu=$EXPECTED_NPROC" >&2; exit 2
fi
export NPROC_PER_NODE="$EXPECTED_NPROC"
NODE_COUNT="${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}"
NODE_RANK="${RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-localhost}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-$DEFAULT_PORT}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$PYTHON_EXECUTABLE" -m torch.distributed.run \
    --nproc-per-node="$NPROC_PER_NODE" \
    --nnodes="$NODE_COUNT" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    "$ENTRY_POINT" \
    --workers-per-gpu "$WORKERS" "${FORWARD[@]}"
