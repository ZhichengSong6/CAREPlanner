#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CHECKPOINT="${CHECKPOINT:-${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict.pt}"
CDF_ENV="${CDF_ENV:-ncdf_l4c}"
DEVICE="${DEVICE:-cpu}"
ROS_PORT="${ROS_PORT:-11319}"
OUT="${OUT:-${REPO}/outputs/c5_collision_cdf_smoke}"

cd "${REPO}"
echo "[C5] branch: $(git branch --show-current)"
echo "[C5] head:   $(git rev-parse HEAD)"
echo "[C5] checkpoint: ${CHECKPOINT}"
echo "[C5] CDF runtime conda env: ${CDF_ENV}"
echo "[C5] device: ${DEVICE}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] checkpoint not found:"
  echo "  ${CHECKPOINT}"
  echo ""
  echo "Copy/rename your trained file to:"
  echo "  ${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict.pt"
  exit 2
fi

# Exactly the same split used by the working visibility-CDF C4.x runners:
# ordinary ROS/catkin in this shell, learned CDF Python node in its own conda
# subshell.  Do not mix /usr/lib/python3/dist-packages into a conda PYTHONPATH.
catkin build care_collision_cdf
source devel/setup.bash

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found"
  exit 3
fi

mkdir -p "${OUT}"

echo "[C5] validating CDF runtime env..."
bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${CDF_ENV}'
  cd '${REPO}'
  python - <<'PY'
import sys
import torch
print('[C5-conda] python:', sys.executable)
print('[C5-conda] torch:', torch.__version__)
print('[C5-conda] cuda available:', torch.cuda.is_available())
PY
"

echo "[C5] checking Python syntax in CDF runtime env..."
bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${CDF_ENV}'
  cd '${REPO}'
  python -m py_compile \
    src/care_collision_cdf/scripts/collision_cdf_model.py \
    src/care_collision_cdf/scripts/collision_cdf_server_node.py \
    src/care_collision_cdf/scripts/inspect_collision_cdf_checkpoint.py
"

echo "[C5] inspecting checkpoint in CDF runtime env..."
bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${CDF_ENV}'
  cd '${REPO}'
  exec python -u src/care_collision_cdf/scripts/inspect_collision_cdf_checkpoint.py '${CHECKPOINT}'
" | tee "${OUT}/checkpoint_inspection.txt"

export ROS_MASTER_URI="http://127.0.0.1:${ROS_PORT}"
PIDS=()

kill_group() {
  local pid="${1:-}"
  [[ -z "${pid}" ]] && return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || true
    sleep 0.20
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 0.20
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill_group "${pid}"
  done
}
trap cleanup EXIT INT TERM

setsid roscore -p "${ROS_PORT}" >"${OUT}/roscore.log" 2>&1 &
PIDS+=("$!")

READY=0
for _ in $(seq 1 100); do
  if rostopic list >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 0.05
done
if [[ "${READY}" != "1" ]]; then
  echo "[ERROR] roscore did not become ready"
  tail -n 100 "${OUT}/roscore.log" || true
  exit 4
fi

# Parameters are written by the normal ROS shell.  The learned server is then
# started exactly like care_visibility_cdf in C4.x: activate conda, source the
# catkin workspace inside that subshell, and exec the source Python file.
rosparam load \
  src/care_collision_cdf/config/collision_cdf.yaml \
  /collision_cdf_server
rosparam set /collision_cdf_server/checkpoint "${CHECKPOINT}"
rosparam set /collision_cdf_server/device "${DEVICE}"
rosparam set /collision_cdf_server/checkpoint_key "latest"

setsid bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${CDF_ENV}'
  cd '${REPO}'
  source devel/setup.bash
  exec python -u src/care_collision_cdf/scripts/collision_cdf_server_node.py
" >"${OUT}/collision_cdf_server.log" 2>&1 &
PIDS+=("$!")

READY=0
for _ in $(seq 1 240); do
  if rosservice list 2>/dev/null | grep -q '^/care_planner/collision_cdf/query$'; then
    READY=1
    break
  fi
  if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
    break
  fi
  sleep 0.05
done

if [[ "${READY}" != "1" ]]; then
  echo "[ERROR] collision CDF service did not become ready."
  echo "[SERVER LOG]"
  tail -n 160 "${OUT}/collision_cdf_server.log" || true
  exit 5
fi

echo "[C5] querying batch distance + gradient from ordinary ROS Python..."
rosrun care_collision_cdf query_collision_cdf_smoke.py \
  | tee "${OUT}/query_smoke.txt"

echo ""
echo "[C5 COLLISION CDF SMOKE PASS]"
echo "[OUTPUT] ${OUT}"
