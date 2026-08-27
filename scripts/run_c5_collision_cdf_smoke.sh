#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CHECKPOINT="${CHECKPOINT:-${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict.pt}"
ROS_PORT="${ROS_PORT:-11319}"
DEVICE="${DEVICE:-cuda}"
OUT="${OUT:-${REPO}/outputs/c5_collision_cdf_smoke}"

cd "${REPO}"
echo "[C5] branch: $(git branch --show-current)"
echo "[C5] head:   $(git rev-parse HEAD)"
echo "[C5] checkpoint: ${CHECKPOINT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] checkpoint not found:"
  echo "  ${CHECKPOINT}"
  echo ""
  echo "Copy/rename your trained file to:"
  echo "  ${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict.pt"
  exit 2
fi

# Message/service generation stays in the ordinary ROS Noetic catkin workspace.
catkin build care_collision_cdf
source devel/setup.bash

# IMPORTANT:
# The collision CDF is a PyTorch node.  Do NOT launch it through rosrun/roslaunch
# because catkin_install_python wrappers may have been generated with
# /usr/bin/python3, which does not contain the viscdf PyTorch installation.
#
# Prefer the currently active conda interpreter explicitly.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="$(command -v python || true)"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] Could not resolve an executable Python interpreter."
  echo "Activate the visibility CDF environment first:"
  echo "  conda activate viscdf"
  exit 4
fi

echo "[C5] conda env: ${CONDA_DEFAULT_ENV:-none}"
echo "[C5] conda prefix: ${CONDA_PREFIX:-none}"
echo "[C5] python: ${PYTHON_BIN}"

# Bridge ROS Noetic / Ubuntu dist-packages into the conda interpreter.
# Conda intentionally omits /usr/lib/python3/dist-packages, where rospkg and
# several ROS Python dependencies live on Ubuntu 20.04.
ROS_PY="/opt/ros/noetic/lib/python3/dist-packages"
SYS_PY="/usr/lib/python3/dist-packages"
export PYTHONPATH="${REPO}/devel/lib/python3/dist-packages:${ROS_PY}:${SYS_PY}:${PYTHONPATH:-}"

"${PYTHON_BIN}" - <<'PY'
import sys
print("[C5] python executable:", sys.executable)
print("[C5] sys.path contains ROS:", any("/opt/ros/noetic" in p for p in sys.path))
print("[C5] sys.path contains system dist-packages:", "/usr/lib/python3/dist-packages" in sys.path)
PY

if ! "${PYTHON_BIN}" - <<'PY'
import sys
import torch
import rospkg
import rospy
from care_collision_cdf.srv import QueryCollisionCDF
print("[C5] torch:", torch.__version__)
print("[C5] torch cuda available:", torch.cuda.is_available())
print("[C5] rospkg:", rospkg.__file__)
print("[C5] rospy:", rospy.__file__)
print("[C5] generated ROS service import: OK")
PY
then
  echo ""
  echo "[ERROR] Active conda Python cannot import torch + rospy + generated service."
  echo "Expected workflow:"
  echo "  conda activate viscdf"
  echo "  source /opt/ros/noetic/setup.bash"
  echo "  source ${REPO}/devel/setup.bash"
  echo "  bash scripts/run_c5_collision_cdf_smoke.sh"
  exit 5
fi

"${PYTHON_BIN}" -m py_compile \
  src/care_collision_cdf/scripts/collision_cdf_model.py \
  src/care_collision_cdf/scripts/collision_cdf_server_node.py \
  src/care_collision_cdf/scripts/inspect_collision_cdf_checkpoint.py \
  src/care_collision_cdf/scripts/query_collision_cdf_smoke.py

mkdir -p "${OUT}"

echo "[C5] inspecting checkpoint with conda Python..."
"${PYTHON_BIN}" \
  src/care_collision_cdf/scripts/inspect_collision_cdf_checkpoint.py \
  "${CHECKPOINT}" \
  | tee "${OUT}/checkpoint_inspection.txt"

export ROS_MASTER_URI="http://127.0.0.1:${ROS_PORT}"
PIDS=()

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill -INT -- "-${pid}" 2>/dev/null || true
    sleep 0.1
    kill -TERM -- "-${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

setsid roscore -p "${ROS_PORT}" >"${OUT}/roscore.log" 2>&1 &
PIDS+=("$!")

for _ in $(seq 1 100); do
  if rostopic list >/dev/null 2>&1; then
    break
  fi
  sleep 0.05
done

if ! rostopic list >/dev/null 2>&1; then
  echo "[ERROR] roscore did not become ready."
  tail -n 80 "${OUT}/roscore.log" || true
  exit 6
fi

# Directly populate the private namespace expected by rospy.init_node
# and execute the source node with the active conda Python.
rosparam load \
  src/care_collision_cdf/config/collision_cdf.yaml \
  /collision_cdf_server
rosparam set /collision_cdf_server/checkpoint "${CHECKPOINT}"
rosparam set /collision_cdf_server/device "${DEVICE}"
rosparam set /collision_cdf_server/checkpoint_key "latest"

setsid "${PYTHON_BIN}" \
  src/care_collision_cdf/scripts/collision_cdf_server_node.py \
  >"${OUT}/collision_cdf_server.log" 2>&1 &
PIDS+=("$!")

for _ in $(seq 1 200); do
  if rosservice list 2>/dev/null | grep -q '^/care_planner/collision_cdf/query$'; then
    break
  fi
  sleep 0.05
done

if ! rosservice list 2>/dev/null | grep -q '^/care_planner/collision_cdf/query$'; then
  echo "[ERROR] collision CDF service did not become ready."
  echo "[SERVER LOG]"
  tail -n 120 "${OUT}/collision_cdf_server.log" || true
  exit 3
fi

echo "[C5] querying batch distance + gradient with conda Python..."
"${PYTHON_BIN}" \
  src/care_collision_cdf/scripts/query_collision_cdf_smoke.py \
  | tee "${OUT}/query_smoke.txt"

echo ""
echo "[C5 COLLISION CDF SMOKE PASS]"
echo "[OUTPUT] ${OUT}"
