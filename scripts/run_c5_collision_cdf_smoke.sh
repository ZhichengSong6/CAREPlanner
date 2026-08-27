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

catkin build care_collision_cdf
source devel/setup.bash

python3 -m py_compile   src/care_collision_cdf/scripts/collision_cdf_model.py   src/care_collision_cdf/scripts/collision_cdf_server_node.py   src/care_collision_cdf/scripts/inspect_collision_cdf_checkpoint.py   src/care_collision_cdf/scripts/query_collision_cdf_smoke.py

mkdir -p "${OUT}"
echo "[C5] inspecting checkpoint..."
rosrun care_collision_cdf inspect_collision_cdf_checkpoint.py "${CHECKPOINT}"   | tee "${OUT}/checkpoint_inspection.txt"

export ROS_MASTER_URI="http://127.0.0.1:${ROS_PORT}"
PIDS=()
cleanup() {
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
  if rostopic list >/dev/null 2>&1; then break; fi
  sleep 0.05
done

setsid roslaunch care_collision_cdf collision_cdf_server.launch   checkpoint:="${CHECKPOINT}" device:="${DEVICE}"   >"${OUT}/collision_cdf_server.log" 2>&1 &
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
  tail -n 80 "${OUT}/collision_cdf_server.log" || true
  exit 3
fi

echo "[C5] querying batch distance + gradient..."
rosrun care_collision_cdf query_collision_cdf_smoke.py   | tee "${OUT}/query_smoke.txt"

echo ""
echo "[C5 COLLISION CDF SMOKE PASS]"
echo "[OUTPUT] ${OUT}"
