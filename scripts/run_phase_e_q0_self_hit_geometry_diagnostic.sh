#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
DURATION="${DURATION:-8}"
MAX_RAYS="${MAX_RAYS:-12}"

cd "${REPO}"
source /opt/ros/noetic/setup.bash
if [[ -f "${REPO}/devel/setup.bash" ]]; then
  source "${REPO}/devel/setup.bash"
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="${REPO}/outputs/phase_e_q0_self_hit_geometry/${STAMP}"
mkdir -p "${ROOT}"
LOG="${ROOT}/diagnostic.log"
JSON="${ROOT}/q0_self_hit_geometry.json"

cleanup() {
  set +e
  [[ -n "${LAUNCH_PID:-}" ]] && kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
  sleep 0.5
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  sleep 0.5
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${n}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

cleanup
trap cleanup EXIT INT TERM

echo "=============================================================="
echo "PHASE-E q0 SELF-HIT GEOMETRY DIAGNOSTIC"
echo "world      : ${WORLD_FILE}"
echo "sensor     : /link2_sensor2/tof/cloud"
echo "watch voxel: [0.20, -0.05, 0.45] +/- 0.025 m"
echo "planner    : OFF"
echo "q0         : exact zero/home (gazebo launch defaults)"
echo "=============================================================="

roslaunch care_confidence_map phase_e_sensor_smoke.launch   world_file:="${WORLD_FILE}"   gazebo_gui:="${GAZEBO_GUI}"   use_rviz:="${USE_RVIZ}"   paused:=false   > "${ROOT}/roslaunch.log" 2>&1 &
LAUNCH_PID=$!

echo "[WAIT] /care_arm/joint_states"
timeout 30 rostopic echo -n 1 /care_arm/joint_states >/dev/null

echo "[WAIT] /link2_sensor2/tof/cloud"
timeout 30 rostopic echo -n 1 /link2_sensor2/tof/cloud >/dev/null

echo "[RUN] exact visual-mesh ray intersection"
python3 scripts/diagnose_phase_e_q0_self_hit_geometry.py   --repo "${REPO}"   --duration "${DURATION}"   --max-rays "${MAX_RAYS}"   --output-json "${JSON}"   2>&1 | tee "${LOG}"

GIT_SHORT="$(git rev-parse --short=8 HEAD)"
ZIP="${REPO}/CAREPlanner_PHASE_E_Q0_SELF_HIT_GEOMETRY_${STAMP}_${GIT_SHORT}.zip"
rm -f "${ZIP}"
python3 - "${ROOT}" "${ZIP}" <<'PY'
import os, sys, zipfile
root, dst = sys.argv[1:]
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for base, _, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            z.write(path, os.path.relpath(path, root))
print(dst)
PY

echo ""
echo "[RESULT] ${JSON}"
echo "[LOG]    ${LOG}"
echo "[UPLOAD] ${ZIP}"
ls -lh "${ZIP}"
