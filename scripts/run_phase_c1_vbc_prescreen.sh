#!/usr/bin/env bash
set -euo pipefail

REPO="/home/zhicheng/Project/CAREPlanner"
OUT="${REPO}/outputs/phase_c1_vbc_prescreen"
LOG="${REPO}/logs/phase_c1_vbc_prescreen"
NUM_GOALS="${NUM_GOALS:-20}"
SEED="${SEED:-20260820}"

cd "${REPO}"
source devel/setup.bash

# Preserve any previous C1 run instead of mixing traces/results across runs.
if [ -d "${OUT}" ] && [ -n "$(find "${OUT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  PREV_STAMP="$(date +%Y%m%d_%H%M%S)"
  mv "${OUT}" "${OUT}_previous_${PREV_STAMP}"
fi
mkdir -p "${OUT}/waypoint_traces" "${LOG}/ros"

# Isolate roslaunch logs for this experiment. This avoids the global
# ~/.ros/log >1GB warning without deleting the user's existing ROS logs.
export ROS_LOG_DIR="${LOG}/ros"

# Start Gazebo separately before this script. It supplies TF + JointState but no
# execution node is launched in Phase C1, so q0 remains fixed.
if ! timeout 3 rostopic echo -n 1 /care_arm/joint_states >/dev/null 2>&1; then
  echo "[ERROR] No /care_arm/joint_states. Start Gazebo first:"
  echo "  roslaunch arm_description gazebo_velocity_control.launch"
  exit 1
fi

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found"
  exit 1
fi

GEN_PID=""
LAUNCH_PID=""
cleanup() {
  if [ -n "${LAUNCH_PID}" ] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    kill -INT "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
  if [ -n "${GEN_PID}" ] && kill -0 "${GEN_PID}" 2>/dev/null; then
    kill -INT "${GEN_PID}" 2>/dev/null || true
    wait "${GEN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  source "${CONDA_SH}"
  conda activate ncdf_l4c
  cd "${REPO}"
  source devel/setup.bash
  exec python -u src/care_visibility_cdf/scripts/vbc_deadline_waypoint_node.py \
    _device:=cpu \
    _safety_margin_s:=0.30 \
    _projection_iters:=10 \
    _projection_damping:=0.5 \
    _projection_epsilon_f:=0.03 \
    _projection_max_step_norm:=0.25 \
    _root_refine_iters:=12 \
    _root_tolerance_f:=0.002 \
    _ascent_steps:=1 \
    _ascent_step_size:=0.05 \
    _ascent_max_step_norm:=0.25 \
    _output_root:="${OUT}/waypoint_traces"
) > "${LOG}/waypoint_generator.log" 2>&1 &
GEN_PID=$!

for _ in $(seq 1 120); do
  if grep -q "frozen model READY" "${LOG}/waypoint_generator.log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${GEN_PID}" 2>/dev/null; then
    echo "[ERROR] waypoint generator exited"
    tail -n 80 "${LOG}/waypoint_generator.log"
    exit 1
  fi
  sleep 1
done

if ! grep -q "frozen model READY" "${LOG}/waypoint_generator.log"; then
  echo "[ERROR] waypoint generator did not become ready"
  exit 1
fi

echo "[READY] frozen VisCDF loaded"
echo "[RUN] deterministic Phase-C1 prescreen: NUM_GOALS=${NUM_GOALS}, SEED=${SEED}"

# Run roslaunch in the background. The prescreen node exits normally when all
# goals are processed; this wrapper then shuts the remaining read-only nodes
# down cleanly instead of using required=true (which prints a misleading
# "REQUIRED process has died" banner even on success).
roslaunch egocentric_arm_planner phaseC1_vbc_prescreen.launch \
  num_goals:="${NUM_GOALS}" \
  random_seed:="${SEED}" \
  select_num_cases:=12 \
  safety_margin_s:=0.30 \
  output_root:="${OUT}" \
  > >(tee "${LOG}/prescreen_launch.log") 2>&1 &
LAUNCH_PID=$!

DONE=0
for _ in $(seq 1 600); do
  if [ -f "${OUT}/vbc_robustness_prescreen.json" ]; then
    DONE=$(python3 - <<PY
import json
from pathlib import Path
p=Path("${OUT}/vbc_robustness_prescreen.json")
try:
    d=json.loads(p.read_text())
    print(1 if int(d.get("summary", {}).get("num_attempted", -1)) >= int("${NUM_GOALS}") else 0)
except Exception:
    print(0)
PY
)
    if [ "${DONE}" = "1" ]; then
      break
    fi
  fi
  if ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
    echo "[ERROR] Phase-C1 roslaunch exited before completion"
    exit 1
  fi
  sleep 0.25
done

if [ "${DONE}" != "1" ]; then
  echo "[ERROR] Phase-C1 prescreen did not finish within 150 s"
  exit 1
fi

kill -INT "${LAUNCH_PID}" 2>/dev/null || true
wait "${LAUNCH_PID}" 2>/dev/null || true
LAUNCH_PID=""

# Replace topic-race-prone projector fields with the authoritative trace result
# and rebuild the difficulty selection.  require-all makes trace pairing a hard
# validation gate before any benchmark cases are accepted.
python3 scripts/finalize_phase_c1_vbc_prescreen.py \
  --input "${OUT}/vbc_robustness_prescreen.json" \
  --trace-dir "${OUT}/waypoint_traces" \
  --select-num 12 \
  --require-all

echo
python3 - <<PY
import json
from pathlib import Path
p=Path("${OUT}/vbc_robustness_prescreen.json")
d=json.loads(p.read_text())
print(json.dumps(d.get("summary", {}), indent=2))
print("selected_case_ids:", d.get("selected_case_ids", []))
PY
