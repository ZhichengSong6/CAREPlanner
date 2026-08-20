#!/usr/bin/env bash
set -euo pipefail

REPO="/home/zhicheng/Project/CAREPlanner"
OUT="${REPO}/outputs/phase_c1_vbc_prescreen"
LOG="${REPO}/logs/phase_c1_vbc_prescreen"
NUM_GOALS="${NUM_GOALS:-20}"
SEED="${SEED:-20260820}"

cd "${REPO}"
source devel/setup.bash
mkdir -p "${OUT}/waypoint_traces" "${LOG}"

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
cleanup() {
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

roslaunch egocentric_arm_planner phaseC1_vbc_prescreen.launch \
  num_goals:="${NUM_GOALS}" \
  random_seed:="${SEED}" \
  select_num_cases:=12 \
  safety_margin_s:=0.30 \
  output_root:="${OUT}" \
  2>&1 | tee "${LOG}/prescreen_launch.log"

echo
python3 - <<PY
import json
from pathlib import Path
p=Path("${OUT}/vbc_robustness_prescreen.json")
d=json.loads(p.read_text())
print(json.dumps(d.get("summary", {}), indent=2))
print("selected_case_ids:", d.get("selected_case_ids", []))
PY
