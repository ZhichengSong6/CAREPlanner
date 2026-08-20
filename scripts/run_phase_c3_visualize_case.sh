#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
CASE_ID="${CASE_ID:-case_003}"
MODE="${MODE:-predictive_recovery}"
RUN_SECONDS="${RUN_SECONDS:-8.0}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
PRED_ERROR_INF="${PRED_ERROR_INF:-0.10}"
PRED_STREAK="${PRED_STREAK:-3}"
PRED_MIN_IMPROVEMENT_INF="${PRED_MIN_IMPROVEMENT_INF:-0.002}"
OUT="${OUT:-${REPO}/outputs/phase_c3_visualization/${CASE_ID}/${MODE}}"
LOG="${LOG:-${REPO}/logs/phase_c3_visualization/${CASE_ID}/${MODE}}"

cd "${REPO}" || exit 1
source devel/setup.bash
rm -rf "${OUT}" "${LOG}"
mkdir -p "${OUT}/projector_traces" "${LOG}"

case "${MODE}" in
  baseline)
    WEIGHT=0.0; RECOVERY_ENABLED=false; PREDICTIVE_ENABLED=false ;;
  deadline_recovery)
    WEIGHT="${CARE_WEIGHT}"; RECOVERY_ENABLED=true; PREDICTIVE_ENABLED=false ;;
  predictive_recovery)
    WEIGHT="${CARE_WEIGHT}"; RECOVERY_ENABLED=true; PREDICTIVE_ENABLED=true ;;
  *)
    echo "[ERROR] MODE must be baseline, deadline_recovery, or predictive_recovery"
    exit 2 ;;
esac

if timeout 2 rosnode list >/dev/null 2>&1; then
  echo "[ERROR] ROS master already running. Close other ROS/Gazebo sessions first."
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

read -r GX GY GZ GQX GQY GQZ GQW <<< "$(python3 - "${CASE_FILE}" "${CASE_ID}" <<'PY'
import json,sys
p,cid=sys.argv[1:]
d=json.load(open(p)); c=next((x for x in d['cases'] if x['case_id']==cid),None)
if c is None: raise SystemExit('unknown case_id: '+cid)
print(*c['goal_position'],*c['goal_orientation'])
PY
)"

GAZEBO_PID=""; GEN_PID=""; CONTROL_PID=""
kill_group() {
  local pid="${1:-}"; [ -z "${pid}" ] && return 0
  if kill -0 "${pid}" 2>/dev/null; then kill -INT -- "-${pid}" 2>/dev/null || true; sleep 1; fi
  if kill -0 "${pid}" 2>/dev/null; then kill -TERM -- "-${pid}" 2>/dev/null || true; sleep 1; fi
  if kill -0 "${pid}" 2>/dev/null; then kill -KILL -- "-${pid}" 2>/dev/null || true; fi
  wait "${pid}" 2>/dev/null || true
}
cleanup() {
  kill_group "${CONTROL_PID}"; kill_group "${GEN_PID}"; kill_group "${GAZEBO_PID}"
  pkill -TERM -x gzclient 2>/dev/null || true
  pkill -TERM -x gzserver 2>/dev/null || true
  pkill -TERM -x rviz 2>/dev/null || true
  pkill -TERM -x rosmaster 2>/dev/null || true
  pkill -TERM -x roscore 2>/dev/null || true
  pkill -TERM -x roslaunch 2>/dev/null || true
}
trap cleanup EXIT INT TERM

setsid roslaunch arm_description gazebo_velocity_control.launch gazebo_gui:=true use_rviz:=true > "${LOG}/gazebo.log" 2>&1 &
GAZEBO_PID=$!

READY=0
for _ in $(seq 1 120); do
  if timeout 2 rostopic echo -n 1 /care_arm/joint_states >/dev/null 2>&1; then READY=1; break; fi
  sleep 0.25
done
if [ "${READY}" != "1" ]; then echo "[ERROR] Gazebo joint state timeout"; exit 1; fi

setsid bash -lc "source '${CONDA_SH}'; conda activate '${NCDF_ENV}'; cd '${REPO}'; source devel/setup.bash; exec python -u src/care_visibility_cdf/scripts/vbc_deadline_waypoint_node.py _device:=cpu _safety_margin_s:=${SAFETY_MARGIN} _projection_iters:=10 _projection_damping:=0.5 _projection_epsilon_f:=0.03 _projection_max_step_norm:=0.25 _root_refine_iters:=12 _root_tolerance_f:=0.002 _ascent_steps:=1 _ascent_step_size:=0.05 _ascent_max_step_norm:=0.25 _output_root:='${OUT}/projector_traces'" > "${LOG}/waypoint_generator.log" 2>&1 &
GEN_PID=$!

READY=0
for _ in $(seq 1 120); do
  if grep -q "frozen model READY" "${LOG}/waypoint_generator.log" 2>/dev/null; then READY=1; break; fi
  sleep 1
done
if [ "${READY}" != "1" ]; then echo "[ERROR] projector not ready"; exit 1; fi

setsid roslaunch egocentric_arm_planner phaseB2_vbc_waypoint_controlled.launch \
  waypoint_weight:="${WEIGHT}" recovery_enabled:="${RECOVERY_ENABLED}" recovery_weight_scale:=1.0 \
  predictive_recovery_enabled:="${PREDICTIVE_ENABLED}" \
  predictive_recovery_error_threshold_inf:="${PRED_ERROR_INF}" \
  predictive_recovery_min_improvement_inf:="${PRED_MIN_IMPROVEMENT_INF}" \
  predictive_recovery_consecutive_misses:="${PRED_STREAK}" \
  vbc_min_margin_s:="${SAFETY_MARGIN}" trial_label:="${CASE_ID}_${MODE}_visual" \
  log_output_root:="${OUT}" use_trajectory_visualizer:=true \
  goal_x:="${GX}" goal_y:="${GY}" goal_z:="${GZ}" \
  goal_qx:="${GQX}" goal_qy:="${GQY}" goal_qz:="${GQZ}" goal_qw:="${GQW}" \
  > "${LOG}/controlled.log" 2>&1 &
CONTROL_PID=$!

echo "[VISUALIZE] case=${CASE_ID} mode=${MODE}"
echo "[VISUALIZE] Gazebo/RViz are enabled; observing for ${RUN_SECONDS}s after gate release."

RELEASED=0
for _ in $(seq 1 300); do
  S="$(timeout 2 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.1
done
if [ "${RELEASED}" != "1" ]; then echo "[ERROR] initial gate release timeout"; exit 1; fi

sleep "${RUN_SECONDS}"

echo "[RESULT] ${OUT}"
echo "[LOG] ${LOG}"
