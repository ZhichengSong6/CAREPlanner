#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json"
OUT="${REPO}/outputs/phase_c3_predictive_recovery_smoke"
LOG="${REPO}/logs/phase_c3_predictive_recovery_smoke"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-6.0}"
PRED_ERROR_INF="${PRED_ERROR_INF:-0.10}"
PRED_MIN_IMPROVEMENT_INF="${PRED_MIN_IMPROVEMENT_INF:-0.002}"
PRED_STREAK="${PRED_STREAK:-3}"

cd "${REPO}" || exit 1
source devel/setup.bash
rm -rf "${OUT}" "${LOG}"
mkdir -p "${OUT}/projector_traces" "${LOG}"

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
d=json.load(open(p)); c=next(x for x in d['cases'] if x['case_id']==cid)
print(*c['goal_position'],*c['goal_orientation'])
PY
)"

GAZEBO_PID=""; GEN_PID=""; CONTROL_PID=""
kill_group() {
  local pid="${1:-}"
  [ -z "${pid}" ] && return 0
  if kill -0 "${pid}" 2>/dev/null; then kill -INT -- "-${pid}" 2>/dev/null || true; sleep 1; fi
  if kill -0 "${pid}" 2>/dev/null; then kill -TERM -- "-${pid}" 2>/dev/null || true; sleep 1; fi
  if kill -0 "${pid}" 2>/dev/null; then kill -KILL -- "-${pid}" 2>/dev/null || true; fi
  wait "${pid}" 2>/dev/null || true
}
cleanup() {
  kill_group "${CONTROL_PID}"; kill_group "${GEN_PID}"; kill_group "${GAZEBO_PID}"
  pkill -TERM -x gzserver 2>/dev/null || true
  pkill -TERM -x rosmaster 2>/dev/null || true
  pkill -TERM -x roscore 2>/dev/null || true
  pkill -TERM -x roslaunch 2>/dev/null || true
}
trap cleanup EXIT INT TERM

setsid roslaunch arm_description gazebo_velocity_control.launch gazebo_gui:=false use_rviz:=false \
  > "${LOG}/gazebo.log" 2>&1 &
GAZEBO_PID=$!

READY=0
for _ in $(seq 1 120); do
  if timeout 2 rostopic echo -n 1 /care_arm/joint_states >/dev/null 2>&1; then READY=1; break; fi
  sleep 0.25
done
if [ "${READY}" != "1" ]; then echo "[ERROR] Gazebo joint state timeout"; exit 1; fi

setsid bash -lc "source '${CONDA_SH}'; conda activate ncdf_l4c; cd '${REPO}'; source devel/setup.bash; exec python -u src/care_visibility_cdf/scripts/vbc_deadline_waypoint_node.py _device:=cpu _safety_margin_s:=0.30 _projection_iters:=10 _projection_damping:=0.5 _projection_epsilon_f:=0.03 _projection_max_step_norm:=0.25 _root_refine_iters:=12 _root_tolerance_f:=0.002 _ascent_steps:=1 _ascent_step_size:=0.05 _ascent_max_step_norm:=0.25 _output_root:='${OUT}/projector_traces'" \
  > "${LOG}/waypoint_generator.log" 2>&1 &
GEN_PID=$!

READY=0
for _ in $(seq 1 120); do
  if grep -q "frozen model READY" "${LOG}/waypoint_generator.log" 2>/dev/null; then READY=1; break; fi
  sleep 1
done
if [ "${READY}" != "1" ]; then echo "[ERROR] projector not ready"; exit 1; fi

setsid roslaunch egocentric_arm_planner phaseB2_vbc_waypoint_controlled.launch \
  waypoint_weight:=3000.0 recovery_enabled:=true recovery_weight_scale:=1.0 \
  predictive_recovery_enabled:=true \
  predictive_recovery_error_threshold_inf:="${PRED_ERROR_INF}" \
  predictive_recovery_min_improvement_inf:="${PRED_MIN_IMPROVEMENT_INF}" \
  predictive_recovery_consecutive_misses:="${PRED_STREAK}" \
  vbc_min_margin_s:=0.30 trial_label:="${CASE_ID}_predictive_recovery_smoke" \
  log_output_root:="${OUT}" use_trajectory_visualizer:=false \
  goal_x:="${GX}" goal_y:="${GY}" goal_z:="${GZ}" \
  goal_qx:="${GQX}" goal_qy:="${GQY}" goal_qz:="${GQZ}" goal_qw:="${GQW}" \
  > "${LOG}/controlled.log" 2>&1 &
CONTROL_PID=$!

RELEASED=0
for _ in $(seq 1 300); do
  S="$(timeout 2 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.1
done
if [ "${RELEASED}" != "1" ]; then echo "[ERROR] initial gate release timeout"; exit 1; fi

echo "[RUN] ${CASE_ID}: initial gate released; observing predictive recovery for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

MPC="$(timeout 2 rostopic echo -n 1 /velocity_qp_mpc_waypoint_node/summary 2>/dev/null || true)"
GATE="$(timeout 2 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
PRED="$(timeout 2 rostopic echo -n 1 /care_planner/execution/predictive_recovery_summary 2>/dev/null || true)"
printf '%s\n' "${MPC}" > "${OUT}/final_mpc_summary.yaml"
printf '%s\n' "${GATE}" > "${OUT}/final_gate_summary.yaml"
printf '%s\n' "${PRED}" > "${OUT}/final_predictive_recovery_summary.yaml"

TRIGGER_COUNT="$(grep -c 'PREDICTIVE RECOVERY TRIGGER' "${LOG}/controlled.log" 2>/dev/null || true)"
RECOVERY_COUNT="$(grep -c 'control_mode=recovery' "${LOG}/controlled.log" 2>/dev/null || true)"
HOLD_COUNT="$(grep -c 'RECOVERY COMPLETE\|recovery_hold' "${LOG}/controlled.log" 2>/dev/null || true)"
REPLAN_COUNT="$(echo "${GATE}" | sed -n 's/.*replan_count=\([0-9][0-9]*\).*/\1/p' | head -n1)"
REPLAN_COUNT="${REPLAN_COUNT:-0}"
TRIGGER_LEAD="$(grep 'PREDICTIVE RECOVERY TRIGGER' "${LOG}/controlled.log" 2>/dev/null | sed -n 's/.*physical_deadline_lead=\([-0-9.]*\)s.*/\1/p' | head -n1)"
TRIGGER_LEAD="${TRIGGER_LEAD:-nan}"

echo "[CHECK] predictive trigger hits=${TRIGGER_COUNT}"
echo "[CHECK] first predictive trigger lead=${TRIGGER_LEAD}s"
echo "[CHECK] recovery log hits=${RECOVERY_COUNT}"
echo "[CHECK] recovery-complete/hold log hits=${HOLD_COUNT}"
echo "[CHECK] gate replan_count=${REPLAN_COUNT}"

if [ "${TRIGGER_COUNT}" -lt 1 ]; then
  echo "[FAIL] predictive recovery was never triggered"
  exit 2
fi
if ! python3 - "${TRIGGER_LEAD}" <<'PY'
import math,sys
try: x=float(sys.argv[1])
except Exception: raise SystemExit(1)
raise SystemExit(0 if math.isfinite(x) and x > 0.0 else 1)
PY
then
  echo "[FAIL] predictive trigger did not occur before the physical deadline"
  exit 2
fi
if [ "${RECOVERY_COUNT}" -lt 1 ]; then
  echo "[FAIL] recovery was never entered after predictive trigger"
  exit 2
fi
if [ "${HOLD_COUNT}" -lt 1 ]; then
  echo "[FAIL] recovery never completed / hold was never observed"
  exit 2
fi
if [ "${REPLAN_COUNT}" -lt 1 ]; then
  echo "[FAIL] post-recovery task replan was not installed"
  exit 2
fi

echo "[PASS] ${CASE_ID}: predictive miss -> early recovery -> recovery hold -> task replan observed"
echo "[RESULT] ${OUT}"
