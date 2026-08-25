#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
CONFIG_FILE="${CONFIG_FILE:-${REPO}/src/egocentric_arm_planner/config/planner_phase1_c4_3_planner.yaml}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-8.0}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
C4_STREAK="${C4_STREAK:-2}"
C4_SAFE_STREAK="${C4_SAFE_STREAK:-2}"
C4_INPUT_TIMEOUT="${C4_INPUT_TIMEOUT:-0.25}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.20}"
AUDIT_WARN_MS="${AUDIT_WARN_MS:-5.0}"
OUT="${OUT:-${REPO}/outputs/phase_c4_3_planner_split_smoke/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_3_planner_split_smoke/${CASE_ID}}"

ACTUATOR_TOPIC="/care_arm/arm_group_velocity_controller/command"
OPTIMIZED_TOPIC="/care_planner/mpc/predicted_trajectory"
MPC_UNUSED_COMMAND_TOPIC="/care_planner/mpc/internal_velocity_command_unused"

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
d=json.load(open(p)); c=next((x for x in d['cases'] if x['case_id']==cid),None)
if c is None: raise SystemExit('unknown case_id: '+cid)
print(*c['goal_position'],*c['goal_orientation'])
PY
)"

GAZEBO_PID=""; GEN_PID=""; CONTROL_PID=""; TRACKER_PID=""
REC_PIDS=()
kill_group() {
  local pid="${1:-}"
  [ -z "${pid}" ] && return 0
  if kill -0 "${pid}" 2>/dev/null; then kill -INT -- "-${pid}" 2>/dev/null || true; sleep 0.4; fi
  if kill -0 "${pid}" 2>/dev/null; then kill -TERM -- "-${pid}" 2>/dev/null || true; sleep 0.4; fi
  if kill -0 "${pid}" 2>/dev/null; then kill -KILL -- "-${pid}" 2>/dev/null || true; fi
  wait "${pid}" 2>/dev/null || true
}
cleanup() {
  for pid in "${REC_PIDS[@]:-}"; do kill_group "${pid}"; done
  REC_PIDS=()
  kill_group "${CONTROL_PID}"; CONTROL_PID=""
  kill_group "${GEN_PID}"; GEN_PID=""
  kill_group "${TRACKER_PID}"; TRACKER_PID=""
  kill_group "${GAZEBO_PID}"; GAZEBO_PID=""
  for name in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${name}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

setsid roslaunch arm_description gazebo_velocity_control.launch \
  gazebo_gui:=false use_rviz:=false > "${LOG}/gazebo.log" 2>&1 &
GAZEBO_PID=$!

READY=0
for _ in $(seq 1 120); do
  if timeout 2 rostopic echo -n 1 /care_arm/joint_states >/dev/null 2>&1; then
    READY=1; break
  fi
  sleep 0.25
done
[ "${READY}" = "1" ] || { echo "[ERROR] Gazebo joint state timeout"; exit 1; }

# Low-level controller is now the sole owner of the actuator velocity topic.
setsid roslaunch egocentric_arm_planner c4_3_low_level_tracker.launch \
  config_file:="${CONFIG_FILE}" input_trajectory:="${OPTIMIZED_TOPIC}" \
  > "${LOG}/low_level_tracker.log" 2>&1 &
TRACKER_PID=$!

setsid bash -lc "source '${CONDA_SH}'; conda activate '${NCDF_ENV}'; cd '${REPO}'; source devel/setup.bash; exec python -u src/care_visibility_cdf/scripts/vbc_deadline_waypoint_rolling_node.py _device:=cpu _safety_margin_s:=${SAFETY_MARGIN} _predicted_trajectory_timeout:=${PREDICTION_TIMEOUT} _target_cell_resolution:=0.05 _projection_iters:=10 _projection_damping:=0.5 _projection_epsilon_f:=0.03 _projection_max_step_norm:=0.25 _root_refine_iters:=12 _root_tolerance_f:=0.002 _ascent_steps:=1 _ascent_step_size:=0.05 _ascent_max_step_norm:=0.25 _output_root:='${OUT}/projector_traces'" \
  > "${LOG}/waypoint_generator.log" 2>&1 &
GEN_PID=$!

READY=0
for _ in $(seq 1 120); do
  if grep -q "frozen model READY" "${LOG}/waypoint_generator.log" 2>/dev/null; then
    READY=1; break
  fi
  if ! kill -0 "${GEN_PID}" 2>/dev/null; then break; fi
  sleep 1
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] rolling waypoint generator not ready"
  tail -n 100 "${LOG}/waypoint_generator.log" || true
  exit 1
fi

setsid roslaunch egocentric_arm_planner phaseB2_vbc_waypoint_controlled.launch \
  config_file:="${CONFIG_FILE}" \
  waypoint_weight:="${CARE_WEIGHT}" \
  recovery_enabled:=true recovery_weight_scale:=1.0 \
  rolling_target_mode:=true \
  rolling_safe_clear_consecutive:="${C4_SAFE_STREAK}" \
  selector_prefer_predicted_trajectory:=true \
  selector_predicted_trajectory_timeout:="${PREDICTION_TIMEOUT}" \
  recovery_physical_deadline_topic:=/care_planner/active_sensing/visibility_waypoint_deadline_rolling \
  predictive_recovery_enabled:=false \
  enable_predicted_vbc_auditor:=true \
  predicted_vbc_audit_warn_ms:="${AUDIT_WARN_MS}" \
  predicted_vbc_recovery_enabled:=true \
  predicted_vbc_recovery_use_global_selector:=true \
  predicted_vbc_recovery_consecutive_violations:="${C4_STREAK}" \
  predicted_vbc_recovery_consecutive_safe:="${C4_SAFE_STREAK}" \
  predicted_vbc_recovery_input_timeout:="${C4_INPUT_TIMEOUT}" \
  vbc_min_margin_s:="${SAFETY_MARGIN}" \
  enable_executed_vbc_logger:=false \
  enable_waypoint_logger:=false \
  trial_label:="${CASE_ID}_c4_3_planner_split" \
  log_output_root:="${OUT}" \
  use_trajectory_visualizer:=false \
  goal_x:="${GX}" goal_y:="${GY}" goal_z:="${GZ}" \
  goal_qx:="${GQX}" goal_qy:="${GQY}" goal_qz:="${GQZ}" goal_qw:="${GQW}" \
  > "${LOG}/controlled.log" 2>&1 &
CONTROL_PID=$!

READY=0
for _ in $(seq 1 250); do
  NODES="$(rosnode list 2>/dev/null || true)"
  if echo "${NODES}" | grep -q '^/predicted_vbc_recovery_guard$' && \
     echo "${NODES}" | grep -q '^/trajectory_vbc_selector_node$' && \
     echo "${NODES}" | grep -q '^/phase_b2_controlled_trial$' && \
     echo "${NODES}" | grep -q '^/velocity_qp_mpc_waypoint_node$' && \
     echo "${NODES}" | grep -q '^/trajectory_execution_manager_node$'; then
    READY=1; break
  fi
  sleep 0.1
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] planner/controller-split nodes did not start"
  tail -n 140 "${LOG}/controlled.log" || true
  tail -n 100 "${LOG}/low_level_tracker.log" || true
  exit 1
fi

# Explicit architecture assertion: MPC must not publish the actuator command;
# TrajectoryExecutionManager must be the sole planner-side publisher.
rosnode info /velocity_qp_mpc_waypoint_node > "${OUT}/mpc_node_info.txt" 2>&1 || true
rosnode info /trajectory_execution_manager_node > "${OUT}/tracker_node_info.txt" 2>&1 || true
if grep -Fq "* ${ACTUATOR_TOPIC} " "${OUT}/mpc_node_info.txt"; then
  echo "[ERROR] MPC still publishes actuator command in planner mode"
  cat "${OUT}/mpc_node_info.txt"
  exit 1
fi
if ! grep -Fq "* ${MPC_UNUSED_COMMAND_TOPIC} " "${OUT}/mpc_node_info.txt"; then
  echo "[ERROR] MPC unused/debug command topic not observed"
  cat "${OUT}/mpc_node_info.txt"
  exit 1
fi
if ! grep -Fq "* ${ACTUATOR_TOPIC} " "${OUT}/tracker_node_info.txt"; then
  echo "[ERROR] low-level tracker does not own actuator command topic"
  cat "${OUT}/tracker_node_info.txt"
  exit 1
fi

echo "[ARCH] CAREPlanner MPC -> ${OPTIMIZED_TOPIC} -> TrajectoryExecutionManager -> ${ACTUATOR_TOPIC}"

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" \
    > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}

record_topic /care_planner/trajectory_risk/vbc_summary "${OUT}/selector_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic /care_planner/execution/predicted_vbc_recovery_summary "${OUT}/predicted_vbc_recovery_guard.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 2 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then
    RELEASED=1; break
  fi
  sleep 0.1
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 160 "${LOG}/controlled.log" || true
  tail -n 100 "${LOG}/low_level_tracker.log" || true
  exit 1
fi

echo "[RUN] ${CASE_ID}: C4.3 planner/controller split for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
sleep 0.2
for pid in "${REC_PIDS[@]:-}"; do kill_group "${pid}"; done
REC_PIDS=()

python3 scripts/summarize_phase_c4_3_planner_split.py \
  "${CASE_ID}" "${OUT}" "${LOG}/controlled.log"

echo "[RESULT]  ${OUT}/c4_3_planner_split_summary.json"
echo "[MPC]     ${OUT}/mpc_summary.csv"
echo "[TRACKER] ${OUT}/low_level_reference_state.csv"
echo "[ARCH]    ${OUT}/mpc_node_info.txt and ${OUT}/tracker_node_info.txt"
