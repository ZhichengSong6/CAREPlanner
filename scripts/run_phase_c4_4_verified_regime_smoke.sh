#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
CONFIG_FILE="${CONFIG_FILE:-${REPO}/src/egocentric_arm_planner/config/planner_phase1_c4_3_planner.yaml}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-8.0}"
# Visualization controls. Keep formal benchmarks headless by default; set these
# to true for an otherwise identical Gazebo + RViz execution.
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map.yaml}"
TOF_FUSION_ENABLED="${TOF_FUSION_ENABLED:-false}"
TRAJECTORY_RISK_REFRESH_BODY_PRIOR_BEFORE_QUERY="${TRAJECTORY_RISK_REFRESH_BODY_PRIOR_BEFORE_QUERY:-false}"
EXECUTION_GCDF_AUDIT_ENABLED="${EXECUTION_GCDF_AUDIT_ENABLED:-false}"
EXECUTION_GCDF_WARNING_MARGIN="${EXECUTION_GCDF_WARNING_MARGIN:-0.05}"
EXECUTION_GCDF_HARD_MARGIN="${EXECUTION_GCDF_HARD_MARGIN:-0.0}"
EXECUTION_GCDF_STALE_TIMEOUT_S="${EXECUTION_GCDF_STALE_TIMEOUT_S:-0.35}"
GCDF_BODY_INFLATION_M="${GCDF_BODY_INFLATION_M:-0.0}"
INITIAL_GATE_MAX_TRIES="${INITIAL_GATE_MAX_TRIES:-400}"
INITIAL_GATE_ECHO_TIMEOUT="${INITIAL_GATE_ECHO_TIMEOUT:-1.0}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
NCDF_DEVICE="${NCDF_DEVICE:-cpu}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.20}"
CANDIDATE_UNSAFE_REQUIRED="${CANDIDATE_UNSAFE_REQUIRED:-2}"
EXECUTION_UNSAFE_REQUIRED="${EXECUTION_UNSAFE_REQUIRED:-2}"
PROBE_SAFE_COMMITS="${PROBE_SAFE_COMMITS:-3}"
# Preserve C4.4 behavior by default. C4.5/C4.6 wrappers override this explicitly.
REGION_SCHEDULE_MODE="${REGION_SCHEDULE_MODE:-shared_persistent}"
TRAJECTORY_RISK_INPUT_TOPIC="${TRAJECTORY_RISK_INPUT_TOPIC:-/care_planner/task_trajectory}"
FORBIDDEN_SPACE_PAIR_PUBLISH_ENABLED="${FORBIDDEN_SPACE_PAIR_PUBLISH_ENABLED:-false}"
FORBIDDEN_SPACE_PAIR_TOPIC="${FORBIDDEN_SPACE_PAIR_TOPIC:-/care_planner/trajectory_risk/body_sweep_anchors}"
FORBIDDEN_SPACE_CONFIDENCE_THRESHOLD="${FORBIDDEN_SPACE_CONFIDENCE_THRESHOLD:-0.50}"
CDF_SHADOW_ENABLED="${CDF_SHADOW_ENABLED:-false}"
CDF_SHADOW_SAFETY_MARGIN="${CDF_SHADOW_SAFETY_MARGIN:-0.0}"
CDF_SHADOW_TRUST_REGION_Q_INF="${CDF_SHADOW_TRUST_REGION_Q_INF:-0.20}"
CDF_SHADOW_HORIZON_STEPS="${CDF_SHADOW_HORIZON_STEPS:-20}"
CDF_SHADOW_SNAPSHOT_TIMEOUT="${CDF_SHADOW_SNAPSHOT_TIMEOUT:-0.75}"
CDF_SHADOW_CONSTRAINT_BATCH_TOPIC="${CDF_SHADOW_CONSTRAINT_BATCH_TOPIC:-/care_planner/collision_cdf/constraint_batch}"
CDF_SHADOW_PREDICTION_TOPIC="${CDF_SHADOW_PREDICTION_TOPIC:-/care_planner/mpc/cdf_shadow_predicted_trajectory}"
CDF_SHADOW_SUMMARY_TOPIC="${CDF_SHADOW_SUMMARY_TOPIC:-/care_planner/mpc/cdf_shadow_summary}"
CDF_SELECTOR_ENABLED="${CDF_SELECTOR_ENABLED:-false}"
CDF_SELECTOR_GPU_SOCKET="${CDF_SELECTOR_GPU_SOCKET:-/tmp/care_collision_cdf_gpu.sock}"
CDF_SELECTOR_OUTPUT_JSONL="${CDF_SELECTOR_OUTPUT_JSONL:-/tmp/c5_3a_cpp_selector_gpu.jsonl}"
CDF_SELECTOR_RATE="${CDF_SELECTOR_RATE:-20.0}"
CDF_SELECTOR_MAP_RESOLUTION="${CDF_SELECTOR_MAP_RESOLUTION:-0.05}"
CDF_SELECTOR_PROXIMITY_MARGIN="${CDF_SELECTOR_PROXIMITY_MARGIN:-0.075}"
CDF_SELECTOR_MAX_PAIRS_PER_STEP="${CDF_SELECTOR_MAX_PAIRS_PER_STEP:-250}"
CDF_SELECTOR_SIGNED_ZERO_BAND="${CDF_SELECTOR_SIGNED_ZERO_BAND:-0.05}"
CDF_SHADOW_VBC_AUDIT_ENABLED="${CDF_SHADOW_VBC_AUDIT_ENABLED:-false}"
CDF_SHADOW_VBC_SUMMARY_TOPIC="${CDF_SHADOW_VBC_SUMMARY_TOPIC:-/care_planner/cdf_shadow_vbc/summary}"

# C5.4 event-triggered local trajectory optimizer. Default false preserves
# frozen C4.x behavior.
USE_LOCAL_SPARSE_SCP="${USE_LOCAL_SPARSE_SCP:-false}"
LOCAL_SCP_GPU_SOCKET="${LOCAL_SCP_GPU_SOCKET:-/tmp/care_collision_cdf_gpu_c5_4.sock}"
LOCAL_SCP_SELECTOR_JSONL="${LOCAL_SCP_SELECTOR_JSONL:-/tmp/c5_4_local_scp_selector.jsonl}"
LOCAL_SCP_PROXIMITY_MARGIN="${LOCAL_SCP_PROXIMITY_MARGIN:-0.025}"
VBC_SWEPT_VOLUME_MARGIN_M="${VBC_SWEPT_VOLUME_MARGIN_M:-0.0}"
LOCAL_SCP_CANDIDATE_TOPIC="${LOCAL_SCP_CANDIDATE_TOPIC:-/care_planner/local_planner/candidate_trajectory}"
LOCAL_SCP_SUMMARY_TOPIC="${LOCAL_SCP_SUMMARY_TOPIC:-/care_planner/local_planner/summary}"
LOCAL_SCP_REPLAN_TOPIC="${LOCAL_SCP_REPLAN_TOPIC:-/care_planner/local_planner/replan_request}"

# C5.8/C5.9 commit-pipeline capabilities. Defaults preserve legacy C4.x.
FINAL_EXECUTABLE_GCDF_ENABLED="${FINAL_EXECUTABLE_GCDF_ENABLED:-false}"
COMMITTED_CONTINUATION_ENABLED="${COMMITTED_CONTINUATION_ENABLED:-true}"
EXECUTION_AUDIT_STREAM_ENABLED="${EXECUTION_AUDIT_STREAM_ENABLED:-false}"
EXECUTION_VBC_TRAJECTORY_TOPIC="${EXECUTION_VBC_TRAJECTORY_TOPIC:-/care_planner/committed_trajectory}"
PROBE_SINGLE_FLIGHT_ENABLED="${PROBE_SINGLE_FLIGHT_ENABLED:-false}"
PROBE_SINGLE_FLIGHT_TOPIC="${PROBE_SINGLE_FLIGHT_TOPIC:-/care_planner/local_planner/candidate_trajectory_single_flight}"
PROBE_SINGLE_FLIGHT_SUMMARY_TOPIC="${PROBE_SINGLE_FLIGHT_SUMMARY_TOPIC:-/care_planner/execution/probe_single_flight_summary}"

# C4.8 compatibility. C4.9 exports the historical C4_REPAIR_* variables;
# C5.4 leaves this disabled and verifies the complete optimized trajectory.
REPAIR_PREFIX_VERIFY="${REPAIR_PREFIX_VERIFY:-${C4_REPAIR_PREFIX_VERIFY:-0}}"
REPAIR_PREFIX_S="${REPAIR_PREFIX_S:-${C4_REPAIR_PREFIX_S:-0.15}}"
PROBE_PREFIX_S="${PROBE_PREFIX_S:-0.15}"
SMOOTH_HANDOFF_ENABLED="${SMOOTH_HANDOFF_ENABLED:-false}"
REPAIR_BRAKE_DT_S="${REPAIR_BRAKE_DT_S:-${C4_REPAIR_BRAKE_DT_S:-0.05}}"
REPAIR_HOLD_S="${REPAIR_HOLD_S:-${C4_REPAIR_HOLD_S:-0.10}}"
CYCLE_RECOVERY_ENABLED="${CYCLE_RECOVERY_ENABLED:-false}"
ADAPTIVE_REFINEMENT_ENABLED="${ADAPTIVE_REFINEMENT_ENABLED:-false}"
VBC_GATED_FRONTIER_STEP_ENABLED="${VBC_GATED_FRONTIER_STEP_ENABLED:-false}"
LOCAL_SENSING_ACTION_SEARCH_ENABLED="${LOCAL_SENSING_ACTION_SEARCH_ENABLED:-false}"
LOCAL_SENSING_ACTION_STEP_INF="${LOCAL_SENSING_ACTION_STEP_INF:-0.04}"
LOCAL_SENSING_ACTION_MAX_TRIALS="${LOCAL_SENSING_ACTION_MAX_TRIALS:-4}"
LOCAL_SENSING_ACTION_RECOMPUTE_Q_INF="${LOCAL_SENSING_ACTION_RECOMPUTE_Q_INF:-0.03}"
ENABLE_ORACLE_DIAGNOSTICS="${ENABLE_ORACLE_DIAGNOSTICS:-false}"

# Optional online task-success stop. Defaults off here so historical C4/C5
# diagnostics retain their fixed-duration semantics. Phase-D enables it.
EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-false}"
GOAL_POSITION_TOLERANCE_M="${GOAL_POSITION_TOLERANCE_M:-0.02}"
GOAL_ORIENTATION_TOLERANCE_RAD="${GOAL_ORIENTATION_TOLERANCE_RAD:-0.20}"
GOAL_SUCCESS_HOLD_S="${GOAL_SUCCESS_HOLD_S:-0.10}"
GOAL_SETTLE_VELOCITY_INF_RAD_S="${GOAL_SETTLE_VELOCITY_INF_RAD_S:-0.05}"
GOAL_SETTLE_TIMEOUT_S="${GOAL_SETTLE_TIMEOUT_S:-1.0}"
GOAL_POST_SUCCESS_RECORD_S="${GOAL_POST_SUCCESS_RECORD_S:-0.0}"

OUT="${OUT:-${REPO}/outputs/phase_c4_4_verified_regime_smoke/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_4_verified_regime_smoke/${CASE_ID}}"

if [ "${USE_LOCAL_SPARSE_SCP}" = "true" ]; then
  RAW_PLANNER_TOPIC="${LOCAL_SCP_CANDIDATE_TOPIC}"
  TRAJECTORY_RISK_INPUT_TOPIC="${LOCAL_SCP_CANDIDATE_TOPIC}"
else
  RAW_PLANNER_TOPIC="/care_planner/mpc/predicted_trajectory"
fi
RAW_MPC_TOPIC="${RAW_PLANNER_TOPIC}"
if [ "${PROBE_SINGLE_FLIGHT_ENABLED}" = "true" ]; then
  COMMIT_PIPELINE_CANDIDATE_TOPIC="${PROBE_SINGLE_FLIGHT_TOPIC}"
else
  COMMIT_PIPELINE_CANDIDATE_TOPIC="${RAW_PLANNER_TOPIC}"
fi

if [ "${PROBE_SINGLE_FLIGHT_ENABLED}" = "true" ] && \
   [ "${COMMIT_PIPELINE_CANDIDATE_TOPIC}" = "${RAW_PLANNER_TOPIC}" ]; then
  echo "[ERROR] PROBE single-flight enabled but commit pipeline still bypasses gate" >&2
  exit 2
fi

VERIFY_TOPIC="/care_planner/optimized_trajectory"
COMMITTED_TOPIC="/care_planner/committed_trajectory"
CANDIDATE_VBC_TOPIC="/care_planner/candidate_vbc/summary"
EXECUTION_VBC_TOPIC="/care_planner/execution_vbc/summary"
REGIME_TOPIC="/care_planner/c4_4/regime_summary"
TRACKER_DESIRED_TOPIC="/care_planner/execution/tracker_velocity_desired"
ACTUATOR_TOPIC="/care_arm/arm_group_velocity_controller/command"
SCHEDULE_TOPIC="/care_planner/active_sensing/visibility_waypoint_schedule"
SCHEDULE_SUMMARY_TOPIC="/care_planner/active_sensing/visibility_waypoint_schedule_summary"

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

if ! GOAL_LINE="$(python3 - "${CASE_FILE}" "${CASE_ID}" <<'PY'
import json,sys
p,cid=sys.argv[1:]
d=json.load(open(p)); c=next((x for x in d['cases'] if x['case_id']==cid),None)
if c is None:
    raise SystemExit('unknown case_id: '+cid)
print(*c['goal_position'],*c['goal_orientation'])
PY
)"; then
  echo "[ERROR] failed to resolve controlled-trial CASE_ID=${CASE_ID}" >&2
  exit 2
fi
read -r GX GY GZ GQX GQY GQZ GQW <<< "${GOAL_LINE}"
if [[ -z "${GX:-}" || -z "${GY:-}" || -z "${GZ:-}" || -z "${GQW:-}" ]]; then
  echo "[ERROR] incomplete goal resolved for CASE_ID=${CASE_ID}: ${GOAL_LINE}" >&2
  exit 2
fi

GAZEBO_PID=""; GEN_PID=""; CONTROL_PID=""; TRACKER_PID=""
REC_PIDS=()
kill_group() {
  local pid="${1:-}"; [ -z "${pid}" ] && return 0
  if kill -0 "${pid}" 2>/dev/null; then kill -INT -- "-${pid}" 2>/dev/null || true; sleep 0.25; fi
  if kill -0 "${pid}" 2>/dev/null; then kill -TERM -- "-${pid}" 2>/dev/null || true; sleep 0.25; fi
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
  for name in gzclient gzserver rviz rosmaster roscore roslaunch; do pkill -TERM -x "${name}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

echo "[VIS] gazebo_gui=${GAZEBO_GUI} use_rviz=${USE_RVIZ} world_file=${WORLD_FILE}"
setsid roslaunch arm_description gazebo_velocity_control.launch \
  world_file:="${WORLD_FILE}" \
  gazebo_gui:="${GAZEBO_GUI}" use_rviz:="${USE_RVIZ}" > "${LOG}/gazebo.log" 2>&1 &
GAZEBO_PID=$!

echo "[WAIT] Gazebo robot/controller readiness: waiting for one /care_arm/joint_states message"
if ! python3 - <<'PY'
import sys
import rospy
from sensor_msgs.msg import JointState

topic = "/care_arm/joint_states"
rospy.init_node(
    "careplanner_gazebo_joint_state_ready",
    anonymous=True,
    disable_signals=True,
)
try:
    msg = rospy.wait_for_message(topic, JointState, timeout=20.0)
except rospy.ROSException as exc:
    print(f"[ERROR] timed out waiting for {topic}: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(
    "[READY] Gazebo robot spawned and joint state is live: "
    f"names={len(msg.name)} positions={len(msg.position)}"
)
PY
then
  echo "[ERROR] Gazebo robot/controller readiness failed"
  echo "[DEBUG] joint-state-like topics:"
  rostopic list 2>/dev/null | grep -E 'joint|state' || true
  echo "[DEBUG] expected topic info:"
  rostopic info /care_arm/joint_states 2>/dev/null || true
  echo "[DEBUG] controller states:"
  rosservice call /care_arm/controller_manager/list_controllers 2>/dev/null || true
  echo "[DEBUG] ROS nodes:"
  rosnode list 2>/dev/null || true
  echo "[DEBUG] gazebo log: ${LOG}/gazebo.log"
  tail -n 260 "${LOG}/gazebo.log" 2>/dev/null || true
  exit 1
fi

setsid roslaunch egocentric_arm_planner c4_3_low_level_tracker.launch \
  config_file:="${CONFIG_FILE}" input_trajectory:="${COMMITTED_TOPIC}" \
  output_velocity_command:="${TRACKER_DESIRED_TOPIC}" \
  use_acceleration_limiter:=true \
  rate_limiter_output_velocity_command:="${ACTUATOR_TOPIC}" \
  > "${LOG}/low_level_tracker.log" 2>&1 &
TRACKER_PID=$!

setsid bash -lc "source '${CONDA_SH}'; conda activate '${NCDF_ENV}'; cd '${REPO}'; source devel/setup.bash; exec python -u src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py _device:=${NCDF_DEVICE} _rate:=50.0 _enable_oracle_diagnostics:=${ENABLE_ORACLE_DIAGNOSTICS} _region_schedule_mode:='${REGION_SCHEDULE_MODE}' _predicted_trajectory_topic:='${VERIFY_TOPIC}' _safety_margin_s:=${SAFETY_MARGIN} _predicted_trajectory_timeout:=${PREDICTION_TIMEOUT} _target_cell_resolution:=0.05 _projection_iters:=10 _projection_damping:=0.5 _projection_epsilon_f:=0.03 _projection_max_step_norm:=0.25 _root_refine_iters:=12 _root_tolerance_f:=0.002 _ascent_steps:=1 _ascent_step_size:=0.05 _ascent_max_step_norm:=0.25 _adaptive_refinement_enabled:=${ADAPTIVE_REFINEMENT_ENABLED} _vbc_gated_frontier_step_enabled:=${VBC_GATED_FRONTIER_STEP_ENABLED} _local_sensing_action_search_enabled:=${LOCAL_SENSING_ACTION_SEARCH_ENABLED} _local_sensing_action_step_inf:=${LOCAL_SENSING_ACTION_STEP_INF} _local_sensing_action_max_trials:=${LOCAL_SENSING_ACTION_MAX_TRIALS} _local_sensing_action_recompute_q_inf:=${LOCAL_SENSING_ACTION_RECOMPUTE_Q_INF} _output_root:='${OUT}/projector_traces'" \
  > "${LOG}/waypoint_generator.log" 2>&1 &
GEN_PID=$!

READY=0
for _ in $(seq 1 120); do
  if grep -q "ONLINE WARMUP READY" "${LOG}/waypoint_generator.log" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "${GEN_PID}" 2>/dev/null; then break; fi
  sleep 0.25
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] online waypoint generator warm-up not ready"
  tail -n 160 "${LOG}/waypoint_generator.log" || true
  exit 1
fi

echo "[MODE] region_schedule_mode=${REGION_SCHEDULE_MODE}"

setsid roslaunch egocentric_arm_planner phaseC4_4_verified_regime_planner.launch \
  config_file:="${CONFIG_FILE}" \
  confidence_map_config_file:="${CONFIDENCE_MAP_CONFIG_FILE}" \
  tof_fusion_enabled:="${TOF_FUSION_ENABLED}" \
  waypoint_weight:="${CARE_WEIGHT}" \
  vbc_min_margin_s:="${SAFETY_MARGIN}" \
  selector_predicted_trajectory_timeout:="${PREDICTION_TIMEOUT}" \
  trajectory_risk_input_topic:="${TRAJECTORY_RISK_INPUT_TOPIC}" \
  trajectory_risk_refresh_body_prior_before_query:="${TRAJECTORY_RISK_REFRESH_BODY_PRIOR_BEFORE_QUERY}" \
  forbidden_space_pair_publish_enabled:="${FORBIDDEN_SPACE_PAIR_PUBLISH_ENABLED}" \
  forbidden_space_pair_topic:="${FORBIDDEN_SPACE_PAIR_TOPIC}" \
  forbidden_space_confidence_threshold:="${FORBIDDEN_SPACE_CONFIDENCE_THRESHOLD}" \
  cdf_shadow_enabled:="${CDF_SHADOW_ENABLED}" \
  cdf_shadow_safety_margin:="${CDF_SHADOW_SAFETY_MARGIN}" \
  cdf_shadow_trust_region_q_inf:="${CDF_SHADOW_TRUST_REGION_Q_INF}" \
  cdf_shadow_constraint_horizon_steps:="${CDF_SHADOW_HORIZON_STEPS}" \
  cdf_shadow_snapshot_timeout:="${CDF_SHADOW_SNAPSHOT_TIMEOUT}" \
  cdf_shadow_constraint_batch_topic:="${CDF_SHADOW_CONSTRAINT_BATCH_TOPIC}" \
  cdf_shadow_prediction_topic:="${CDF_SHADOW_PREDICTION_TOPIC}" \
  cdf_shadow_summary_topic:="${CDF_SHADOW_SUMMARY_TOPIC}" \
  cdf_selector_enabled:="${CDF_SELECTOR_ENABLED}" \
  cdf_selector_gpu_socket:="${CDF_SELECTOR_GPU_SOCKET}" \
  cdf_selector_output_jsonl:="${CDF_SELECTOR_OUTPUT_JSONL}" \
  cdf_selector_rate:="${CDF_SELECTOR_RATE}" \
  cdf_selector_map_resolution:="${CDF_SELECTOR_MAP_RESOLUTION}" \
  cdf_selector_proximity_margin:="${CDF_SELECTOR_PROXIMITY_MARGIN}" \
  cdf_selector_max_pairs_per_step:="${CDF_SELECTOR_MAX_PAIRS_PER_STEP}" \
  cdf_selector_signed_zero_band:="${CDF_SELECTOR_SIGNED_ZERO_BAND}" \
  cdf_shadow_vbc_audit_enabled:="${CDF_SHADOW_VBC_AUDIT_ENABLED}" \
  cdf_shadow_vbc_summary_topic:="${CDF_SHADOW_VBC_SUMMARY_TOPIC}" \
  use_local_sparse_scp:="${USE_LOCAL_SPARSE_SCP}" \
  raw_mpc_trajectory_topic:="${RAW_PLANNER_TOPIC}" \
  final_executable_gcdf_enabled:="${FINAL_EXECUTABLE_GCDF_ENABLED}" \
  committed_continuation_enabled:="${COMMITTED_CONTINUATION_ENABLED}" \
  execution_audit_stream_enabled:="${EXECUTION_AUDIT_STREAM_ENABLED}" \
  execution_vbc_trajectory_topic:="${EXECUTION_VBC_TRAJECTORY_TOPIC}" \
  execution_gcdf_audit_enabled:="${EXECUTION_GCDF_AUDIT_ENABLED}" \
  execution_gcdf_warning_margin:="${EXECUTION_GCDF_WARNING_MARGIN}" \
  execution_gcdf_hard_margin:="${EXECUTION_GCDF_HARD_MARGIN}" \
  execution_gcdf_stale_timeout_s:="${EXECUTION_GCDF_STALE_TIMEOUT_S}" \
  gcdf_body_inflation_m:="${GCDF_BODY_INFLATION_M}" \
  probe_single_flight_enabled:="${PROBE_SINGLE_FLIGHT_ENABLED}" \
  commit_pipeline_candidate_topic:="${COMMIT_PIPELINE_CANDIDATE_TOPIC}" \
  probe_single_flight_summary_topic:="${PROBE_SINGLE_FLIGHT_SUMMARY_TOPIC}" \
  local_scp_candidate_trajectory_topic:="${LOCAL_SCP_CANDIDATE_TOPIC}" \
  local_scp_summary_topic:="${LOCAL_SCP_SUMMARY_TOPIC}" \
  local_scp_replan_request_topic:="${LOCAL_SCP_REPLAN_TOPIC}" \
  local_scp_gpu_socket:="${LOCAL_SCP_GPU_SOCKET}" \
  local_scp_selector_jsonl:="${LOCAL_SCP_SELECTOR_JSONL}" \
  local_scp_proximity_margin:="${LOCAL_SCP_PROXIMITY_MARGIN}" \
  vbc_swept_volume_margin_m:="${VBC_SWEPT_VOLUME_MARGIN_M}" \
  repair_prefix_verification_enabled:="${REPAIR_PREFIX_VERIFY}" \
  repair_execution_prefix_s:="${REPAIR_PREFIX_S}" \
  probe_execution_prefix_s:="${PROBE_PREFIX_S}" \
  smooth_handoff_enabled:="${SMOOTH_HANDOFF_ENABLED}" \
  repair_brake_dt_s:="${REPAIR_BRAKE_DT_S}" \
  repair_hold_s:="${REPAIR_HOLD_S}" \
  cycle_recovery_enabled:="${CYCLE_RECOVERY_ENABLED}" \
  candidate_unsafe_required:="${CANDIDATE_UNSAFE_REQUIRED}" \
  execution_unsafe_required:="${EXECUTION_UNSAFE_REQUIRED}" \
  probe_safe_commits_required:="${PROBE_SAFE_COMMITS}" \
  trial_label:="${CASE_ID}_c4_4_verified_regime" log_output_root:="${OUT}" \
  goal_x:="${GX}" goal_y:="${GY}" goal_z:="${GZ}" \
  goal_qx:="${GQX}" goal_qy:="${GQY}" goal_qz:="${GQZ}" goal_qw:="${GQW}" \
  > "${LOG}/controlled.log" 2>&1 &
CONTROL_PID=$!

READY=0
for _ in $(seq 1 400); do
  NODES="$(rosnode list 2>/dev/null || true)"
  COMMON_READY=0
  BACKEND_READY=0
  CDF_READY=1

  if echo "${NODES}" | grep -q '^/trajectory_execution_manager_node$' && \
     echo "${NODES}" | grep -q '^/joint_velocity_rate_limiter$' && \
     echo "${NODES}" | grep -q '^/optimized_trajectory_continuity$' && \
     echo "${NODES}" | grep -q '^/trajectory_vbc_selector_node$' && \
     echo "${NODES}" | grep -q '^/execution_vbc_audit/trajectory_vbc_selector_node$' && \
     echo "${NODES}" | grep -q '^/c4_4_verified_regime_manager$' && \
     echo "${NODES}" | grep -q '^/phase_b2_controlled_trial$' && \
     echo "${NODES}" | grep -q '^/vbc_execution_reference_gate$'; then
    COMMON_READY=1
  fi

  if [ "${USE_LOCAL_SPARSE_SCP}" = "true" ]; then
    if echo "${NODES}" | grep -q '^/local_sparse_scp_planner_node$' && \
       echo "${NODES}" | grep -q '^/local_scp_pair_export/trajectory_risk_node$' && \
       echo "${NODES}" | grep -q '^/c5_4_local_scp_cdf_selector$'; then
      BACKEND_READY=1
    fi
  else
    if echo "${NODES}" | grep -q '^/velocity_qp_mpc_waypoint_node$'; then
      BACKEND_READY=1
    fi
  fi

  if [ "${CDF_SELECTOR_ENABLED}" = "true" ]; then
    if ! echo "${NODES}" | grep -q '^/c5_3a_cpp_forbidden_voxel_gpu_shadow$'; then
      CDF_READY=0
    fi
  fi

  if [ "${CDF_SHADOW_VBC_AUDIT_ENABLED}" = "true" ]; then
    if ! echo "${NODES}" | grep -q '^/cdf_shadow_vbc/trajectory_vbc_selector_node$'; then
      CDF_READY=0
    fi
  fi

  if [ "${FINAL_EXECUTABLE_GCDF_ENABLED}" = "true" ]; then
    if ! echo "${NODES}" | grep -q '^/final_executable_gcdf_pair_export/trajectory_risk_node$'; then
      CDF_READY=0
    fi
  fi

  if [ "${PROBE_SINGLE_FLIGHT_ENABLED}" = "true" ]; then
    if ! echo "${NODES}" | grep -q '^/probe_single_flight_gate$'; then
      CDF_READY=0
    fi
  fi

  if [ "${TOF_FUSION_ENABLED}" = "true" ]; then
    if ! echo "${NODES}" | grep -q '^/tof_fusion_self_filter$'; then
      CDF_READY=0
    fi
  fi

  if [ "${EXECUTION_GCDF_AUDIT_ENABLED}" = "true" ]; then
    if ! echo "${NODES}" | grep -q '^/execution_gcdf_audit/measured_state_trajectory$' || \
       ! echo "${NODES}" | grep -q '^/execution_gcdf_audit/trajectory_risk_node$' || \
       ! echo "${NODES}" | grep -q '^/execution_gcdf_audit/safety_monitor$'; then
      CDF_READY=0
    fi
  fi

  if [ "${COMMON_READY}" = "1" ] &&
     [ "${BACKEND_READY}" = "1" ] &&
     [ "${CDF_READY}" = "1" ]; then
    READY=1
    break
  fi
  sleep 0.1
done

if [ "${READY}" != "1" ]; then
  echo "[ERROR] required planner/controller nodes did not all start"
  echo "[DEBUG] USE_LOCAL_SPARSE_SCP=${USE_LOCAL_SPARSE_SCP}"
  echo "[DEBUG] CDF_SELECTOR_ENABLED=${CDF_SELECTOR_ENABLED}"
  echo "[DEBUG] CDF_SHADOW_VBC_AUDIT_ENABLED=${CDF_SHADOW_VBC_AUDIT_ENABLED}"
  rosnode list 2>/dev/null || true
  tail -n 320 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

expect_rosparam_bool() {
  local name="$1"
  local expected="$2"
  local actual
  actual="$(rosparam get "${name}" 2>/dev/null || true)"
  actual="$(echo "${actual}" | tr '[:upper:]' '[:lower:]' | xargs)"
  if [ "${actual}" != "${expected}" ]; then
    echo "[ERROR] runtime semantic mismatch: ${name}=${actual:-missing}, expected=${expected}"
    tail -n 220 "${LOG}/controlled.log" || true
    exit 1
  fi
}

expect_rosparam_bool /optimized_trajectory_continuity/final_gcdf_enabled "${FINAL_EXECUTABLE_GCDF_ENABLED}"
expect_rosparam_bool /optimized_trajectory_continuity/continuation_enabled "${COMMITTED_CONTINUATION_ENABLED}"
expect_rosparam_bool /optimized_trajectory_continuity/execution_audit_enabled "${EXECUTION_AUDIT_STREAM_ENABLED}"
expect_rosparam_bool /optimized_trajectory_continuity/cycle_recovery_enabled "${CYCLE_RECOVERY_ENABLED}"

if [ "${PROBE_SINGLE_FLIGHT_ENABLED}" = "true" ]; then
  if ! rosnode list | grep -q '^/probe_single_flight_gate$'; then
    echo "[ERROR] C5.9 probe_single_flight requested but node is missing"
    exit 1
  fi
fi

echo "[RUNTIME] final_gcdf=${FINAL_EXECUTABLE_GCDF_ENABLED} continuation=${COMMITTED_CONTINUATION_ENABLED} execution_audit=${EXECUTION_AUDIT_STREAM_ENABLED} probe_single_flight=${PROBE_SINGLE_FLIGHT_ENABLED}"
echo "[PHASE E] tof_fusion=${TOF_FUSION_ENABLED} confidence_map_config=${CONFIDENCE_MAP_CONFIG_FILE}"
echo "[BODY PRIOR A/B] main_trajectory_risk_refresh=${TRAJECTORY_RISK_REFRESH_BODY_PRIOR_BEFORE_QUERY}; local/final/execution exporters remain false"
echo "[PHASE E5] execution_gcdf=${EXECUTION_GCDF_AUDIT_ENABLED} warn=${EXECUTION_GCDF_WARNING_MARGIN} hard=${EXECUTION_GCDF_HARD_MARGIN} stale=${EXECUTION_GCDF_STALE_TIMEOUT_S} body_inflation=${GCDF_BODY_INFLATION_M}"
echo "[MARGIN SPLIT] VBC swept margin=${VBC_SWEPT_VOLUME_MARGIN_M} m; GCDF proximity margin=${LOCAL_SCP_PROXIMITY_MARGIN} m"

RUNTIME_BRANCH="$(git branch --show-current)"
RUNTIME_HEAD="$(git rev-parse HEAD)"
RUNTIME_FINAL_GCDF="$(rosparam get /optimized_trajectory_continuity/final_gcdf_enabled)"
RUNTIME_CONTINUATION="$(rosparam get /optimized_trajectory_continuity/continuation_enabled)"
RUNTIME_EXEC_AUDIT="$(rosparam get /optimized_trajectory_continuity/execution_audit_enabled)"
RUNTIME_PROBE_NODE_COUNT="$(rosnode list | grep -c '^/probe_single_flight_gate$' || true)"

cat > "${OUT}/runtime_semantics.txt" <<EOF
branch=${RUNTIME_BRANCH}
head=${RUNTIME_HEAD}
case_id=${CASE_ID}
run_id=${RUN_ID:-${CASE_ID}}
gazebo_gui=${GAZEBO_GUI}
use_rviz=${USE_RVIZ}
world_file=${WORLD_FILE}
confidence_map_config_file=${CONFIDENCE_MAP_CONFIG_FILE}
tof_fusion_enabled=${TOF_FUSION_ENABLED}
main_trajectory_risk_refresh_body_prior_before_query=${TRAJECTORY_RISK_REFRESH_BODY_PRIOR_BEFORE_QUERY}
execution_gcdf_audit_enabled=${EXECUTION_GCDF_AUDIT_ENABLED}
execution_gcdf_warning_margin=${EXECUTION_GCDF_WARNING_MARGIN}
execution_gcdf_hard_margin=${EXECUTION_GCDF_HARD_MARGIN}
execution_gcdf_stale_timeout_s=${EXECUTION_GCDF_STALE_TIMEOUT_S}
gcdf_body_inflation_m=${GCDF_BODY_INFLATION_M}
use_local_sparse_scp=${USE_LOCAL_SPARSE_SCP}
local_scp_proximity_margin_m=${LOCAL_SCP_PROXIMITY_MARGIN}
vbc_swept_volume_margin_m=${VBC_SWEPT_VOLUME_MARGIN_M}
raw_planner_topic=${RAW_PLANNER_TOPIC}
commit_pipeline_candidate_topic=${COMMIT_PIPELINE_CANDIDATE_TOPIC}
final_gcdf_enabled=${RUNTIME_FINAL_GCDF}
continuation_enabled=${RUNTIME_CONTINUATION}
execution_audit_enabled=${RUNTIME_EXEC_AUDIT}
execution_vbc_trajectory_topic=${EXECUTION_VBC_TRAJECTORY_TOPIC}
probe_single_flight_enabled=${PROBE_SINGLE_FLIGHT_ENABLED}
probe_single_flight_node=${RUNTIME_PROBE_NODE_COUNT}
probe_single_flight_input_topic=${RAW_PLANNER_TOPIC}
probe_single_flight_output_topic=${COMMIT_PIPELINE_CANDIDATE_TOPIC}
continuity_input_topic=${COMMIT_PIPELINE_CANDIDATE_TOPIC}
probe_single_flight_wiring_distinct=$([ "${RAW_PLANNER_TOPIC}" != "${COMMIT_PIPELINE_CANDIDATE_TOPIC}" ] && echo 1 || echo 0)
probe_repair_requires_visibility_obligation=true
adaptive_refinement_enabled=${ADAPTIVE_REFINEMENT_ENABLED}
adaptive_refinement_policy=coarse_default_refine_learned_incompatible_dependency_cycle_once
local_sensing_action_search_enabled=${LOCAL_SENSING_ACTION_SEARCH_ENABLED}
local_sensing_action_step_inf=${LOCAL_SENSING_ACTION_STEP_INF}
local_sensing_action_max_trials=${LOCAL_SENSING_ACTION_MAX_TRIALS}
local_sensing_action_recompute_q_inf=${LOCAL_SENSING_ACTION_RECOMPUTE_Q_INF}
probe_solver_failure_uses_blocker_rediscovery=true
probe_vbc_unsafe_uses_blocker_rediscovery=true
probe_final_gcdf_unsafe_uses_direct_recovery_evidence=true
phase_e4_occupied_gcdf_never_creates_visibility_obligation=true
final_gcdf_recovery_trajectory_topic=/care_planner/final_gcdf/recovery_trajectory
final_gcdf_recovery_event_topic=/care_planner/final_gcdf/recovery_visibility_event
EOF

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /care_planner/c4_4/probe_active "${OUT}/probe_active.csv"
record_topic /care_planner/local_planner/task_infeasible "${OUT}/task_infeasible.csv"
record_topic /care_planner/local_planner/task_obstacle_blocked "${OUT}/task_obstacle_blocked.csv"
record_topic /care_planner/local_planner/task_uncertified "${OUT}/task_uncertified.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /care_planner/active_sensing/visibility_acquisition_summary "${OUT}/visibility_acquisition_summary.csv"
record_topic /care_planner/active_sensing/visibility_acquisition_complete "${OUT}/visibility_acquisition_complete.csv"
if [ "${TOF_FUSION_ENABLED}" = "true" ]; then
  record_topic /care_planner/perception/tof_fusion_summary "${OUT}/tof_fusion_summary.csv"
  record_topic /care_planner/confidence_map/e3_summary "${OUT}/e3_summary.csv"
fi
if [ "${EXECUTION_GCDF_AUDIT_ENABLED}" = "true" ]; then
  record_topic /care_planner/execution_gcdf/selector_summary "${OUT}/execution_gcdf_selector_summary.csv"
  record_topic /care_planner/execution_gcdf/safety_summary "${OUT}/execution_gcdf_safety_summary.csv"
  record_topic /care_planner/execution_gcdf/hard_hold "${OUT}/execution_gcdf_hard_hold.csv"
fi
record_topic /care_planner/active_sensing/blocker_stack_summary "${OUT}/blocker_stack_summary.csv"
record_topic /care_planner/active_sensing/visibility_frontier_summary "${OUT}/visibility_frontier_summary.csv"
record_topic /care_planner/active_sensing/visibility_frontier_target "${OUT}/visibility_frontier_target.csv"
record_topic /care_planner/trajectory_risk/force_bootstrap "${OUT}/force_bootstrap.csv"
record_topic /care_planner/final_gcdf/risk/summary "${OUT}/final_gcdf_risk_summary.csv"
record_topic /care_planner/final_gcdf/selector_summary "${OUT}/final_gcdf_selector_summary.csv"
record_topic /care_planner/final_gcdf/recovery_visibility_event "${OUT}/final_gcdf_recovery_event.csv"
record_topic "${PROBE_SINGLE_FLIGHT_SUMMARY_TOPIC}" "${OUT}/probe_single_flight_summary.csv"
if [ "${USE_LOCAL_SPARSE_SCP}" = "true" ]; then
  record_topic "${LOCAL_SCP_SUMMARY_TOPIC}" "${OUT}/local_planner_summary.csv"
  record_topic /care_planner/local_planner/cdf_selector_summary "${OUT}/local_cdf_selector_summary.csv"
  # Keep the historical filename for downstream summary scripts; fields that
  # are specific to legacy MPC will simply be absent.
  record_topic "${LOCAL_SCP_SUMMARY_TOPIC}" "${OUT}/mpc_summary.csv"
else
  record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
fi
record_topic /care_planner/execution/tracker_summary "${OUT}/tracker_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/verification_outcome "${OUT}/verification_outcome.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
# Visualization artifacts: preserve the upstream nominal task reference and the
# exact safety-certified trajectory that the tracker receives.
record_topic /care_planner/task_trajectory "${OUT}/task_trajectory.csv"
record_topic /care_planner/committed_trajectory "${OUT}/committed_trajectory.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

echo "[WAIT] initial execution gate release (tries=${INITIAL_GATE_MAX_TRIES}, msg_timeout=${INITIAL_GATE_ECHO_TIMEOUT}s)"
if ! python3 - "${INITIAL_GATE_MAX_TRIES}" "${INITIAL_GATE_ECHO_TIMEOUT}" <<'PY'
import sys
import time

import rospy
from std_msgs.msg import String

tries = int(sys.argv[1])
msg_timeout = float(sys.argv[2])
topic = "/care_planner/execution/gate_summary"

# Keep one ROS subscriber alive for the whole wait. Repeated shell-level
# 'timeout rostopic echo' calls can expire during process startup / subscriber
# handshake even when the gate has already published released=1.
rospy.init_node(
    "careplanner_initial_gate_wait",
    anonymous=True,
    disable_signals=True,
)

last = ""
for i in range(1, tries + 1):
    try:
        msg = rospy.wait_for_message(topic, String, timeout=msg_timeout)
        last = str(msg.data)
    except rospy.ROSException:
        pass

    if "released=1" in last:
        print(f"[READY] initial execution gate released after {i} checks")
        raise SystemExit(0)

    if i % 20 == 0:
        print(f"[WAIT] gate still closed after {i} checks")
        if last:
            interesting = []
            for token in last.split():
                if token.startswith((
                    "released=", "decision=", "waypoint_ready=",
                    "release_reason="
                )):
                    interesting.append(token)
            if interesting:
                print(" ".join(interesting))
    time.sleep(0.05)

print("[ERROR] initial gate release timeout")
if last:
    print("[LAST GATE SUMMARY] " + last)
raise SystemExit(1)
PY
then
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if [ "${USE_LOCAL_SPARSE_SCP}" = "true" ]; then
  echo "[ARCH] Sparse-SCP -> executable GCDF(${FINAL_EXECUTABLE_GCDF_ENABLED}) -> exact VBC -> single commit"
  echo "[ARCH] continuation=${COMMITTED_CONTINUATION_ENABLED} execution_audit=${EXECUTION_AUDIT_STREAM_ENABLED} probe_single_flight=${PROBE_SINGLE_FLIGHT_ENABLED}"
else
  echo "[ARCH] candidate verifier != committed execution auditor"
fi
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
if [ "${EARLY_STOP_ON_GOAL}" = "true" ] || [ "${EARLY_STOP_ON_GOAL}" = "1" ]; then
  echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} up to ${RUN_SECONDS}s; early stop on stable EE goal"
  python3 scripts/wait_for_phase_d_goal.py \
    --repo "${REPO}" \
    --timeout-s "${RUN_SECONDS}" \
    --position-tolerance-m "${GOAL_POSITION_TOLERANCE_M}" \
    --orientation-tolerance-rad "${GOAL_ORIENTATION_TOLERANCE_RAD}" \
    --hold-s "${GOAL_SUCCESS_HOLD_S}" \
    --settle-velocity-inf-rad-s "${GOAL_SETTLE_VELOCITY_INF_RAD_S}" \
    --settle-timeout-s "${GOAL_SETTLE_TIMEOUT_S}" \
    --post-success-record-s "${GOAL_POST_SUCCESS_RECORD_S}" \
    --goal-position "${GX}" "${GY}" "${GZ}" \
    --goal-orientation "${GQX}" "${GQY}" "${GQZ}" "${GQW}" \
    --status-json "${OUT}/goal_stop_status.json"
else
  echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for fixed ${RUN_SECONDS}s ROS/Gazebo simulation time"
  python3 scripts/wait_for_ros_duration.py --duration-s "${RUN_SECONDS}"
fi

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
# Teardown should not leave the velocity controller holding the last nonzero
# command after the tracker process exits.
if [ "${EARLY_STOP_ON_GOAL}" = "true" ] || [ "${EARLY_STOP_ON_GOAL}" = "1" ]; then
  timeout 1 rostopic pub -1 "${ACTUATOR_TOPIC}" std_msgs/Float64MultiArray \
    "data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]" \
    >/dev/null 2>&1 || true
fi
sleep 0.2
for pid in "${REC_PIDS[@]:-}"; do kill_group "${pid}"; done
REC_PIDS=()

python3 - "${CASE_ID}" "${OUT}" "${REGION_SCHEDULE_MODE}" <<'PY'
import csv,json,math,os,re,statistics,sys
cid,out,mode=sys.argv[1:]
TOK=re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')
def recs(name):
 p=os.path.join(out,name); a=[]
 if not os.path.isfile(p): return a
 with open(p,newline='',errors='replace') as f:
  rd=csv.reader(f); h=next(rd,[])
  if not h:return a
  ti=h.index('%time') if '%time' in h else 0
  di=h.index('field.data') if 'field.data' in h else 1
  for r in rd:
   if len(r)<=di:continue
   d=dict(TOK.findall(','.join(r[di:])))
   try:d['_t']=float(r[ti])/1e9
   except:d['_t']=math.nan
   if d:a.append(d)
 return a
def f(x):
 try:return float(str(x).replace('ms',''))
 except:return math.nan
reg=recs('regime_summary.csv'); cand=recs('candidate_vbc_summary.csv'); exe=recs('execution_vbc_summary.csv'); commit=recs('commit_summary.csv'); prog=recs('nominal_progress_summary.csv'); mpc=recs('mpc_summary.csv'); sched=recs('waypoint_schedule_summary.csv')
lastr=reg[-1] if reg else {}; lastc=commit[-1] if commit else {}; lastp=prog[-1] if prog else {}; lasts=sched[-1] if sched else {}
sol=[f(r.get('solve','nan')) for r in mpc]; sol=[x for x in sol if math.isfinite(x)]
repair_multi=[r for r in mpc if r.get('vbc_wp')=='multi_deadline_repair']
payload={
 'case_id':cid,
 'region_schedule_mode':mode,
 'architecture':'candidate VBC verifier != committed execution auditor; NORMAL->REPAIR->PROBE_NORMAL->NORMAL',
 'candidate_vbc_records':len(cand),
 'candidate_unsafe_records':sum(r.get('has_violation')=='1' for r in cand if r.get('trajectory_source')=='predicted'),
 'candidate_safe_records':sum(r.get('has_violation')=='0' for r in cand if r.get('trajectory_source')=='predicted'),
 'execution_vbc_records':len(exe),
 'execution_unsafe_records':sum(r.get('has_violation')=='1' for r in exe),
 'execution_safe_records':sum(r.get('has_violation')=='0' for r in exe),
 'final_regime_state':lastr.get('state') if lastr else None,
 'repair_entry_count':int(lastr.get('repair_entry_count','0')) if lastr else 0,
 'candidate_repair_entry_count':int(lastr.get('candidate_repair_entry_count','0')) if lastr else 0,
 'execution_repair_entry_count':int(lastr.get('execution_repair_entry_count','0')) if lastr else 0,
 'execution_safety_event_count':int(lastr.get('execution_safety_event_count','0')) if lastr else 0,
 'probe_entry_count':int(lastr.get('probe_entry_count','0')) if lastr else 0,
 'probe_failure_count':int(lastr.get('probe_failure_count','0')) if lastr else 0,
 'normal_entry_count':int(lastr.get('normal_entry_count','0')) if lastr else 0,
 'verification_safe_count':int(lastc.get('verification_safe_count','0')) if lastc else 0,
 'verification_unsafe_count':int(lastc.get('verification_unsafe_count','0')) if lastc else 0,
 'commit_count':int(lastc.get('commit_count','0')) if lastc else 0,
 'final_progress_phase_s':f(lastp.get('phase_s','nan')) if lastp else None,
 'final_wall_elapsed_s':f(lastp.get('wall_elapsed_s','nan')) if lastp else None,
 'mpc_solve_ms_median':statistics.median(sol) if sol else None,
 'mpc_solve_ms_max':max(sol) if sol else None,
 'multi_deadline_repair_cycles':len(repair_multi),
 'max_repair_obligation_count':max([int(r.get('repair_obligation_count','0')) for r in repair_multi] or [0]),
 'last_schedule_obligation_count':int(lasts.get('obligation_count','0')) if lasts else 0,
 'last_schedule_unreachable_at_discovery_count':int(lasts.get('unreachable_at_discovery_count','0')) if lasts else 0,
}
json.dump(payload,open(os.path.join(out,'c4_4_verified_regime_summary.json'),'w'),indent=2)
print(json.dumps(payload,indent=2))
PY

echo "[RESULT]    ${OUT}/c4_4_verified_regime_summary.json"
echo "[REGIME]    ${OUT}/regime_summary.csv"
echo "[CANDIDATE] ${OUT}/candidate_vbc_summary.csv"
echo "[EXECUTION] ${OUT}/execution_vbc_summary.csv"
echo "[SCHEDULE]  ${OUT}/waypoint_schedule_summary.csv"
