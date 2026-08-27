#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
CONFIG_FILE="${CONFIG_FILE:-${REPO}/src/egocentric_arm_planner/config/planner_phase1_c4_3_planner.yaml}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-8.0}"
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
OUT="${OUT:-${REPO}/outputs/phase_c4_4_verified_regime_smoke/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_4_verified_regime_smoke/${CASE_ID}}"

RAW_MPC_TOPIC="/care_planner/mpc/predicted_trajectory"
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

setsid roslaunch arm_description gazebo_velocity_control.launch \
  gazebo_gui:=false use_rviz:=false > "${LOG}/gazebo.log" 2>&1 &
GAZEBO_PID=$!

READY=0
for _ in $(seq 1 120); do
  if timeout 2 rostopic echo -n 1 /care_arm/joint_states >/dev/null 2>&1; then READY=1; break; fi
  sleep 0.25
done
[ "${READY}" = "1" ] || { echo "[ERROR] Gazebo joint state timeout"; exit 1; }

setsid roslaunch egocentric_arm_planner c4_3_low_level_tracker.launch \
  config_file:="${CONFIG_FILE}" input_trajectory:="${COMMITTED_TOPIC}" \
  output_velocity_command:="${TRACKER_DESIRED_TOPIC}" \
  use_acceleration_limiter:=true \
  rate_limiter_output_velocity_command:="${ACTUATOR_TOPIC}" \
  > "${LOG}/low_level_tracker.log" 2>&1 &
TRACKER_PID=$!

setsid bash -lc "source '${CONDA_SH}'; conda activate '${NCDF_ENV}'; cd '${REPO}'; source devel/setup.bash; exec python -u src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py _device:=${NCDF_DEVICE} _rate:=50.0 _enable_oracle_diagnostics:=false _region_schedule_mode:='${REGION_SCHEDULE_MODE}' _predicted_trajectory_topic:='${VERIFY_TOPIC}' _safety_margin_s:=${SAFETY_MARGIN} _predicted_trajectory_timeout:=${PREDICTION_TIMEOUT} _target_cell_resolution:=0.05 _projection_iters:=10 _projection_damping:=0.5 _projection_epsilon_f:=0.03 _projection_max_step_norm:=0.25 _root_refine_iters:=12 _root_tolerance_f:=0.002 _ascent_steps:=1 _ascent_step_size:=0.05 _ascent_max_step_norm:=0.25 _output_root:='${OUT}/projector_traces'" \
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
  config_file:="${CONFIG_FILE}" waypoint_weight:="${CARE_WEIGHT}" \
  vbc_min_margin_s:="${SAFETY_MARGIN}" \
  selector_predicted_trajectory_timeout:="${PREDICTION_TIMEOUT}" \
  trajectory_risk_input_topic:="${TRAJECTORY_RISK_INPUT_TOPIC}" \
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
  BASE_READY=0
  CDF_READY=1

  if echo "${NODES}" | grep -q '^/velocity_qp_mpc_waypoint_node$' && \
     echo "${NODES}" | grep -q '^/trajectory_execution_manager_node$' && \
     echo "${NODES}" | grep -q '^/joint_velocity_rate_limiter$' && \
     echo "${NODES}" | grep -q '^/optimized_trajectory_continuity$' && \
     echo "${NODES}" | grep -q '^/trajectory_vbc_selector_node$' && \
     echo "${NODES}" | grep -q '^/execution_vbc_audit/trajectory_vbc_selector_node$' && \
     echo "${NODES}" | grep -q '^/c4_4_verified_regime_manager$' && \
     echo "${NODES}" | grep -q '^/phase_b2_controlled_trial$' && \
     echo "${NODES}" | grep -q '^/vbc_execution_reference_gate$'; then
    BASE_READY=1
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

  if [ "${BASE_READY}" = "1" ] && [ "${CDF_READY}" = "1" ]; then
    READY=1
    break
  fi

  sleep 0.1
done

if [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4/C5.3a required nodes did not all start"
  echo "[DEBUG] CDF_SELECTOR_ENABLED=${CDF_SELECTOR_ENABLED}"
  echo "[DEBUG] CDF_SHADOW_VBC_AUDIT_ENABLED=${CDF_SHADOW_VBC_AUDIT_ENABLED}"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
 && \
     echo "${NODES}" | grep -q '^/trajectory_execution_manager_nodeif [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
 && \
     echo "${NODES}" | grep -q '^/joint_velocity_rate_limiterif [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
 && \
     echo "${NODES}" | grep -q '^/optimized_trajectory_continuityif [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
 && \
     echo "${NODES}" | grep -q '^/trajectory_vbc_selector_nodeif [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
 && \
     echo "${NODES}" | grep -q '^/execution_vbc_audit/trajectory_vbc_selector_nodeif [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
 && \
     echo "${NODES}" | grep -q '^/c4_4_verified_regime_managerif [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
 && \
     echo "${NODES}" | grep -q '^/phase_b2_controlled_trialif [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
 && \
     echo "${NODES}" | grep -q '^/vbc_execution_reference_gateif [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
; then
    BASE_READY=1
  fi

  if [ "${CDF_SELECTOR_ENABLED}" = "true" ]; then
    if ! echo "${NODES}" | grep -q '^/c5_3a_cpp_forbidden_voxel_gpu_shadowif [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
; then
      CDF_READY=0
    fi
  fi
  if [ "${CDF_SHADOW_VBC_AUDIT_ENABLED}" = "true" ]; then
    if ! echo "${NODES}" | grep -q '^/cdf_shadow_vbc/trajectory_vbc_selector_nodeif [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
; then
      CDF_READY=0
    fi
  fi

  if [ "${BASE_READY}" = "1" ] && [ "${CDF_READY}" = "1" ]; then
    READY=1
    break
  fi
  sleep 0.1
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.4 nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

if rosnode list | grep -q '^/predicted_vbc_recovery_guard$'; then
  echo "[ERROR] legacy predicted_vbc_recovery_guard unexpectedly running"
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic "${CANDIDATE_VBC_TOPIC}" "${OUT}/candidate_vbc_summary.csv"
record_topic "${EXECUTION_VBC_TOPIC}" "${OUT}/execution_vbc_summary.csv"
record_topic "${REGIME_TOPIC}" "${OUT}/regime_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic "${SCHEDULE_SUMMARY_TOPIC}" "${OUT}/waypoint_schedule_summary.csv"
record_topic "${SCHEDULE_TOPIC}" "${OUT}/waypoint_schedule.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 260 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] candidate verifier != committed execution auditor"
echo "[REGIME] NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL (${PROBE_SAFE_COMMITS} safe probe commits required)"
echo "[RUN] ${CASE_ID}: ${REGION_SCHEDULE_MODE} for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
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
