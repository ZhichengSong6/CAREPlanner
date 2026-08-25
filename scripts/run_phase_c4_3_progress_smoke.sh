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
C4_STREAK="${C4_STREAK:-2}"
C4_SAFE_STREAK="${C4_SAFE_STREAK:-2}"
C4_INPUT_TIMEOUT="${C4_INPUT_TIMEOUT:-0.25}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.20}"
OUT="${OUT:-${REPO}/outputs/phase_c4_3_progress_smoke/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_3_progress_smoke/${CASE_ID}}"

RAW_MPC_TOPIC="/care_planner/mpc/predicted_trajectory"
VERIFY_TOPIC="/care_planner/optimized_trajectory"
COMMITTED_TOPIC="/care_planner/committed_trajectory"
TRACKER_DESIRED_TOPIC="/care_planner/execution/tracker_velocity_desired"
ACTUATOR_TOPIC="/care_arm/arm_group_velocity_controller/command"

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

setsid bash -lc "source '${CONDA_SH}'; conda activate '${NCDF_ENV}'; cd '${REPO}'; source devel/setup.bash; exec python -u src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py _device:=${NCDF_DEVICE} _rate:=50.0 _enable_oracle_diagnostics:=false _predicted_trajectory_topic:='${VERIFY_TOPIC}' _safety_margin_s:=${SAFETY_MARGIN} _predicted_trajectory_timeout:=${PREDICTION_TIMEOUT} _target_cell_resolution:=0.05 _projection_iters:=10 _projection_damping:=0.5 _projection_epsilon_f:=0.03 _projection_max_step_norm:=0.25 _root_refine_iters:=12 _root_tolerance_f:=0.002 _ascent_steps:=1 _ascent_step_size:=0.05 _ascent_max_step_norm:=0.25 _output_root:='${OUT}/projector_traces'" \
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

setsid roslaunch egocentric_arm_planner phaseC4_3_progress_planner.launch \
  config_file:="${CONFIG_FILE}" waypoint_weight:="${CARE_WEIGHT}" \
  vbc_min_margin_s:="${SAFETY_MARGIN}" \
  predicted_vbc_recovery_consecutive_violations:="${C4_STREAK}" \
  predicted_vbc_recovery_consecutive_safe:="${C4_SAFE_STREAK}" \
  predicted_vbc_recovery_input_timeout:="${C4_INPUT_TIMEOUT}" \
  selector_predicted_trajectory_timeout:="${PREDICTION_TIMEOUT}" \
  rolling_safe_clear_consecutive:="${C4_SAFE_STREAK}" \
  trial_label:="${CASE_ID}_c4_3_progress" log_output_root:="${OUT}" \
  goal_x:="${GX}" goal_y:="${GY}" goal_z:="${GZ}" \
  goal_qx:="${GQX}" goal_qy:="${GQY}" goal_qz:="${GQZ}" goal_qw:="${GQW}" \
  > "${LOG}/controlled.log" 2>&1 &
CONTROL_PID=$!

READY=0
for _ in $(seq 1 300); do
  NODES="$(rosnode list 2>/dev/null || true)"
  if echo "${NODES}" | grep -q '^/velocity_qp_mpc_waypoint_node$' && \
     echo "${NODES}" | grep -q '^/trajectory_execution_manager_node$' && \
     echo "${NODES}" | grep -q '^/joint_velocity_rate_limiter$' && \
     echo "${NODES}" | grep -q '^/optimized_trajectory_continuity$' && \
     echo "${NODES}" | grep -q '^/trajectory_vbc_selector_node$' && \
     echo "${NODES}" | grep -q '^/phase_b2_controlled_trial$' && \
     echo "${NODES}" | grep -q '^/vbc_execution_reference_gate$'; then READY=1; break; fi
  sleep 0.1
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] progress-planner nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 220 "${LOG}/controlled.log" || true
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/execution/nominal_progress_summary "${OUT}/nominal_progress_summary.csv"
record_topic /care_planner/trajectory_risk/vbc_summary "${OUT}/selector_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic /care_planner/execution/predicted_vbc_recovery_summary "${OUT}/predicted_vbc_recovery_guard.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_planner/execution/rate_limiter_summary "${OUT}/rate_limiter_summary.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

# Gate summary is latched/continuous, so it is a reliable readiness milestone.
RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  tail -n 240 "${LOG}/controlled.log" || true
  exit 1
fi

echo "[ARCH] progress-anchored nominal -> 1s MPC candidate -> VBC verify -> committed trajectory -> tracker -> acceleration limiter -> actuator"
echo "[RUN] ${CASE_ID}: progress-anchored CAREPlanner for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
sleep 0.2
for pid in "${REC_PIDS[@]:-}"; do kill_group "${pid}"; done
REC_PIDS=()

python3 - "${CASE_ID}" "${OUT}" <<'PY'
import csv,json,math,os,re,statistics,sys
cid,out=sys.argv[1:]
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
prog=recs('nominal_progress_summary.csv'); guard=recs('predicted_vbc_recovery_guard.csv'); commit=recs('commit_summary.csv'); mpc=recs('mpc_summary.csv')
lastp=prog[-1] if prog else {}; lastc=commit[-1] if commit else {}
sol=[f(r.get('solve','nan')) for r in mpc]; sol=[x for x in sol if math.isfinite(x)]
payload={
 'case_id':cid,
 'nominal_progress_records':len(prog),
 'final_progress_phase_s':f(lastp.get('phase_s','nan')) if lastp else None,
 'final_wall_elapsed_s':f(lastp.get('wall_elapsed_s','nan')) if lastp else None,
 'final_progress_lag_s':f(lastp.get('lag_s','nan')) if lastp else None,
 'max_progress_lag_s':max([f(r.get('lag_s','nan')) for r in prog if math.isfinite(f(r.get('lag_s','nan')))] or [0.0]),
 'max_projection_error_inf':max([f(r.get('projection_error_inf','nan')) for r in prog if math.isfinite(f(r.get('projection_error_inf','nan')))] or [0.0]),
 'progress_frozen_count':int(lastp.get('frozen_count','0')) if lastp else 0,
 'verification_safe_count':int(lastc.get('verification_safe_count','0')) if lastc else 0,
 'verification_unsafe_count':int(lastc.get('verification_unsafe_count','0')) if lastc else 0,
 'commit_count':int(lastc.get('commit_count','0')) if lastc else 0,
 'guard_trigger_count_total':max([int(r.get('trigger_count_total','0')) for r in guard] or [0]),
 'guard_clear_count_total':max([int(r.get('clear_count_total','0')) for r in guard] or [0]),
 'mpc_solve_ms_median':statistics.median(sol) if sol else None,
 'mpc_solve_ms_max':max(sol) if sol else None,
}
json.dump(payload,open(os.path.join(out,'c4_3_progress_summary.json'),'w'),indent=2)
print(json.dumps(payload,indent=2))
PY

echo "[RESULT]   ${OUT}/c4_3_progress_summary.json"
echo "[PROGRESS] ${OUT}/nominal_progress_summary.csv"
echo "[DYNAMICS] ${OUT}/actuator_command.csv + rate_limiter_summary.csv"
