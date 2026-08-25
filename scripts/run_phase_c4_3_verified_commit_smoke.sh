#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
CONFIG_FILE="${CONFIG_FILE:-${REPO}/src/egocentric_arm_planner/config/planner_phase1_c4_3_planner.yaml}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-8.0}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
# This experiment intentionally keeps the previous CPU runtime and changes only
# startup warm-up. Override manually only for a later dedicated CPU/GPU study.
NCDF_DEVICE="${NCDF_DEVICE:-cpu}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
C4_STREAK="${C4_STREAK:-2}"
C4_SAFE_STREAK="${C4_SAFE_STREAK:-2}"
C4_INPUT_TIMEOUT="${C4_INPUT_TIMEOUT:-0.25}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.20}"
OUT="${OUT:-${REPO}/outputs/phase_c4_3_verified_commit_smoke/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_3_verified_commit_smoke/${CASE_ID}}"

RAW_MPC_TOPIC="/care_planner/mpc/predicted_trajectory"
VERIFY_TOPIC="/care_planner/optimized_trajectory"
COMMITTED_TOPIC="/care_planner/committed_trajectory"
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

echo "[NCDF] online device=${NCDF_DEVICE}; startup autograd warm-up enabled; oracle diagnostics disabled"

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
  if kill -0 "${pid}" 2>/dev/null; then kill -INT -- "-${pid}" 2>/dev/null || true; sleep 0.3; fi
  if kill -0 "${pid}" 2>/dev/null; then kill -TERM -- "-${pid}" 2>/dev/null || true; sleep 0.3; fi
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

# The low-level controller sees ONLY globally verified committed trajectories.
setsid roslaunch egocentric_arm_planner c4_3_low_level_tracker.launch \
  config_file:="${CONFIG_FILE}" input_trajectory:="${COMMITTED_TOPIC}" \
  > "${LOG}/low_level_tracker.log" 2>&1 &
TRACKER_PID=$!

# The waypoint generator reasons about the serialized candidate under VBC audit,
# never about an unassociated newer raw MPC horizon.
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

setsid roslaunch egocentric_arm_planner phaseC4_3_continuous_planner.launch \
  config_file:="${CONFIG_FILE}" waypoint_weight:="${CARE_WEIGHT}" \
  vbc_min_margin_s:="${SAFETY_MARGIN}" \
  predicted_vbc_recovery_consecutive_violations:="${C4_STREAK}" \
  predicted_vbc_recovery_consecutive_safe:="${C4_SAFE_STREAK}" \
  predicted_vbc_recovery_input_timeout:="${C4_INPUT_TIMEOUT}" \
  selector_predicted_trajectory_timeout:="${PREDICTION_TIMEOUT}" \
  rolling_safe_clear_consecutive:="${C4_SAFE_STREAK}" \
  trial_label:="${CASE_ID}_c4_3_verified_commit" log_output_root:="${OUT}" \
  goal_x:="${GX}" goal_y:="${GY}" goal_z:="${GZ}" \
  goal_qx:="${GQX}" goal_qy:="${GQY}" goal_qz:="${GQZ}" goal_qw:="${GQW}" \
  > "${LOG}/controlled.log" 2>&1 &
CONTROL_PID=$!

READY=0
for _ in $(seq 1 300); do
  NODES="$(rosnode list 2>/dev/null || true)"
  if echo "${NODES}" | grep -q '^/velocity_qp_mpc_waypoint_node$' && \
     echo "${NODES}" | grep -q '^/trajectory_execution_manager_node$' && \
     echo "${NODES}" | grep -q '^/optimized_trajectory_continuity$' && \
     echo "${NODES}" | grep -q '^/trajectory_vbc_selector_node$' && \
     echo "${NODES}" | grep -q '^/phase_b2_controlled_trial$' && \
     echo "${NODES}" | grep -q '^/initial_body_prior_initializer$' && \
     echo "${NODES}" | grep -q '^/vbc_execution_reference_gate$'; then READY=1; break; fi
  sleep 0.1
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] verified-commit planner nodes did not all start"
  rosnode list 2>/dev/null || true
  tail -n 220 "${LOG}/controlled.log" || true
  exit 1
fi

rosnode info /velocity_qp_mpc_waypoint_node > "${OUT}/mpc_node_info.txt" 2>&1 || true
rosnode info /trajectory_execution_manager_node > "${OUT}/tracker_node_info.txt" 2>&1 || true
rosnode info /optimized_trajectory_continuity > "${OUT}/commit_node_info.txt" 2>&1 || true
if grep -Fq "* ${ACTUATOR_TOPIC} " "${OUT}/mpc_node_info.txt"; then
  echo "[ERROR] MPC still owns actuator command"; exit 1
fi
if ! grep -Fq "* ${ACTUATOR_TOPIC} " "${OUT}/tracker_node_info.txt"; then
  echo "[ERROR] low-level tracker does not own actuator command"; exit 1
fi
if ! grep -Fq "* ${RAW_MPC_TOPIC} " "${OUT}/commit_node_info.txt" || \
   ! grep -Fq "* ${VERIFY_TOPIC} " "${OUT}/commit_node_info.txt" || \
   ! grep -Fq "* ${COMMITTED_TOPIC} " "${OUT}/commit_node_info.txt"; then
  echo "[ERROR] candidate-verify-commit wiring mismatch"
  cat "${OUT}/commit_node_info.txt" || true
  exit 1
fi

echo "[ARCH] raw MPC candidate -> serialized VBC verification -> committed trajectory -> low-level tracker"

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}
record_topic /care_planner/trajectory_risk/vbc_summary "${OUT}/selector_summary.csv"
record_topic /phase_b2_controlled_trial/summary "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary "${OUT}/waypoint_summary.csv"
record_topic /care_planner/execution/predicted_vbc_recovery_summary "${OUT}/predicted_vbc_recovery_guard.csv"
record_topic /care_planner/execution/gate_summary "${OUT}/gate_summary.csv"
record_topic /velocity_qp_mpc_waypoint_node/summary "${OUT}/mpc_summary.csv"
record_topic /care_planner/optimized_trajectory_summary "${OUT}/commit_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/low_level_reference_state.csv"
record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

# Do not wait on one-shot /task_trajectory. The upstream command trajectory is
# persistent and is the correct startup milestone after the initial EE goal.
TASK_READY=0
for _ in $(seq 1 160); do
  if timeout 1 rostopic echo -n 1 /care_planner/command_trajectory_candidate >/dev/null 2>&1; then TASK_READY=1; break; fi
  sleep 0.05
done
if [ "${TASK_READY}" != "1" ]; then
  echo "[ERROR] initial EE goal did not produce persistent command trajectory"
  grep -E 'phase_b2_trial|RecedingHorizonPlanner|Traceback|ImportError' "${LOG}/controlled.log" | tail -n 180 || true
  exit 1
fi

RELEASED=0
for _ in $(seq 1 400); do
  S="$(timeout 1 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.05
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  grep -E 'vbc_execution_reference_gate|trajectory_vbc|waypoint|phase_b2_trial' "${LOG}/controlled.log" | tail -n 220 || true
  exit 1
fi

echo "[RUN] ${CASE_ID}: verified-commit CAREPlanner for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
kill_group "${TRACKER_PID}"; TRACKER_PID=""
sleep 0.2
for pid in "${REC_PIDS[@]:-}"; do kill_group "${pid}"; done
REC_PIDS=()

python3 - "${CASE_ID}" "${OUT}" "${LOG}/controlled.log" <<'PY'
import csv,glob,json,math,os,re,statistics,sys
cid,out,log=sys.argv[1:]
TOK=re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')
def records(name):
 p=os.path.join(out,name); a=[]
 if not os.path.isfile(p): return a
 with open(p,newline='',errors='replace') as f:
  rd=csv.reader(f); h=next(rd,[])
  if not h:return a
  ti=h.index('%time') if '%time' in h else 0
  di=h.index('field.data') if 'field.data' in h else 1
  for row in rd:
   if len(row)<=di: continue
   d=dict(TOK.findall(','.join(row[di:])))
   try:d['_t']=float(row[ti])/1e9
   except:d['_t']=math.nan
   if d:a.append(d)
 return a
def num(x):
 try:return float(str(x).replace('ms',''))
 except:return math.nan
mpc=records('mpc_summary.csv'); guard=records('predicted_vbc_recovery_guard.csv')
commit=records('commit_summary.csv'); sel=records('selector_summary.csv'); broker=records('broker_summary.csv')
solves=[num(r.get('solve','nan')) for r in mpc]; solves=[x for x in solves if math.isfinite(x)]
traces=[]
for p in glob.glob(os.path.join(out,'projector_traces','*.json')):
 try:
  d=json.load(open(p)); x=d.get('generation_compute_ms')
  if x is not None and math.isfinite(float(x)): traces.append(float(x))
 except: pass
last=commit[-1] if commit else {}
text=open(log,errors='replace').read() if os.path.isfile(log) else ''
payload={
 'case_id':cid,
 'architecture':'raw candidate -> serialized global VBC verification -> committed trajectory -> low-level tracker',
 'selector_records':len(sel),
 'selector_no_violation_records':sum(r.get('has_violation')=='0' for r in sel),
 'mpc_summary_records':len(mpc),
 'mpc_solve_ms_median':statistics.median(solves) if solves else None,
 'mpc_solve_ms_max':max(solves) if solves else None,
 'waypoint_generation_count':len(glob.glob(os.path.join(out,'projector_traces','*.json'))),
 'waypoint_compute_ms_median':statistics.median(traces) if traces else None,
 'waypoint_compute_ms_max':max(traces) if traces else None,
 'raw_input_count':int(last.get('raw_input_count','0')) if last else 0,
 'verification_publish_count':int(last.get('verification_publish_count','0')) if last else 0,
 'verification_safe_count':int(last.get('verification_safe_count','0')) if last else 0,
 'verification_unsafe_count':int(last.get('verification_unsafe_count','0')) if last else 0,
 'verification_timeout_count':int(last.get('verification_timeout_count','0')) if last else 0,
 'commit_count':int(last.get('commit_count','0')) if last else 0,
 'committed_publish_count':int(last.get('committed_publish_count','0')) if last else 0,
 'continuation_count':int(last.get('continuation_count','0')) if last else 0,
 'max_raw_mpc_input_gap_s':num(last.get('max_input_gap_s','0')) if last else 0.0,
 'guard_trigger_count_total':max([int(r.get('trigger_count_total','0')) for r in guard] or [0]),
 'guard_clear_count_total':max([int(r.get('clear_count_total','0')) for r in guard] or [0]),
 'broker_replan_requested_count':max([int(r.get('replan_count','0')) for r in broker] or [0]),
 'measured_state_replan_logs':len(re.findall(r'RECOVERY EPISODE COMPLETE -> measured-state replan',text)),
 'final_guard_status':guard[-1].get('status') if guard else None,
 'final_guard_routing':guard[-1].get('routing') if guard else None,
}
json.dump(payload,open(os.path.join(out,'c4_3_verified_commit_summary.json'),'w'),indent=2)
print(json.dumps(payload,indent=2))
PY

echo "[RESULT] ${OUT}/c4_3_verified_commit_summary.json"
echo "[VERIFY/COMMIT] ${OUT}/commit_summary.csv"
echo "[TIMING] ${OUT}/mpc_summary.csv + projector_traces/*.json"
echo "[TRACKING] ${OUT}/joint_states.csv + low_level_reference_state.csv + actuator_command.csv"
