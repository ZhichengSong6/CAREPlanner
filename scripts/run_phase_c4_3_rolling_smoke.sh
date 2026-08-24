#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-8.0}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
C4_STREAK="${C4_STREAK:-2}"
C4_INPUT_TIMEOUT="${C4_INPUT_TIMEOUT:-0.25}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.20}"
PRIMARY_WINDOW_S="${PRIMARY_WINDOW_S:-0.25}"
SWITCH_HYSTERESIS_S="${SWITCH_HYSTERESIS_S:-0.05}"
AUDIT_WARN_MS="${AUDIT_WARN_MS:-5.0}"
OUT="${OUT:-${REPO}/outputs/phase_c4_3_rolling_smoke/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_3_rolling_smoke/${CASE_ID}}"

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

GAZEBO_PID=""; GEN_PID=""; CONTROL_PID=""
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
  waypoint_weight:="${CARE_WEIGHT}" \
  recovery_enabled:=true recovery_weight_scale:=1.0 \
  rolling_target_mode:=true \
  selector_prefer_predicted_trajectory:=true \
  selector_predicted_trajectory_timeout:="${PREDICTION_TIMEOUT}" \
  primary_frontier_window_s:="${PRIMARY_WINDOW_S}" \
  target_switch_hysteresis_s:="${SWITCH_HYSTERESIS_S}" \
  recovery_physical_deadline_topic:=/care_planner/active_sensing/visibility_waypoint_deadline_rolling \
  predictive_recovery_enabled:=false \
  enable_predicted_vbc_auditor:=true \
  predicted_vbc_audit_warn_ms:="${AUDIT_WARN_MS}" \
  predicted_vbc_recovery_enabled:=true \
  predicted_vbc_recovery_use_global_selector:=true \
  predicted_vbc_recovery_consecutive_violations:="${C4_STREAK}" \
  predicted_vbc_recovery_input_timeout:="${C4_INPUT_TIMEOUT}" \
  vbc_min_margin_s:="${SAFETY_MARGIN}" \
  enable_executed_vbc_logger:=false \
  enable_waypoint_logger:=false \
  trial_label:="${CASE_ID}_c4_3" \
  log_output_root:="${OUT}" \
  use_trajectory_visualizer:=false \
  goal_x:="${GX}" goal_y:="${GY}" goal_z:="${GZ}" \
  goal_qx:="${GQX}" goal_qy:="${GQY}" goal_qz:="${GQZ}" goal_qw:="${GQW}" \
  > "${LOG}/controlled.log" 2>&1 &
CONTROL_PID=$!

READY=0
for _ in $(seq 1 200); do
  NODES="$(rosnode list 2>/dev/null || true)"
  if echo "${NODES}" | grep -q '^/predicted_vbc_recovery_guard$' && \
     echo "${NODES}" | grep -q '^/predicted_vbc_auditor_node$' && \
     echo "${NODES}" | grep -q '^/trajectory_vbc_selector_node$' && \
     echo "${NODES}" | grep -q '^/phase_b2_controlled_trial$'; then
    READY=1; break
  fi
  sleep 0.1
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.3 nodes did not start"
  tail -n 120 "${LOG}/controlled.log" || true
  exit 1
fi

record_topic() {
  local topic="$1"; local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" \
    > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}

record_topic /care_planner/trajectory_risk/vbc_summary \
  "${OUT}/selector_summary.csv"
record_topic /phase_b2_controlled_trial/summary \
  "${OUT}/broker_summary.csv"
record_topic /care_planner/active_sensing/visibility_waypoint_summary \
  "${OUT}/waypoint_summary.csv"
record_topic /care_planner/execution/predicted_vbc_audit_summary \
  "${OUT}/predicted_vbc_audit.csv"
record_topic /care_planner/execution/predicted_vbc_recovery_summary \
  "${OUT}/predicted_vbc_recovery_guard.csv"
record_topic /care_planner/execution/gate_summary \
  "${OUT}/gate_summary.csv"

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
  tail -n 120 "${LOG}/waypoint_generator.log" || true
  exit 1
fi

echo "[RUN] ${CASE_ID}: C4.3 rolling Primary/Secondary VBC for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

# Freeze the experiment first, then stop all recorders.  This gives every topic
# trace the same effective end time instead of letting later recorders observe
# extra Recovery/replan events while earlier recorders are already stopped.
kill_group "${CONTROL_PID}"; CONTROL_PID=""
kill_group "${GEN_PID}"; GEN_PID=""
sleep 0.2
for pid in "${REC_PIDS[@]:-}"; do kill_group "${pid}"; done
REC_PIDS=()

python3 - "${CASE_ID}" "${OUT}" "${LOG}/controlled.log" <<'PY'
import csv,glob,json,os,re,sys
from collections import Counter

cid,out,controlled=sys.argv[1:]
TOKEN_RE=re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')

def records(path):
    ans=[]
    if not os.path.isfile(path): return ans
    try:
        with open(path,newline='',errors='replace') as fh:
            rd=csv.reader(fh); h=next(rd,[])
            if not h: return ans
            di=h.index('field.data') if 'field.data' in h else 1
            for row in rd:
                if len(row)<=di: continue
                # rostopic -p does not quote commas embedded inside String data.
                # Reconstruct field.data before tokenizing target=[x,y,z] etc.
                message=','.join(row[di:])
                tok=dict(TOKEN_RE.findall(message))
                if tok: ans.append(tok)
    except Exception:
        return []
    return ans

def i(v,default=0):
    try: return int(v)
    except: return default

selector=records(os.path.join(out,'selector_summary.csv'))
broker=records(os.path.join(out,'broker_summary.csv'))
waypoint=records(os.path.join(out,'waypoint_summary.csv'))
audit=records(os.path.join(out,'predicted_vbc_audit.csv'))
guard=records(os.path.join(out,'predicted_vbc_recovery_guard.csv'))
gate=records(os.path.join(out,'gate_summary.csv'))

selected=[r for r in selector if r.get('has_violation')=='1']
source_counts=Counter(r.get('trajectory_source','unknown') for r in selector)
group_counts=Counter(r.get('selected_group','none') for r in selected)
reason_counts=Counter(r.get('selection_reason','none') for r in selected)

# Logical rolling-target transitions are voxel transitions, not millimetre
# changes of a sampled point inside the same confidence-map cell.
cells=[]
for r in waypoint:
    if r.get('selection_active')!='1': continue
    cell=r.get('target_cell')
    if cell and cell!='none' and (not cells or cells[-1]!=cell): cells.append(cell)

raw_targets=[]
for r in broker:
    if r.get('selected')!='1' or r.get('selected_active')!='1': continue
    t=r.get('target')
    if t and (not raw_targets or raw_targets[-1]!=t): raw_targets.append(t)

# This auditor follows only the selected steering target in C4.3. It is kept as
# a diagnostic and is no longer the Recovery safety source.
active_audit=[r for r in audit if r.get('status') not in ('inactive','waiting_target')]
selected_target_viol=[r for r in active_audit if r.get('violation')=='1']
audit_status=Counter(r.get('status','unknown') for r in active_audit)

global_guard=[r for r in guard if r.get('mode')=='global_set']
global_violation_records=[r for r in global_guard if r.get('violation')=='1']
max_trigger=max([i(r.get('trigger_count_total')) for r in guard] or [0])
replan_requested=max([i(r.get('replan_count')) for r in broker] or [0])
replan_installed=max([i(r.get('replan_count')) for r in gate] or [0])
lock_records=sum(i(r.get('target_lock'))==1 for r in broker)
hold_records=sum(i(r.get('verification_hold'))==1 for r in guard)
stale_hold_records=sum(r.get('routing')=='global_summary_stale_verification_hold' for r in guard)
prediction_stale_records=sum(i(r.get('prediction_fresh'),1)==0 for r in broker if r.get('released')=='1')

text=open(controlled,errors='replace').read() if os.path.isfile(controlled) else ''
recovery_entries=len(re.findall(
    r'(?:entering VISIBILITY RECOVERY|VBC RECOVERY TRIGGERED WHILE UNSEEN)',text))
lock_logs=len(re.findall(r'rolling target LOCKED',text))
unlock_logs=len(re.findall(r'rolling target selection UNLOCKED',text))

payload={
  'case_id':cid,
  'selector_records':len(selector),
  'selector_source_counts':dict(source_counts),
  'predicted_source_observed':source_counts.get('predicted',0)>0,
  'selected_group_counts':dict(group_counts),
  'selection_reason_counts':dict(reason_counts),
  'selector_no_violation_records':sum(r.get('has_violation')=='0' for r in selector),
  'target_cell_transition_count':max(0,len(cells)-1),
  'selected_target_cell_sequence':cells,
  'unique_selected_target_cell_count':len(set(cells)),
  'raw_broker_target_transition_count':max(0,len(raw_targets)-1),
  'raw_broker_target_sequence':raw_targets,
  'waypoint_generation_trace_count':len(glob.glob(os.path.join(out,'projector_traces','vbc_visibility_waypoint_*.json'))),
  'waypoint_records':len(waypoint),
  'waypoint_predicted_source_observed':any(r.get('trajectory_source')=='predicted' for r in waypoint),
  'selected_target_audit_records':len(active_audit),
  'selected_target_violation_audits':len(selected_target_viol),
  'selected_target_audit_status_counts':dict(audit_status),
  'global_guard_records':len(global_guard),
  'global_guard_violation_records':len(global_violation_records),
  'global_guard_mode_observed':bool(global_guard),
  'guard_trigger_count_total':max_trigger,
  'recovery_entry_log_count':recovery_entries,
  'replan_requested_count':replan_requested,
  'replan_installed_count':replan_installed,
  'target_lock_summary_records':lock_records,
  'target_lock_log_count':lock_logs,
  'target_unlock_log_count':unlock_logs,
  'prediction_stale_broker_records':prediction_stale_records,
  'verification_hold_summary_records':hold_records,
  'global_stale_verification_hold_records':stale_hold_records,
  'gate_released_observed':any(r.get('released')=='1' for r in gate),
  'final_guard_routing':guard[-1].get('routing') if guard else None,
  'final_guard_status':guard[-1].get('status') if guard else None,
  'final_guard_episode_armed':guard[-1].get('episode_armed') if guard else None,
}
json.dump(payload,open(os.path.join(out,'c4_3_rolling_summary.json'),'w'),indent=2)
print(json.dumps(payload,indent=2))
PY

echo "[RESULT]   ${OUT}/c4_3_rolling_summary.json"
echo "[SELECTOR] ${OUT}/selector_summary.csv"
echo "[BROKER]   ${OUT}/broker_summary.csv"
echo "[WAYPOINT] ${OUT}/waypoint_summary.csv"
echo "[AUDIT]    ${OUT}/predicted_vbc_audit.csv  (selected-target diagnostic only)"
echo "[GUARD]    ${OUT}/predicted_vbc_recovery_guard.csv  (global-set Recovery source)"
