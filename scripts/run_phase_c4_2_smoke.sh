#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-6.0}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
C4_STREAK="${C4_STREAK:-2}"
C4_INPUT_TIMEOUT="${C4_INPUT_TIMEOUT:-0.25}"
AUDIT_WARN_MS="${AUDIT_WARN_MS:-5.0}"
OUT="${OUT:-${REPO}/outputs/phase_c4_2_smoke/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_2_smoke/${CASE_ID}}"

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

GAZEBO_PID=""; GEN_PID=""; CONTROL_PID=""; AUDIT_PID=""; GUARD_PID=""
kill_group() {
  local pid="${1:-}"
  [ -z "${pid}" ] && return 0
  if kill -0 "${pid}" 2>/dev/null; then kill -INT -- "-${pid}" 2>/dev/null || true; sleep 0.4; fi
  if kill -0 "${pid}" 2>/dev/null; then kill -TERM -- "-${pid}" 2>/dev/null || true; sleep 0.4; fi
  if kill -0 "${pid}" 2>/dev/null; then kill -KILL -- "-${pid}" 2>/dev/null || true; fi
  wait "${pid}" 2>/dev/null || true
}
cleanup() {
  kill_group "${AUDIT_PID}"; AUDIT_PID=""
  kill_group "${GUARD_PID}"; GUARD_PID=""
  kill_group "${CONTROL_PID}"; CONTROL_PID=""
  kill_group "${GEN_PID}"; GEN_PID=""
  kill_group "${GAZEBO_PID}"; GAZEBO_PID=""
  for name in gzclient gzserver rviz rosmaster roscore roslaunch; do pkill -TERM -x "${name}" 2>/dev/null || true; done
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
[ "${READY}" = "1" ] || { echo "[ERROR] Gazebo joint state timeout"; exit 1; }

setsid bash -lc "source '${CONDA_SH}'; conda activate '${NCDF_ENV}'; cd '${REPO}'; source devel/setup.bash; exec python -u src/care_visibility_cdf/scripts/vbc_deadline_waypoint_node.py _device:=cpu _safety_margin_s:=${SAFETY_MARGIN} _projection_iters:=10 _projection_damping:=0.5 _projection_epsilon_f:=0.03 _projection_max_step_norm:=0.25 _root_refine_iters:=12 _root_tolerance_f:=0.002 _ascent_steps:=1 _ascent_step_size:=0.05 _ascent_max_step_norm:=0.25 _output_root:='${OUT}/projector_traces'" \
  > "${LOG}/waypoint_generator.log" 2>&1 &
GEN_PID=$!

READY=0
for _ in $(seq 1 120); do
  if grep -q "frozen model READY" "${LOG}/waypoint_generator.log" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 "${GEN_PID}" 2>/dev/null; then break; fi
  sleep 1
done
[ "${READY}" = "1" ] || { echo "[ERROR] projector not ready"; exit 1; }

setsid roslaunch egocentric_arm_planner phaseB2_vbc_waypoint_controlled.launch \
  waypoint_weight:="${CARE_WEIGHT}" \
  recovery_enabled:=true recovery_weight_scale:=1.0 \
  predictive_recovery_enabled:=false \
  enable_predicted_vbc_auditor:=true \
  predicted_vbc_audit_warn_ms:="${AUDIT_WARN_MS}" \
  predicted_vbc_recovery_enabled:=true \
  predicted_vbc_recovery_consecutive_violations:="${C4_STREAK}" \
  predicted_vbc_recovery_input_timeout:="${C4_INPUT_TIMEOUT}" \
  vbc_min_margin_s:="${SAFETY_MARGIN}" \
  trial_label:="${CASE_ID}_c4_2" \
  log_output_root:="${OUT}" \
  use_trajectory_visualizer:=false \
  goal_x:="${GX}" goal_y:="${GY}" goal_z:="${GZ}" \
  goal_qx:="${GQX}" goal_qy:="${GQY}" goal_qz:="${GQZ}" goal_qw:="${GQW}" \
  > "${LOG}/controlled.log" 2>&1 &
CONTROL_PID=$!

READY=0
for _ in $(seq 1 160); do
  if rosnode list 2>/dev/null | grep -q '^/predicted_vbc_recovery_guard$' && \
     rosnode list 2>/dev/null | grep -q '^/predicted_vbc_auditor_node$'; then READY=1; break; fi
  sleep 0.1
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] C4.2 guard/auditor did not start"
  tail -n 100 "${LOG}/controlled.log" || true
  exit 1
fi

setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p /care_planner/execution/predicted_vbc_audit_summary" \
  > "${OUT}/predicted_vbc_audit.csv" 2>&1 &
AUDIT_PID=$!
setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p /care_planner/execution/predicted_vbc_recovery_summary" \
  > "${OUT}/predicted_vbc_recovery_guard.csv" 2>&1 &
GUARD_PID=$!

RELEASED=0
for _ in $(seq 1 300); do
  S="$(timeout 2 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then RELEASED=1; break; fi
  sleep 0.1
done
[ "${RELEASED}" = "1" ] || { echo "[ERROR] initial gate release timeout"; exit 1; }

echo "[RUN] ${CASE_ID}: C4.2 closed-loop for ${RUN_SECONDS}s; audit -> ${C4_STREAK}-cycle trigger -> Recovery"
sleep "${RUN_SECONDS}"

kill_group "${AUDIT_PID}"; AUDIT_PID=""
kill_group "${GUARD_PID}"; GUARD_PID=""

python3 - "${CASE_ID}" "${OUT}" "${LOG}/controlled.log" "${C4_STREAK}" <<'PY'
import csv,glob,json,math,os,re,sys
cid,out,controlled,streak=sys.argv[1:]; streak=int(streak)

def records(path):
    ans=[]
    if not os.path.isfile(path): return ans
    with open(path,newline='',errors='replace') as fh:
        rd=csv.reader(fh); h=next(rd,[])
        if not h: return ans
        di=h.index('field.data') if 'field.data' in h else 1
        for row in rd:
            if len(row)<=di: continue
            tok=dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)',row[di]))
            if tok: ans.append(tok)
    return ans

audit=records(os.path.join(out,'predicted_vbc_audit.csv'))
guard=records(os.path.join(out,'predicted_vbc_recovery_guard.csv'))
active_audit=[r for r in audit if r.get('status') not in ('inactive','waiting_target') and 'audit_ms' in r]
viol=[r for r in active_audit if r.get('violation')=='1']
hold_records=[r for r in guard if r.get('verification_hold')=='1']

def f(v):
    try:
        x=float(v); return x if math.isfinite(x) else None
    except: return None

times=[f(r.get('audit_ms')) for r in active_audit]; times=[x for x in times if x is not None]
lastg=guard[-1] if guard else {}
text=open(controlled,errors='replace').read() if os.path.isfile(controlled) else ''
recovery_entries=len(re.findall(
    r'(?:entering VISIBILITY RECOVERY|VBC RECOVERY TRIGGERED WHILE UNSEEN)', text))
trigger_logs=len(re.findall(r'PREDICTED VBC RECOVERY TRIGGER',text))

exe={}
paths=sorted(glob.glob(os.path.join(out,'executed_vbc_*.json')))
if paths:
    try: exe=json.load(open(paths[-1]))
    except: pass
wp={}
paths=sorted(glob.glob(os.path.join(out,'*_weight_*','summary.json')))
if paths:
    try: wp=json.load(open(paths[-1]))
    except: pass
gate={}
paths=sorted(glob.glob(os.path.join(out,'execution_gate_*.json')))
if paths:
    try: gate=json.load(open(paths[-1]))
    except: pass

first_recovery_delay=f(
    wp.get('first_recovery_delay_from_target_s', wp.get('first_recovery_delay_s')))
margin=f(exe.get('executed_vbc_margin_s'))
see=f(exe.get('see_delay_from_target_s'))
sweep=f(exe.get('sweep_delay_from_target_s'))
if margin is not None:
    outcome='safe_margin' if margin+1e-9>=0.30 else 'unsafe_margin'
elif see is not None and sweep is None:
    outcome='safe_by_avoidance'
elif sweep is not None and see is None:
    outcome='unsafe_unseen_sweep'
else:
    outcome='inconclusive'

payload={
  'case_id':cid,
  'c4_streak_required':streak,
  'num_active_audits':len(active_audit),
  'num_violation_audits':len(viol),
  'audit_violation_observed':bool(viol),
  'audit_ms_mean':sum(times)/len(times) if times else None,
  'audit_ms_max':max(times) if times else None,
  'verification_hold_observed':bool(hold_records),
  'verification_hold_summary_records':len(hold_records),
  'guard_triggered':trigger_logs>0 or lastg.get('trigger_count_total','0') not in ('0',None),
  'guard_trigger_count_total':int(lastg.get('trigger_count_total','0')) if lastg else 0,
  'guard_last_trigger_lead_s':f(lastg.get('last_trigger_lead_s')),
  'guard_last_routing':lastg.get('routing'),
  'recovery_entered':recovery_entries>0,
  'recovery_entry_log_count':recovery_entries,
  'first_recovery_delay_s':first_recovery_delay,
  'first_recovery_delay_from_target_s':first_recovery_delay,
  'executed_vbc_margin_s':margin,
  'executed_see_delay_s':see,
  'executed_sweep_delay_s':sweep,
  'executed_outcome':outcome,
  'min_clearance_all_m':f(exe.get('min_clearance_all_m')),
  'replan_count':gate.get('replan_count'),
}
json.dump(payload,open(os.path.join(out,'c4_2_summary.json'),'w'),indent=2)
print(json.dumps(payload,indent=2))
PY

echo "[RESULT] ${OUT}/c4_2_summary.json"
echo "[AUDIT]  ${OUT}/predicted_vbc_audit.csv"
echo "[GUARD]  ${OUT}/predicted_vbc_recovery_guard.csv"
