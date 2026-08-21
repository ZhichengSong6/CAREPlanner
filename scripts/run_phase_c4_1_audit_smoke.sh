#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-2.0}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
AUDIT_WARN_MS="${AUDIT_WARN_MS:-5.0}"
OUT="${OUT:-${REPO}/outputs/phase_c4_1_audit_smoke/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_1_audit_smoke/${CASE_ID}}"

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

GAZEBO_PID=""
GEN_PID=""
CONTROL_PID=""
AUDIT_ECHO_PID=""

kill_group() {
  local pid="${1:-}"
  [ -z "${pid}" ] && return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || true
    sleep 0.5
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 0.5
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  kill_group "${AUDIT_ECHO_PID}"; AUDIT_ECHO_PID=""
  kill_group "${CONTROL_PID}"; CONTROL_PID=""
  kill_group "${GEN_PID}"; GEN_PID=""
  kill_group "${GAZEBO_PID}"; GAZEBO_PID=""
  for name in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${name}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

setsid roslaunch arm_description gazebo_velocity_control.launch \
  gazebo_gui:=false use_rviz:=false \
  > "${LOG}/gazebo.log" 2>&1 &
GAZEBO_PID=$!

READY=0
for _ in $(seq 1 120); do
  if timeout 2 rostopic echo -n 1 /care_arm/joint_states >/dev/null 2>&1; then
    READY=1; break
  fi
  sleep 0.25
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] Gazebo joint state timeout"
  exit 1
fi

setsid bash -lc "source '${CONDA_SH}'; conda activate '${NCDF_ENV}'; cd '${REPO}'; source devel/setup.bash; exec python -u src/care_visibility_cdf/scripts/vbc_deadline_waypoint_node.py _device:=cpu _safety_margin_s:=${SAFETY_MARGIN} _projection_iters:=10 _projection_damping:=0.5 _projection_epsilon_f:=0.03 _projection_max_step_norm:=0.25 _root_refine_iters:=12 _root_tolerance_f:=0.002 _ascent_steps:=1 _ascent_step_size:=0.05 _ascent_max_step_norm:=0.25 _output_root:='${OUT}/projector_traces'" \
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
  echo "[ERROR] projector not ready"
  exit 1
fi

setsid roslaunch egocentric_arm_planner phaseB2_vbc_waypoint_controlled.launch \
  waypoint_weight:="${CARE_WEIGHT}" \
  recovery_enabled:=true recovery_weight_scale:=1.0 \
  predictive_recovery_enabled:=false \
  enable_predicted_vbc_auditor:=true \
  predicted_vbc_audit_warn_ms:="${AUDIT_WARN_MS}" \
  vbc_min_margin_s:="${SAFETY_MARGIN}" \
  trial_label:="${CASE_ID}_c4_1_audit" \
  log_output_root:="${OUT}" \
  use_trajectory_visualizer:=false \
  goal_x:="${GX}" goal_y:="${GY}" goal_z:="${GZ}" \
  goal_qx:="${GQX}" goal_qy:="${GQY}" goal_qz:="${GQZ}" goal_qw:="${GQW}" \
  > "${LOG}/controlled.log" 2>&1 &
CONTROL_PID=$!

READY=0
for _ in $(seq 1 120); do
  if rosnode list 2>/dev/null | grep -q '^/predicted_vbc_auditor_node$'; then
    READY=1; break
  fi
  sleep 0.1
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] predicted_vbc_auditor_node did not start"
  tail -n 80 "${LOG}/controlled.log" || true
  exit 1
fi

setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo /care_planner/execution/predicted_vbc_audit_summary" \
  > "${OUT}/predicted_vbc_audit.log" 2>&1 &
AUDIT_ECHO_PID=$!

RELEASED=0
for _ in $(seq 1 300); do
  S="$(timeout 2 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
  if echo "${S}" | grep -q "released=1"; then
    RELEASED=1; break
  fi
  sleep 0.1
done
if [ "${RELEASED}" != "1" ]; then
  echo "[ERROR] initial gate release timeout"
  exit 1
fi

echo "[RUN] ${CASE_ID}: C4.1 diagnostic observing for ${RUN_SECONDS}s; C3 predictive trigger is OFF"
sleep "${RUN_SECONDS}"

kill_group "${AUDIT_ECHO_PID}"; AUDIT_ECHO_PID=""

python3 - "${OUT}/predicted_vbc_audit.log" "${OUT}/audit_summary.json" <<'PY'
import json, math, re, sys
src,out=sys.argv[1:]
text=open(src,errors='replace').read()
records=[]
for m in re.finditer(r'data:\s*["\']?([^\n]+)', text):
    s=m.group(1).strip().strip('"\'')
    tok=dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)',s))
    if 'audit_ms' not in tok: continue
    def f(k):
        try:
            x=float(tok.get(k,'nan')); return x if math.isfinite(x) else None
        except: return None
    records.append({
        'status':tok.get('status'),
        'violation':int(tok.get('violation','0')),
        'predicted_seen':int(tok.get('predicted_seen','0')) if 'predicted_seen' in tok else None,
        'predicted_sweep':int(tok.get('predicted_sweep','0')) if 'predicted_sweep' in tok else None,
        'see_time_s':f('see_time_s'),'sweep_time_s':f('sweep_time_s'),'margin_s':f('margin_s'),
        'evaluated_q':int(tok.get('evaluated_q','0')),'prediction_q':int(tok.get('prediction_q','0')),
        'audit_ms':f('audit_ms')
    })
active=[r for r in records if r['audit_ms'] is not None and r['status'] not in ('inactive','waiting_target')]
times=[r['audit_ms'] for r in active]
viol=[r for r in active if r['violation']==1]
payload={
    'num_records':len(records),'num_active_audits':len(active),'num_violation_audits':len(viol),
    'violation_observed':bool(viol),
    'audit_ms_mean':sum(times)/len(times) if times else None,
    'audit_ms_max':max(times) if times else None,
    'evaluated_q_mean':sum(r['evaluated_q'] for r in active)/len(active) if active else None,
    'last_active_record':active[-1] if active else None,
}
json.dump(payload,open(out,'w'),indent=2)
print(json.dumps(payload,indent=2))
PY

echo "[RESULT] ${OUT}/audit_summary.json"
echo "[RAW] ${OUT}/predicted_vbc_audit.log"
echo "[NOTE] For C4.1 validation run this script once with CASE_ID=case_003 and once with CASE_ID=case_007."
