#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
OUT="${OUT:-${REPO}/outputs/phase_c2_vbc_benchmark}"
LOG="${LOG:-${REPO}/logs/phase_c2_vbc_benchmark}"
ARCHIVE="${ARCHIVE:-${REPO}/phase_c2_vbc_benchmark_12x2.zip}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
BASELINE_WEIGHT="${BASELINE_WEIGHT:-0.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
POST_RELEASE_SECONDS="${POST_RELEASE_SECONDS:-4.0}"
INITIAL_Q_TOL="${INITIAL_Q_TOL:-0.02}"
TARGET_TOL_M="${TARGET_TOL_M:-0.001}"
SWEEP_TOL_S="${SWEEP_TOL_S:-0.03}"
PROJECTOR_Q_TOL="${PROJECTOR_Q_TOL:-0.005}"

cd "${REPO}" || exit 1
source devel/setup.bash

# Phase C2 is a reproducible benchmark. Never mix partial/old C2 runs.
rm -rf "${OUT}" "${LOG}" "${ARCHIVE}"
mkdir -p "${OUT}" "${LOG}"
exec > >(tee "${LOG}/batch_console.log") 2>&1

if [ ! -f "${CASE_FILE}" ]; then
  echo "[ERROR] frozen case file not found: ${CASE_FILE}"
  exit 1
fi
cp "${CASE_FILE}" "${OUT}/frozen_cases.json"

GIT_COMMIT="$(git rev-parse HEAD)"
GIT_BRANCH="$(git branch --show-current)"
python3 - "${OUT}/benchmark_metadata.json" "${CASE_FILE}" "${GIT_COMMIT}" "${GIT_BRANCH}" "${BASELINE_WEIGHT}" "${CARE_WEIGHT}" "${SAFETY_MARGIN}" <<'PYMETA'
import json,sys,time
out,case_file,commit,branch,w0,w1,margin=sys.argv[1:]
p={
  'created_wall_time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  'case_file':case_file,
  'git_commit':commit,
  'git_branch':branch,
  'baseline_weight':float(w0),
  'careplanner_weight':float(w1),
  'safety_margin_s':float(margin),
  'gazebo_gui':False,
  'rviz':False,
  'paired_design':'same frozen case, fresh headless Gazebo each run, identical precompute/gate, only waypoint weight differs'
}
json.dump(p,open(out,'w'),indent=2)
PYMETA

python3 - "${CASE_FILE}" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
ids=d.get('selected_case_ids',[])
cases=d.get('cases',[])
if len(ids)!=12 or len(cases)!=12:
    raise SystemExit(f"expected 12 frozen cases, got ids={len(ids)} cases={len(cases)}")
if set(ids)!={c.get('case_id') for c in cases}:
    raise SystemExit('selected_case_ids and cases disagree')
if abs(float(d.get('waypoint_weight_careplanner',3000.0))-3000.0)>1e-9:
    raise SystemExit('frozen benchmark CAREPlanner weight is not 3000')
print('[CHECK] frozen benchmark contains 12 cases:',ids)
PY
if [ $? -ne 0 ]; then exit 1; fi

# The batch must own the ROS master. Never kill a user's pre-existing ROS session.
if timeout 2 rosnode list >/dev/null 2>&1; then
  echo "[ERROR] A ROS master is already running. Close other ROS/Gazebo launches before Phase C2."
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

mapfile -t CASE_IDS < <(python3 - "${CASE_FILE}" <<'PY'
import json,sys
print('\n'.join(json.load(open(sys.argv[1]))['selected_case_ids']))
PY
)

case_fields() {
  python3 - "${CASE_FILE}" "$1" <<'PY'
import json,sys
p,cid=sys.argv[1:3]
d=json.load(open(p)); c=next(x for x in d['cases'] if x['case_id']==cid)
vals=[cid,c['difficulty_bin'],*c['goal_position'],*c['goal_orientation'],*c['selected_target_xyz'],c['nominal_sweep_time_s']]
print('\t'.join(str(x) for x in vals))
PY
}

kill_group() {
  local pid="${1:-}"
  if [ -z "${pid}" ]; then return 0; fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || true
    for _ in $(seq 1 40); do
      if ! kill -0 "${pid}" 2>/dev/null; then break; fi
      sleep 0.1
    done
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 1
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

# Safe only after the initial no-master precondition: from this point the batch
# owns every ROS/Gazebo process it starts. Exact process names avoid broad pkill.
cleanup_owned_ros_fallback() {
  local name
  for name in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${name}" 2>/dev/null || true
  done
  sleep 1
  for name in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${name}" 2>/dev/null || true
  done
}

GAZEBO_PID=""
GEN_PID=""
CONTROL_PID=""
cleanup_run() {
  kill_group "${CONTROL_PID}"; CONTROL_PID=""
  kill_group "${GEN_PID}"; GEN_PID=""
  kill_group "${GAZEBO_PID}"; GAZEBO_PID=""

  for _ in $(seq 1 30); do
    if ! timeout 1 rosnode list >/dev/null 2>&1; then return 0; fi
    sleep 0.1
  done

  echo "[WARN] ROS master still reachable after process-group cleanup; applying owned-process fallback"
  cleanup_owned_ros_fallback
  for _ in $(seq 1 30); do
    if ! timeout 1 rosnode list >/dev/null 2>&1; then return 0; fi
    sleep 0.1
  done
  echo "[ERROR] ROS master is still reachable after forced cleanup"
  return 1
}
trap cleanup_run EXIT
trap 'cleanup_run; exit 130' INT TERM

wait_joint_state() {
  local out="$1"
  for _ in $(seq 1 120); do
    if timeout 2 rostopic echo -n 1 /care_arm/joint_states > "${out}" 2>/dev/null; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

validate_initial_q() {
  local js_file="$1" cid="$2" out_json="$3"
  python3 - "${CASE_FILE}" "${cid}" "${js_file}" "${out_json}" "${INITIAL_Q_TOL}" <<'PY'
import json,sys,yaml
case_file,cid,js_file,out_file,tol=sys.argv[1:]
tol=float(tol)
d=json.load(open(case_file)); c=next(x for x in d['cases'] if x['case_id']==cid)
docs=[x for x in yaml.safe_load_all(open(js_file)) if isinstance(x,dict)]
if not docs: raise SystemExit('no JointState YAML')
m=docs[0]; names=list(m.get('name',[])); pos=list(m.get('position',[])); idx={n:i for i,n in enumerate(names)}
joints=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
q=[float(pos[idx[n]]) for n in joints]
q0=[float(x) for x in c['initial_q']]
err=max(abs(a-b) for a,b in zip(q,q0))
payload={'case_id':cid,'expected_initial_q':q0,'measured_initial_q':q,'error_inf_rad':err,'tolerance_rad':tol,'passed':err<=tol}
json.dump(payload,open(out_file,'w'),indent=2)
print(f"[CHECK] {cid} initial_q error_inf={err:.6g} rad (tol={tol})")
raise SystemExit(0 if err<=tol else 2)
PY
}

wait_frozen_summary() {
  local out="$1"
  for _ in $(seq 1 200); do
    if timeout 2 rostopic echo -n 1 /phase_b2_controlled_trial/summary > "${out}" 2>/dev/null; then
      if grep -q "frozen_target=" "${out}"; then return 0; fi
    fi
    sleep 0.1
  done
  return 1
}

validate_runtime_case() {
  local summary_file="$1" cid="$2" out_json="$3"
  python3 - "${CASE_FILE}" "${cid}" "${summary_file}" "${out_json}" "${TARGET_TOL_M}" "${SWEEP_TOL_S}" <<'PY'
import json,re,sys
case_file,cid,summary_file,out_file,target_tol,sweep_tol=sys.argv[1:]
target_tol=float(target_tol); sweep_tol=float(sweep_tol)
d=json.load(open(case_file)); c=next(x for x in d['cases'] if x['case_id']==cid)
text=open(summary_file).read()
m=re.search(r'frozen_target=\[([^\]]+)\].*?frozen_sweep_time_s=([+\-0-9.eE]+)',text,re.S)
if not m: raise SystemExit('could not parse controlled-trial frozen summary')
actual=[float(x) for x in m.group(1).split(',')]; actual_sweep=float(m.group(2))
expected=[float(x) for x in c['selected_target_xyz']]; expected_sweep=float(c['nominal_sweep_time_s'])
target_err=max(abs(a-b) for a,b in zip(actual,expected)); sweep_err=abs(actual_sweep-expected_sweep)
passed=target_err<=target_tol and sweep_err<=sweep_tol
payload={'case_id':cid,'expected_target_xyz':expected,'actual_target_xyz':actual,'target_error_inf_m':target_err,'target_tolerance_m':target_tol,'expected_sweep_time_s':expected_sweep,'actual_sweep_time_s':actual_sweep,'sweep_error_s':sweep_err,'sweep_tolerance_s':sweep_tol,'passed':passed}
json.dump(payload,open(out_file,'w'),indent=2)
print(f"[CHECK] {cid} frozen target err={target_err:.3g} m (tol={target_tol}), sweep err={sweep_err:.3g} s (tol={sweep_tol})")
raise SystemExit(0 if passed else 2)
PY
}

validate_projector() {
  local run_dir="$1" cid="$2" out_json="$3"
  python3 - "${CASE_FILE}" "${cid}" "${run_dir}" "${out_json}" "${TARGET_TOL_M}" "${SWEEP_TOL_S}" "${PROJECTOR_Q_TOL}" <<'PYPROJ'
import glob,json,os,sys
case_file,cid,run_dir,out_file,target_tol,sweep_tol,q_tol=sys.argv[1:]
target_tol=float(target_tol); sweep_tol=float(sweep_tol); q_tol=float(q_tol)
d=json.load(open(case_file)); c=next(x for x in d['cases'] if x['case_id']==cid)
paths=sorted(glob.glob(os.path.join(run_dir,'projector_traces','vbc_visibility_waypoint_*.json')),key=os.path.getmtime)
if not paths: raise SystemExit('no runtime projector trace')
t=json.load(open(paths[-1]))
def inferr(a,b): return max(abs(float(x)-float(y)) for x,y in zip(a,b))
errs={
  'target_error_inf_m':inferr(t['target_xyz'],c['selected_target_xyz']),
  'sweep_error_s':abs(float(t['nominal_sweep_time_s'])-float(c['nominal_sweep_time_s'])),
  'q_nom_error_inf_rad':inferr(t['q_deadline_nominal'],c['q_nom_deadline']),
  'q_vis_error_inf_rad':inferr(t['q_vis'],c['q_vis'])
}
passed=(errs['target_error_inf_m']<=target_tol and errs['sweep_error_s']<=sweep_tol and errs['q_nom_error_inf_rad']<=q_tol and errs['q_vis_error_inf_rad']<=q_tol)
payload={'case_id':cid,'trace_file':paths[-1],**errs,'passed':passed,'tolerances':{'target_m':target_tol,'sweep_s':sweep_tol,'q_nom_rad':q_tol,'q_vis_rad':q_tol}}
json.dump(payload,open(out_file,'w'),indent=2)
print('[CHECK] %s projector target_err=%.3g m sweep_err=%.3g s q_nom_err=%.3g rad q_vis_err=%.3g rad' % (cid,errs['target_error_inf_m'],errs['sweep_error_s'],errs['q_nom_error_inf_rad'],errs['q_vis_error_inf_rad']))
raise SystemExit(0 if passed else 2)
PYPROJ
}

wait_gate_release() {
  local out="$1"
  for _ in $(seq 1 300); do
    local tmp
    tmp="$(timeout 2 rostopic echo -n 1 /care_planner/execution/gate_summary 2>/dev/null || true)"
    if echo "${tmp}" | grep -q "released=1"; then
      printf '%s\n' "${tmp}" > "${out}"
      return 0
    fi
    sleep 0.1
  done
  return 1
}

write_run_status() {
  local run_dir="$1" cid="$2" difficulty="$3" mode="$4" weight="$5" status="$6" message="$7"
  python3 - "${run_dir}" "${cid}" "${difficulty}" "${mode}" "${weight}" "${status}" "${message}" <<'PY'
import json,sys,time
run_dir,cid,difficulty,mode,weight,status,message=sys.argv[1:]
p={'case_id':cid,'difficulty_bin':difficulty,'mode':mode,'waypoint_weight':float(weight),'status':status,'message':message,'wall_time':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
json.dump(p,open(run_dir+'/run_status.json','w'),indent=2)
PY
}

FAILED=0
TOTAL=0
for CASE_ID in "${CASE_IDS[@]}"; do
  IFS=$'\t' read -r CID DIFFICULTY GX GY GZ GQX GQY GQZ GQW TX TY TZ SWEEP <<< "$(case_fields "${CASE_ID}")"

  for MODE in baseline careplanner; do
    TOTAL=$((TOTAL+1))
    if [ "${MODE}" = "baseline" ]; then WEIGHT="${BASELINE_WEIGHT}"; else WEIGHT="${CARE_WEIGHT}"; fi
    RUN_NAME="${CID}_${DIFFICULTY}_${MODE}"
    RUN_OUT="${OUT}/runs/${CID}/${MODE}"
    RUN_LOG="${LOG}/runs/${CID}/${MODE}"
    mkdir -p "${RUN_OUT}/projector_traces" "${RUN_LOG}"

    echo
    echo "================================================================"
    echo "[RUN ${TOTAL}/24] ${CID} difficulty=${DIFFICULTY} mode=${MODE} weight=${WEIGHT}"
    echo "[RUN] goal=[${GX}, ${GY}, ${GZ}] expected_target=[${TX}, ${TY}, ${TZ}] sweep=${SWEEP}"
    echo "================================================================"

    STATUS="ok"; MESSAGE="completed"

    if timeout 2 rosnode list >/dev/null 2>&1; then
      STATUS="preexisting_ros_master"; MESSAGE="ROS master was still running before run"
    fi

    if [ "${STATUS}" = "ok" ]; then
      # Headless/no-RViz is deliberate for batch reproducibility and clean teardown.
      setsid roslaunch arm_description gazebo_velocity_control.launch \
        gazebo_gui:=false use_rviz:=false \
        > "${RUN_LOG}/gazebo.log" 2>&1 &
      GAZEBO_PID=$!
      if ! wait_joint_state "${RUN_OUT}/initial_joint_state.yaml"; then
        STATUS="gazebo_joint_state_timeout"; MESSAGE="no /care_arm/joint_states after Gazebo startup"
      fi
    fi

    if [ "${STATUS}" = "ok" ]; then
      if ! validate_initial_q "${RUN_OUT}/initial_joint_state.yaml" "${CID}" "${RUN_OUT}/initial_q_validation.json"; then
        STATUS="initial_q_mismatch"; MESSAGE="Gazebo initial configuration does not match frozen case q0"
      fi
    fi

    if [ "${STATUS}" = "ok" ]; then
      setsid bash -lc "source '${CONDA_SH}'; conda activate '${NCDF_ENV}'; cd '${REPO}'; source devel/setup.bash; exec python -u src/care_visibility_cdf/scripts/vbc_deadline_waypoint_node.py _device:=cpu _safety_margin_s:=${SAFETY_MARGIN} _projection_iters:=10 _projection_damping:=0.5 _projection_epsilon_f:=0.03 _projection_max_step_norm:=0.25 _root_refine_iters:=12 _root_tolerance_f:=0.002 _ascent_steps:=1 _ascent_step_size:=0.05 _ascent_max_step_norm:=0.25 _output_root:='${RUN_OUT}/projector_traces'" > "${RUN_LOG}/waypoint_generator.log" 2>&1 &
      GEN_PID=$!
      READY=0
      for _ in $(seq 1 120); do
        if grep -q "frozen model READY" "${RUN_LOG}/waypoint_generator.log" 2>/dev/null; then READY=1; break; fi
        if ! kill -0 "${GEN_PID}" 2>/dev/null; then break; fi
        sleep 1
      done
      if [ "${READY}" != "1" ]; then
        STATUS="waypoint_generator_not_ready"; MESSAGE="VisCDF waypoint generator failed to become ready"
      fi
    fi

    if [ "${STATUS}" = "ok" ]; then
      setsid roslaunch egocentric_arm_planner phaseB2_vbc_waypoint_controlled.launch \
        waypoint_weight:="${WEIGHT}" \
        vbc_min_margin_s:="${SAFETY_MARGIN}" \
        trial_label:="${RUN_NAME}" \
        log_output_root:="${RUN_OUT}" \
        goal_x:="${GX}" goal_y:="${GY}" goal_z:="${GZ}" \
        goal_qx:="${GQX}" goal_qy:="${GQY}" goal_qz:="${GQZ}" goal_qw:="${GQW}" \
        > "${RUN_LOG}/controlled_launch.log" 2>&1 &
      CONTROL_PID=$!

      if ! wait_frozen_summary "${RUN_OUT}/controlled_trial_summary.yaml"; then
        STATUS="frozen_target_timeout"; MESSAGE="controlled trial did not freeze a VBC target"
      elif ! validate_runtime_case "${RUN_OUT}/controlled_trial_summary.yaml" "${CID}" "${RUN_OUT}/runtime_case_validation.json"; then
        STATUS="frozen_case_mismatch"; MESSAGE="runtime VBC target/sweep differs from frozen benchmark case"
      elif ! wait_gate_release "${RUN_OUT}/gate_release_summary.yaml"; then
        STATUS="gate_release_timeout"; MESSAGE="execution gate did not release"
      else
        echo "[RUN] gate released; recording ${POST_RELEASE_SECONDS}s of executed motion"
        sleep "${POST_RELEASE_SECONDS}"
      fi
    fi

    # Stop controller/loggers first so they flush JSON/CSV, then generator/Gazebo.
    if ! cleanup_run; then
      if [ "${STATUS}" = "ok" ]; then
        STATUS="cleanup_failed"; MESSAGE="ROS master/processes survived run cleanup"
      fi
    fi

    if [ "${STATUS}" = "ok" ]; then
      if ! validate_projector "${RUN_OUT}" "${CID}" "${RUN_OUT}/runtime_projector_validation.json"; then
        STATUS="runtime_projector_mismatch"; MESSAGE="runtime projector does not reproduce the frozen Phase-C1 case within configured tolerances"
      elif ! compgen -G "${RUN_OUT}/executed_vbc_*.json" >/dev/null; then
        STATUS="missing_executed_vbc_summary"; MESSAGE="executed-VBC logger JSON missing"
      elif ! compgen -G "${RUN_OUT}/execution_gate_*.json" >/dev/null; then
        STATUS="missing_gate_trace"; MESSAGE="execution gate JSON missing"
      elif ! compgen -G "${RUN_OUT}/*_weight_*/summary.json" >/dev/null; then
        STATUS="missing_mpc_summary"; MESSAGE="waypoint MPC summary JSON missing"
      fi
    fi

    write_run_status "${RUN_OUT}" "${CID}" "${DIFFICULTY}" "${MODE}" "${WEIGHT}" "${STATUS}" "${MESSAGE}"
    if [ "${STATUS}" != "ok" ]; then
      FAILED=$((FAILED+1))
      echo "[FAIL] ${CID}/${MODE}: ${STATUS} - ${MESSAGE}"
    else
      echo "[DONE] ${CID}/${MODE}"
    fi
    sleep 1
  done
done
trap - EXIT INT TERM

python3 scripts/summarize_phase_c2_vbc_benchmark.py \
  --case-file "${CASE_FILE}" \
  --output-root "${OUT}"

python3 - "${OUT}" "${LOG}" "${ARCHIVE}" <<'PYPACK'
import os,sys,zipfile
out,log,archive=sys.argv[1:]
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for root,label in ((out,'outputs/phase_c2_vbc_benchmark'),(log,'logs/phase_c2_vbc_benchmark')):
        for base,_,files in os.walk(root):
            for name in files:
                p=os.path.join(base,name)
                z.write(p,os.path.join(label,os.path.relpath(p,root)))
print('[PACKAGE]',archive)
PYPACK

echo
printf '[SUMMARY] completed=%d failed=%d total=%d\n' "$((TOTAL-FAILED))" "${FAILED}" "${TOTAL}"
echo "[SUMMARY] table: ${OUT}/benchmark_runs.csv"
echo "[SUMMARY] paired: ${OUT}/paired_summary.csv"
echo "[SUMMARY] json: ${OUT}/benchmark_summary.json"
echo "[SUMMARY] zip: ${ARCHIVE}"

if [ "${FAILED}" -ne 0 ]; then
  exit 2
fi
