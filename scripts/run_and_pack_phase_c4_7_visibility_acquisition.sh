#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-10.0}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.20}"

OUT="${OUT:-${REPO}/outputs/phase_c4_7_visibility_acquisition/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_7_visibility_acquisition/${CASE_ID}}"
ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C4_7_visibility_acquisition_${CASE_ID}.zip}"

cd "${REPO}"
echo "[C4.7] branch: $(git branch --show-current)"
echo "[C4.7] head:   $(git rev-parse HEAD)"
echo "[C4.7] case=${CASE_ID} run=${RUN_SECONDS}s weight=${CARE_WEIGHT}"

catkin build
source devel/setup.bash
python3 -m py_compile \
  src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py \
  src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py \
  src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py

rm -f "${ZIP_PATH}"

# Extra C4.7 recorders start only after the base runner has created ROS master;
# therefore its initial rm -rf OUT/LOG cannot delete these output files.
EXTRA_PIDS=()
wait_record() {
  local topic="$1"; local path="$2"
  setsid bash -lc "
    source '${REPO}/devel/setup.bash'
    while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done
    mkdir -p '$(dirname "${path}")'
    exec rostopic echo -p '${topic}' > '${path}' 2>&1
  " >/dev/null 2>&1 &
  EXTRA_PIDS+=("$!")
}
kill_extra() {
  local pid
  for pid in "${EXTRA_PIDS[@]:-}"; do
    kill -INT -- "-${pid}" 2>/dev/null || true
    sleep 0.05
    kill -TERM -- "-${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
  EXTRA_PIDS=()
}
trap kill_extra EXIT INT TERM

wait_record "/care_planner/active_sensing/visibility_acquisition_summary" \
  "${OUT}/visibility_acquisition_summary.csv"
wait_record "/care_planner/active_sensing/visibility_acquisition_complete" \
  "${OUT}/visibility_acquisition_complete.csv"

CASE_ID="${CASE_ID}" \
RUN_SECONDS="${RUN_SECONDS}" \
CARE_WEIGHT="${CARE_WEIGHT}" \
SAFETY_MARGIN="${SAFETY_MARGIN}" \
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT}" \
REGION_SCHEDULE_MODE="visibility_acquisition" \
OUT="${OUT}" \
LOG="${LOG}" \
bash scripts/run_phase_c4_4_verified_regime_smoke.sh

kill_extra
trap - EXIT INT TERM

mkdir -p "${OUT}"
{
  echo "branch=$(git branch --show-current)"
  echo "head=$(git rev-parse HEAD)"
  echo "case_id=${CASE_ID}"
  echo "run_seconds=${RUN_SECONDS}"
  echo "care_weight=${CARE_WEIGHT}"
  echo "safety_margin_s=${SAFETY_MARGIN}"
  echo "prediction_timeout_s=${PREDICTION_TIMEOUT}"
  echo "region_schedule_mode=visibility_acquisition"
  echo "repair_nominal_deadline_used=false"
  echo "repair_completion=actual_confidence"
} > "${OUT}/c4_7_run_metadata.txt"

MODE_OK=0
GATE_OK=0
if grep -q "region_schedule_mode=visibility_acquisition" "${LOG}/waypoint_generator.log" 2>/dev/null; then
  MODE_OK=1
else
  echo "[WARN] C4.7 mode marker not found"
fi
if grep -q "repair_completion_gate_enabled=1" "${OUT}/regime_summary.csv" 2>/dev/null; then
  GATE_OK=1
else
  echo "[WARN] regime completion gate marker not observed"
fi

python3 - "${OUT}" "${MODE_OK}" "${GATE_OK}" <<'PY'
import csv,json,math,os,re,statistics,sys
out,mode_ok,gate_ok=sys.argv[1:]
TOK=re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')

def token_rows(name):
 p=os.path.join(out,name); rows=[]
 if not os.path.isfile(p): return rows
 with open(p,newline='',errors='replace') as f:
  rd=csv.reader(f); h=next(rd,[])
  if not h:return rows
  di=h.index('field.data') if 'field.data' in h else 1
  for r in rd:
   if len(r)<=di:continue
   d=dict(TOK.findall(','.join(r[di:])))
   if d:rows.append(d)
 return rows

def as_i(v,d=0):
 try:return int(float(v))
 except:return d

reg=token_rows('regime_summary.csv')
acq=token_rows('visibility_acquisition_summary.csv')
base={}
p=os.path.join(out,'c4_4_verified_regime_summary.json')
if os.path.isfile(p): base=json.load(open(p))
last_reg=reg[-1] if reg else {}
last_acq=acq[-1] if acq else {}

# Joint displacement from actual measured state.
max_joint_range=None
jp=os.path.join(out,'joint_states.csv')
if os.path.isfile(jp):
 with open(jp,newline='',errors='replace') as f:
  rd=csv.DictReader(f)
  cols=[f'field.position{i}' for i in range(7)]
  lo=[math.inf]*7; hi=[-math.inf]*7
  for r in rd:
   for i,c in enumerate(cols):
    try:v=float(r[c])
    except:continue
    lo[i]=min(lo[i],v); hi[i]=max(hi[i],v)
  rr=[hi[i]-lo[i] for i in range(7) if math.isfinite(lo[i]) and math.isfinite(hi[i])]
  if rr:max_joint_range=max(rr)

payload={
 'region_schedule_mode':'visibility_acquisition',
 'mode_marker_ok':bool(int(mode_ok)),
 'repair_completion_gate_ok':bool(int(gate_ok)),
 'candidate_predicted_safe':base.get('candidate_safe_records'),
 'candidate_predicted_unsafe':base.get('candidate_unsafe_records'),
 'execution_unsafe':base.get('execution_unsafe_records'),
 'commit_count':base.get('commit_count'),
 'final_regime_state':base.get('final_regime_state'),
 'final_progress_phase_s':base.get('final_progress_phase_s'),
 'repair_entry_count':base.get('repair_entry_count'),
 'probe_entry_count':base.get('probe_entry_count'),
 'probe_failure_count':base.get('probe_failure_count'),
 'max_actual_joint_range_rad':max_joint_range,
 'repair_safe_commit_count':as_i(last_reg.get('repair_safe_commit_count'),0),
 'repair_completion_event_count':as_i(last_reg.get('repair_completion_event_count'),0),
 'acquisition_summary_records':len(acq),
 'acquisition_complete_ever':any(r.get('complete')=='1' for r in acq),
 'max_seen_obligation_count':max([as_i(r.get('seen_obligation_count'),0) for r in acq] or [0]),
 'last_remaining_obligation_count':as_i(last_acq.get('remaining_obligation_count'),-1) if last_acq else None,
 'last_acquisition_complete':as_i(last_acq.get('complete'),0) if last_acq else None,
}
json.dump(payload,open(os.path.join(out,'c4_7_visibility_acquisition_summary.json'),'w'),indent=2)
print(json.dumps(payload,indent=2))
PY

cd "${REPO}"
zip -r "${ZIP_PATH}" \
  "${OUT#${REPO}/}" \
  "${LOG#${REPO}/}"

echo ""
echo "[C4.7 COMPLETE]"
echo "[RESULT] ${OUT}/c4_7_visibility_acquisition_summary.json"
echo "[ZIP] ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
