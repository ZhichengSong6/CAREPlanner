#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-15.0}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.20}"
REPAIR_PREFIX_S="${REPAIR_PREFIX_S:-0.15}"
REPAIR_BRAKE_DT_S="${REPAIR_BRAKE_DT_S:-0.05}"
REPAIR_HOLD_S="${REPAIR_HOLD_S:-0.10}"

OUT="${OUT:-${REPO}/outputs/phase_c4_9_blocker_aware/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_9_blocker_aware/${CASE_ID}}"
ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C4_9_blocker_aware_${CASE_ID}.zip}"

cd "${REPO}"
echo "[C4.9] branch: $(git branch --show-current)"
echo "[C4.9] head:   $(git rev-parse HEAD)"

catkin build
source devel/setup.bash
python3 -m py_compile \
  src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py \
  src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py \
  src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py \
  src/egocentric_arm_planner/scripts/optimized_trajectory_continuity_node.py \
  src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py

rm -f "${ZIP_PATH}"
export C4_REPAIR_PREFIX_VERIFY=1
export C4_REPAIR_PREFIX_S="${REPAIR_PREFIX_S}"
export C4_REPAIR_BRAKE_DT_S="${REPAIR_BRAKE_DT_S}"
export C4_REPAIR_HOLD_S="${REPAIR_HOLD_S}"

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
wait_record "/care_planner/active_sensing/blocker_stack_summary" \
  "${OUT}/blocker_stack_summary.csv"

CASE_ID="${CASE_ID}" \
RUN_SECONDS="${RUN_SECONDS}" \
CARE_WEIGHT="${CARE_WEIGHT}" \
SAFETY_MARGIN="${SAFETY_MARGIN}" \
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT}" \
REGION_SCHEDULE_MODE="blocker_aware_acquisition" \
OUT="${OUT}" LOG="${LOG}" \
bash scripts/run_phase_c4_4_verified_regime_smoke.sh

kill_extra
trap - EXIT INT TERM
mkdir -p "${OUT}"

cat > "${OUT}/c4_9_run_metadata.txt" <<EOF
branch=$(git branch --show-current)
head=$(git rev-parse HEAD)
case_id=${CASE_ID}
run_seconds=${RUN_SECONDS}
region_schedule_mode=blocker_aware_acquisition
repair_prefix_verification_enabled=true
repair_execution_prefix_s=${REPAIR_PREFIX_S}
repair_brake_dt_s=${REPAIR_BRAKE_DT_S}
repair_hold_s=${REPAIR_HOLD_S}
blocker_push_max_sweep_s=0.30
blocker_confirmations=2
blocker_priority=earliest_vbc_layer_then_qvis_distance
repair_completion=actual_confidence
EOF

python3 - "${OUT}" <<'PY'
import csv,json,math,os,re,sys
out=sys.argv[1]; TOK=re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')
def rows(name):
 p=os.path.join(out,name); a=[]
 if not os.path.isfile(p): return a
 with open(p,newline='',errors='replace') as f:
  rd=csv.reader(f); h=next(rd,[])
  if not h:return a
  di=h.index('field.data') if 'field.data' in h else 1
  for r in rd:
   if len(r)>di:
    d=dict(TOK.findall(','.join(r[di:])))
    if d:a.append(d)
 return a
def ai(v,d=0):
 try:return int(float(v))
 except:return d
base={}; p=os.path.join(out,'c4_4_verified_regime_summary.json')
if os.path.isfile(p): base=json.load(open(p))
stack=rows('blocker_stack_summary.csv'); acq=rows('visibility_acquisition_summary.csv'); commit=rows('commit_summary.csv')
last_stack=stack[-1] if stack else {}; last_acq=acq[-1] if acq else {}
max_depth=0; targets=[]
for r in stack:
 s=r.get('stack','none'); depth=0 if s=='none' else len([x for x in s.split(':') if x])
 max_depth=max(max_depth,depth)
 t=ai(r.get('current_target_id'),-1)
 if t>=0 and (not targets or targets[-1]!=t): targets.append(t)
payload={
 'architecture':'C4.9 blocker-aware recursive visibility repair + C4.8 exact-VBC safe prefix',
 'candidate_safe':base.get('candidate_safe_records'),
 'candidate_unsafe':base.get('candidate_unsafe_records'),
 'execution_unsafe':base.get('execution_unsafe_records'),
 'commit_count':base.get('commit_count'),
 'final_regime_state':base.get('final_regime_state'),
 'probe_entry_count':base.get('probe_entry_count'),
 'max_seen_obligation_count':max([ai(r.get('seen_obligation_count')) for r in acq] or [0]),
 'last_remaining_obligation_count':ai(last_acq.get('remaining_obligation_count'),-1) if last_acq else None,
 'acquisition_complete_ever':any(r.get('complete')=='1' for r in acq),
 'blocker_push_count':max([ai(r.get('push_count')) for r in stack] or [0]),
 'blocker_pop_count':max([ai(r.get('pop_count')) for r in stack] or [0]),
 'blocker_cycle_block_count':max([ai(r.get('cycle_block_count')) for r in stack] or [0]),
 'max_stack_depth':max_depth,
 'target_transition_sequence':targets,
 'last_stack':last_stack.get('stack'),
 'last_current_target_id':ai(last_stack.get('current_target_id'),-1) if last_stack else None,
 'repair_prefix_safe_count':max([ai(r.get('repair_prefix_safe_count')) for r in commit] or [0]),
 'repair_prefix_unsafe_count':max([ai(r.get('repair_prefix_unsafe_count')) for r in commit] or [0]),
}
json.dump(payload,open(os.path.join(out,'c4_9_blocker_aware_summary.json'),'w'),indent=2)
print(json.dumps(payload,indent=2))
if not stack: raise SystemExit('[ERROR] blocker_stack_summary.csv has no records')
if not any(r.get('repair_prefix_enabled')=='1' for r in commit):
 raise SystemExit('[ERROR] C4.8 safe-prefix verification was not enabled')
PY

cd "${REPO}"
zip -r "${ZIP_PATH}" "${OUT#${REPO}/}" "${LOG#${REPO}/}"
echo "[C4.9 COMPLETE]"
echo "[RESULT] ${OUT}/c4_9_blocker_aware_summary.json"
echo "[ZIP] ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
