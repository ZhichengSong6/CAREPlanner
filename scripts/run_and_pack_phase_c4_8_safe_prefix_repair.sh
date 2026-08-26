#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-12.0}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.20}"
REPAIR_PREFIX_S="${REPAIR_PREFIX_S:-0.15}"
REPAIR_BRAKE_DT_S="${REPAIR_BRAKE_DT_S:-0.05}"
REPAIR_HOLD_S="${REPAIR_HOLD_S:-0.10}"

OUT="${OUT:-${REPO}/outputs/phase_c4_8_safe_prefix_repair/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_8_safe_prefix_repair/${CASE_ID}}"
ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C4_8_safe_prefix_repair_${CASE_ID}.zip}"

cd "${REPO}"
echo "[C4.8] branch: $(git branch --show-current)"
echo "[C4.8] head:   $(git rev-parse HEAD)"
echo "[C4.8] case=${CASE_ID} run=${RUN_SECONDS}s prefix=${REPAIR_PREFIX_S}s brake_dt=${REPAIR_BRAKE_DT_S}s hold=${REPAIR_HOLD_S}s"

# The C4.7 runner also builds, but compile the newly modified verifier explicitly
# before launching Gazebo so syntax errors fail immediately.
catkin build
source devel/setup.bash
python3 -m py_compile \
  src/egocentric_arm_planner/scripts/optimized_trajectory_continuity_node.py \
  src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py \
  src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py \
  src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py

rm -f "${ZIP_PATH}"

# Explicit opt-in. C4.7 and older baselines leave this disabled.
export C4_REPAIR_PREFIX_VERIFY=1
export C4_REPAIR_PREFIX_S="${REPAIR_PREFIX_S}"
export C4_REPAIR_BRAKE_DT_S="${REPAIR_BRAKE_DT_S}"
export C4_REPAIR_HOLD_S="${REPAIR_HOLD_S}"

CASE_ID="${CASE_ID}" \
RUN_SECONDS="${RUN_SECONDS}" \
CARE_WEIGHT="${CARE_WEIGHT}" \
SAFETY_MARGIN="${SAFETY_MARGIN}" \
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT}" \
OUT="${OUT}" \
LOG="${LOG}" \
ZIP_PATH="${ZIP_PATH}" \
bash scripts/run_and_pack_phase_c4_7_visibility_acquisition.sh

mkdir -p "${OUT}"
{
  echo "branch=$(git branch --show-current)"
  echo "head=$(git rev-parse HEAD)"
  echo "case_id=${CASE_ID}"
  echo "run_seconds=${RUN_SECONDS}"
  echo "care_weight=${CARE_WEIGHT}"
  echo "safety_margin_s=${SAFETY_MARGIN}"
  echo "region_schedule_mode=visibility_acquisition"
  echo "repair_prefix_verification_enabled=true"
  echo "repair_execution_prefix_s=${REPAIR_PREFIX_S}"
  echo "repair_brake_dt_s=${REPAIR_BRAKE_DT_S}"
  echo "repair_hold_s=${REPAIR_HOLD_S}"
  echo "verification_semantics=repair_prefix_plus_brake_plus_hold"
  echo "normal_verification_semantics=full_horizon"
} > "${OUT}/c4_8_run_metadata.txt"

python3 - "${OUT}" <<'PY'
import csv,json,math,os,re,sys
out=sys.argv[1]
TOK=re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')

def rows(name):
 p=os.path.join(out,name); a=[]
 if not os.path.isfile(p): return a
 with open(p,newline='',errors='replace') as f:
  rd=csv.reader(f); h=next(rd,[])
  if not h:return a
  di=h.index('field.data') if 'field.data' in h else 1
  for r in rd:
   if len(r)<=di:continue
   d=dict(TOK.findall(','.join(r[di:])))
   if d:a.append(d)
 return a

def ai(v,d=0):
 try:return int(float(v))
 except:return d

def af(v,d=math.nan):
 try:return float(v)
 except:return d

base={}
p=os.path.join(out,'c4_4_verified_regime_summary.json')
if os.path.isfile(p): base=json.load(open(p))
c7={}
p=os.path.join(out,'c4_7_visibility_acquisition_summary.json')
if os.path.isfile(p): c7=json.load(open(p))
commit=rows('commit_summary.csv')
last=commit[-1] if commit else {}

prefix_enabled=any(r.get('repair_prefix_enabled')=='1' for r in commit)
prefix_view_seen=any(
 r.get('last_verification_view')=='repair_prefix_brake_hold' or
 r.get('outstanding_view')=='repair_prefix_brake_hold'
 for r in commit)

payload={
 'architecture':'C4.8 reactive visibility acquisition with exact-VBC verified executable prefix + braking + hold tail',
 'repair_prefix_enabled_observed':prefix_enabled,
 'repair_prefix_view_observed':prefix_view_seen,
 'repair_prefix_build_count':max([ai(r.get('repair_prefix_build_count')) for r in commit] or [0]),
 'repair_prefix_safe_count':max([ai(r.get('repair_prefix_safe_count')) for r in commit] or [0]),
 'repair_prefix_unsafe_count':max([ai(r.get('repair_prefix_unsafe_count')) for r in commit] or [0]),
 'last_repair_prefix_duration_s':af(last.get('repair_prefix_duration_s')) if last else None,
 'last_repair_brake_duration_s':af(last.get('repair_brake_duration_s')) if last else None,
 'candidate_predicted_safe':base.get('candidate_safe_records'),
 'candidate_predicted_unsafe':base.get('candidate_unsafe_records'),
 'execution_unsafe':base.get('execution_unsafe_records'),
 'commit_count':base.get('commit_count'),
 'final_regime_state':base.get('final_regime_state'),
 'repair_entry_count':base.get('repair_entry_count'),
 'probe_entry_count':base.get('probe_entry_count'),
 'probe_failure_count':base.get('probe_failure_count'),
 'final_progress_phase_s':base.get('final_progress_phase_s'),
 'max_actual_joint_range_rad':c7.get('max_actual_joint_range_rad'),
 'max_seen_obligation_count':c7.get('max_seen_obligation_count'),
 'acquisition_complete_ever':c7.get('acquisition_complete_ever'),
 'repair_completion_event_count':c7.get('repair_completion_event_count'),
 'last_remaining_obligation_count':c7.get('last_remaining_obligation_count'),
}
json.dump(payload,open(os.path.join(out,'c4_8_safe_prefix_repair_summary.json'),'w'),indent=2)
print(json.dumps(payload,indent=2))

if not prefix_enabled:
 raise SystemExit('[ERROR] repair_prefix_enabled=1 was not observed in commit_summary.csv')
if not prefix_view_seen:
 raise SystemExit('[ERROR] repair_prefix_brake_hold verification view was not observed')
PY

# C4.7 runner created the archive before the C4.8-specific summary above. Update
# the same zip in place with the final C4.8 metadata/summary.
cd "${REPO}"
zip -r "${ZIP_PATH}" \
  "${OUT#${REPO}/}/c4_8_run_metadata.txt" \
  "${OUT#${REPO}/}/c4_8_safe_prefix_repair_summary.json" >/dev/null

echo ""
echo "[C4.8 COMPLETE]"
echo "[RESULT] ${OUT}/c4_8_safe_prefix_repair_summary.json"
echo "[ZIP] ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
