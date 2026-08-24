#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
OUT_ROOT="${OUT_ROOT:-${REPO}/outputs/phase_c4_2_frozen12_benchmark}"
LOG_ROOT="${LOG_ROOT:-${REPO}/logs/phase_c4_2_frozen12_benchmark}"
ARCHIVE="${ARCHIVE:-${REPO}/phase_c4_2_frozen12_benchmark.zip}"
LEGACY_C4_1_ROOT="${LEGACY_C4_1_ROOT:-${REPO}/outputs/phase_c4_1_frozen12_benchmark}"
RUN_SECONDS="${RUN_SECONDS:-6.0}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
C4_STREAK="${C4_STREAK:-2}"
C4_INPUT_TIMEOUT="${C4_INPUT_TIMEOUT:-0.25}"
AUDIT_WARN_MS="${AUDIT_WARN_MS:-5.0}"

cd "${REPO}" || exit 1
source devel/setup.bash

rm -rf "${OUT_ROOT}" "${LOG_ROOT}" "${ARCHIVE}"
mkdir -p "${OUT_ROOT}/runs" "${LOG_ROOT}"
cp "${CASE_FILE}" "${OUT_ROOT}/frozen_cases.json"

python3 - "${CASE_FILE}" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
ids=d.get('selected_case_ids',[])
cases=d.get('cases',[])
if len(ids)!=12 or len(cases)!=12:
    raise SystemExit(f'expected exactly 12 frozen cases, got ids={len(ids)} cases={len(cases)}')
if set(ids)!={c.get('case_id') for c in cases}:
    raise SystemExit('selected_case_ids and cases disagree')
print('[CHECK] frozen 12-case benchmark:', ids)
PY
if [ $? -ne 0 ]; then exit 1; fi

if timeout 2 rosnode list >/dev/null 2>&1; then
  echo "[ERROR] ROS master already running. Close other ROS/Gazebo sessions first."
  exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
GIT_BRANCH="$(git branch --show-current)"
python3 - "${OUT_ROOT}/benchmark_metadata.json" "${CASE_FILE}" "${GIT_COMMIT}" "${GIT_BRANCH}" \
  "${RUN_SECONDS}" "${CARE_WEIGHT}" "${SAFETY_MARGIN}" "${C4_STREAK}" "${C4_INPUT_TIMEOUT}" "${LEGACY_C4_1_ROOT}" <<'PY'
import json,os,sys,time
(out,case_file,commit,branch,run_s,weight,margin,streak,input_timeout,legacy_root)=sys.argv[1:]
p={
  'created_wall_time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
  'case_file':case_file,
  'git_commit':commit,
  'git_branch':branch,
  'num_cases':12,
  'mode':'c4_2_predicted_vbc_closed_loop',
  'c3_predictive_recovery_enabled':False,
  'c4_predicted_vbc_auditor_enabled':True,
  'c4_controls_recovery':True,
  'c4_recovery_trigger':'direct_bool_after_consecutive_predicted_vbc_violations',
  'legacy_nominal_deadline_triggers_recovery':False,
  'verification_hold_on_stale_audit':True,
  'post_release_observation_s':float(run_s),
  'waypoint_weight':float(weight),
  'safety_margin_s':float(margin),
  'c4_consecutive_violations_required':int(streak),
  'c4_input_timeout_s':float(input_timeout),
  'legacy_c4_1_root':legacy_root,
  'legacy_comparison_available_at_start':os.path.isdir(legacy_root),
  'primary_question':'Can current predicted-trajectory VBC supervision preserve executed safety while avoiding unnecessary legacy Deadline Recovery?',
}
json.dump(p,open(out,'w'),indent=2)
PY

mapfile -t CASE_IDS < <(python3 - "${CASE_FILE}" <<'PY'
import json,sys
print('\n'.join(json.load(open(sys.argv[1]))['selected_case_ids']))
PY
)

FAILED=0
for cid in "${CASE_IDS[@]}"; do
  RUN_DIR="${OUT_ROOT}/runs/${cid}"
  CASE_LOG="${LOG_ROOT}/${cid}"
  echo
  echo "============================================================"
  echo "[CASE] ${cid}"
  echo "============================================================"

  if CASE_ID="${cid}" \
     CASE_FILE="${CASE_FILE}" \
     OUT="${RUN_DIR}" \
     LOG="${CASE_LOG}" \
     RUN_SECONDS="${RUN_SECONDS}" \
     NCDF_ENV="${NCDF_ENV}" \
     CARE_WEIGHT="${CARE_WEIGHT}" \
     SAFETY_MARGIN="${SAFETY_MARGIN}" \
     C4_STREAK="${C4_STREAK}" \
     C4_INPUT_TIMEOUT="${C4_INPUT_TIMEOUT}" \
     AUDIT_WARN_MS="${AUDIT_WARN_MS}" \
     bash scripts/run_phase_c4_2_smoke.sh; then
    RC=0
    STATUS="ok"
    MESSAGE="completed"
  else
    RC=$?
    STATUS="failed"
    MESSAGE="smoke_runner_exit_${RC}"
    FAILED=$((FAILED+1))
  fi

  mkdir -p "${RUN_DIR}"
  python3 - "${RUN_DIR}/run_status.json" "${cid}" "${STATUS}" "${MESSAGE}" "${RC}" <<'PY'
import json,sys,time
out,cid,status,message,rc=sys.argv[1:]
json.dump({
  'case_id':cid,
  'status':status,
  'message':message,
  'return_code':int(rc),
  'wall_time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
},open(out,'w'),indent=2)
PY

done

SUM_ARGS=(
  --case-file "${CASE_FILE}"
  --output-root "${OUT_ROOT}"
)
if [ -d "${LEGACY_C4_1_ROOT}" ]; then
  SUM_ARGS+=(--legacy-c4-1-root "${LEGACY_C4_1_ROOT}")
else
  echo "[INFO] legacy C4.1 output not found at ${LEGACY_C4_1_ROOT}; skipping historical behavior comparison"
fi
python3 scripts/summarize_phase_c4_2_frozen12.py "${SUM_ARGS[@]}"
SUMMARY_RC=$?

python3 - "${ARCHIVE}" "${OUT_ROOT}" "${LOG_ROOT}" <<'PY'
import os,sys,zipfile
archive,out_root,log_root=sys.argv[1:]
repo=os.path.commonpath([out_root,log_root])
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for root in (out_root,log_root):
        for dp,_,files in os.walk(root):
            for fn in files:
                path=os.path.join(dp,fn)
                z.write(path,os.path.relpath(path,repo))
print('[ARCHIVE]',archive)
PY

echo
if [ "${FAILED}" -gt 0 ]; then
  echo "[WARN] ${FAILED}/12 runs failed infrastructure execution. Inspect run_status.json and logs before scientific interpretation."
fi
if [ "${SUMMARY_RC}" -ne 0 ]; then
  echo "[ERROR] summarizer failed with code ${SUMMARY_RC}"
  exit "${SUMMARY_RC}"
fi

echo "[DONE] Phase C4.2 frozen-12 closed-loop benchmark complete"
echo "[RESULT] ${OUT_ROOT}/benchmark_summary.json"
echo "[TABLE]  ${OUT_ROOT}/case_results.csv"
echo "[MD]     ${OUT_ROOT}/case_results.md"
echo "[ZIP]    ${ARCHIVE}"
