#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-8.0}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.20}"

OUT="${OUT:-${REPO}/outputs/phase_c4_6_multi_deadline/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_6_multi_deadline/${CASE_ID}}"
ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C4_6_multi_deadline_${CASE_ID}.zip}"

cd "${REPO}"

echo "[C4.6] branch: $(git branch --show-current)"
echo "[C4.6] head:   $(git rev-parse HEAD)"
echo "[C4.6] case=${CASE_ID} run=${RUN_SECONDS}s weight=${CARE_WEIGHT} margin=${SAFETY_MARGIN}s"

# CAREPlanner uses catkin_tools.
catkin build
source devel/setup.bash

# Fail before Gazebo on Python syntax/import-file mistakes.
python3 -m py_compile \
  src/care_visibility_cdf/scripts/vbc_multi_deadline_obligation_impl.py \
  src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py

rm -f "${ZIP_PATH}"

CASE_ID="${CASE_ID}" \
RUN_SECONDS="${RUN_SECONDS}" \
CARE_WEIGHT="${CARE_WEIGHT}" \
SAFETY_MARGIN="${SAFETY_MARGIN}" \
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT}" \
REGION_SCHEDULE_MODE="accumulated_multi_deadline" \
OUT="${OUT}" \
LOG="${LOG}" \
bash scripts/run_phase_c4_4_verified_regime_smoke.sh

mkdir -p "${OUT}"
{
  echo "branch=$(git branch --show-current)"
  echo "head=$(git rev-parse HEAD)"
  echo "case_id=${CASE_ID}"
  echo "run_seconds=${RUN_SECONDS}"
  echo "care_weight=${CARE_WEIGHT}"
  echo "safety_margin_s=${SAFETY_MARGIN}"
  echo "prediction_timeout_s=${PREDICTION_TIMEOUT}"
  echo "region_schedule_mode=accumulated_multi_deadline"
} > "${OUT}/c4_6_run_metadata.txt"

MODE_OK=0
MPC_MULTI_OK=0
if grep -q "region_schedule_mode=accumulated_multi_deadline" "${LOG}/waypoint_generator.log" 2>/dev/null; then
  MODE_OK=1
else
  echo "[WARN] C4.6 mode marker not found in waypoint_generator.log"
fi

if grep -q "multi_deadline_enabled=1" "${OUT}/mpc_summary.csv" 2>/dev/null; then
  MPC_MULTI_OK=1
else
  echo "[WARN] MPC multi_deadline_enabled=1 not observed in mpc_summary.csv"
fi

python3 - "${OUT}" "${MODE_OK}" "${MPC_MULTI_OK}" <<'PY'
import csv
import json
import math
import os
import re
import statistics
import sys

out, mode_ok, mpc_multi_ok = sys.argv[1:]
TOK = re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')

def recs(name):
    p = os.path.join(out, name)
    rows = []
    if not os.path.isfile(p):
        return rows
    with open(p, newline='', errors='replace') as f:
        rd = csv.reader(f)
        h = next(rd, [])
        if not h:
            return rows
        di = h.index('field.data') if 'field.data' in h else 1
        ti = h.index('%time') if '%time' in h else 0
        for r in rd:
            if len(r) <= di:
                continue
            d = dict(TOK.findall(','.join(r[di:])))
            try:
                d['_t'] = float(r[ti]) / 1e9
            except Exception:
                d['_t'] = math.nan
            if d:
                rows.append(d)
    return rows

def f(v):
    try:
        return float(str(v).replace('ms', ''))
    except Exception:
        return math.nan

def i(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

mpc = recs('mpc_summary.csv')
sched = recs('waypoint_schedule_summary.csv')
cand = recs('candidate_vbc_summary.csv')
exe = recs('execution_vbc_summary.csv')
reg = recs('regime_summary.csv')
commit = recs('commit_summary.csv')

multi = [r for r in mpc if r.get('vbc_wp') == 'multi_deadline_repair']
solve = [f(r.get('solve')) for r in mpc]
solve = [x for x in solve if math.isfinite(x)]
errors = [f(r.get('repair_max_pred_error_inf')) for r in multi]
errors = [x for x in errors if math.isfinite(x)]
counts = [i(r.get('repair_obligation_count')) for r in multi]
expired = [i(r.get('repair_expired_count')) for r in multi]

last_sched = sched[-1] if sched else {}
last_reg = reg[-1] if reg else {}
last_commit = commit[-1] if commit else {}

payload = {
    'region_schedule_mode': 'accumulated_multi_deadline',
    'mode_marker_ok': bool(int(mode_ok)),
    'mpc_multi_deadline_marker_ok': bool(int(mpc_multi_ok)),
    'mpc_records': len(mpc),
    'multi_deadline_repair_cycles': len(multi),
    'max_repair_obligation_count': max(counts or [0]),
    'max_repair_expired_count': max(expired or [0]),
    'repair_max_pred_error_inf_median': statistics.median(errors) if errors else None,
    'mpc_solve_ms_median': statistics.median(solve) if solve else None,
    'mpc_solve_ms_max': max(solve) if solve else None,
    'candidate_predicted_unsafe': sum(
        r.get('trajectory_source') == 'predicted' and r.get('has_violation') == '1'
        for r in cand),
    'candidate_predicted_safe': sum(
        r.get('trajectory_source') == 'predicted' and r.get('has_violation') == '0'
        for r in cand),
    'execution_unsafe': sum(r.get('has_violation') == '1' for r in exe),
    'final_regime_state': last_reg.get('state'),
    'verification_safe_count': i(last_commit.get('verification_safe_count')),
    'verification_unsafe_count': i(last_commit.get('verification_unsafe_count')),
    'commit_count': i(last_commit.get('commit_count')),
    'last_obligation_count': i(last_sched.get('obligation_count')),
    'last_unreachable_at_discovery_count': i(
        last_sched.get('unreachable_at_discovery_count')),
    'obligation_new_count': i(last_sched.get('new_obligation_count')),
    'obligation_match_count': i(last_sched.get('matched_obligation_count')),
    'obligation_clear_count': i(last_sched.get('clear_count')),
    'obligation_generation_failure_count': i(
        last_sched.get('generation_failure_count')),
}
with open(os.path.join(out, 'c4_6_multi_deadline_summary.json'), 'w') as fobj:
    json.dump(payload, fobj, indent=2)
print(json.dumps(payload, indent=2))
PY

cd "${REPO}"
zip -r "${ZIP_PATH}" \
  "${OUT#${REPO}/}" \
  "${LOG#${REPO}/}"

echo ""
echo "[C4.6 COMPLETE]"
echo "[SUMMARY] ${OUT}/c4_6_multi_deadline_summary.json"
echo "[ZIP] ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
