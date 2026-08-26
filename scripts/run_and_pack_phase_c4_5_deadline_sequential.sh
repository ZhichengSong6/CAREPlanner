#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-8.0}"
CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.30}"
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT:-0.20}"

OUT="${OUT:-${REPO}/outputs/phase_c4_5_deadline_sequential/${CASE_ID}}"
LOG="${LOG:-${REPO}/logs/phase_c4_5_deadline_sequential/${CASE_ID}}"
ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C4_5_deadline_sequential_${CASE_ID}.zip}"

cd "${REPO}"

echo "[C4.5] branch: $(git branch --show-current)"
echo "[C4.5] head:   $(git rev-parse HEAD)"
echo "[C4.5] case=${CASE_ID} run=${RUN_SECONDS}s weight=${CARE_WEIGHT} margin=${SAFETY_MARGIN}s"

catkin build
source devel/setup.bash

python3 -m py_compile \
  src/care_visibility_cdf/scripts/vbc_deadline_waypoint_sequential_impl.py \
  src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py

rm -f "${ZIP_PATH}"

CASE_ID="${CASE_ID}" \
RUN_SECONDS="${RUN_SECONDS}" \
CARE_WEIGHT="${CARE_WEIGHT}" \
SAFETY_MARGIN="${SAFETY_MARGIN}" \
PREDICTION_TIMEOUT="${PREDICTION_TIMEOUT}" \
REGION_SCHEDULE_MODE="deadline_sequential" \
OUT="${OUT}" \
LOG="${LOG}" \
bash scripts/run_phase_c4_4_verified_regime_smoke.sh

if ! grep -q "region_schedule_mode=deadline_sequential" "${LOG}/waypoint_generator.log"; then
  echo "[ERROR] C4.5 deadline_sequential mode was not observed in waypoint_generator.log"
  tail -n 200 "${LOG}/waypoint_generator.log" || true
  exit 3
fi

mkdir -p "${OUT}"
{
  echo "branch=$(git branch --show-current)"
  echo "head=$(git rev-parse HEAD)"
  echo "case_id=${CASE_ID}"
  echo "run_seconds=${RUN_SECONDS}"
  echo "care_weight=${CARE_WEIGHT}"
  echo "safety_margin_s=${SAFETY_MARGIN}"
  echo "prediction_timeout_s=${PREDICTION_TIMEOUT}"
  echo "region_schedule_mode=deadline_sequential"
} > "${OUT}/c4_5_run_metadata.txt"

python3 - "${OUT}/waypoint_summary.csv" "${OUT}/c4_5_deadline_sequential_summary.json" <<'PY'
import csv
import json
import os
import re
import sys

src, dst = sys.argv[1:]
tok = re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')
records = []
if os.path.isfile(src):
    with open(src, newline='', errors='replace') as f:
        rd = csv.reader(f)
        header = next(rd, [])
        if header:
            di = header.index('field.data') if 'field.data' in header else 1
            for row in rd:
                if len(row) <= di:
                    continue
                fields = dict(tok.findall(','.join(row[di:])))
                if fields.get('steering_policy') == 'deadline_sequential_region':
                    records.append(fields)

def as_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

region_ids = [as_int(r.get('sequential_selected_region_id'), -1) for r in records]
region_ids = [x for x in region_ids if x >= 0]
payload = {
    'steering_policy': 'deadline_sequential_region',
    'summary_records': len(records),
    'unique_selected_region_ids': sorted(set(region_ids)),
    'num_unique_selected_regions': len(set(region_ids)),
    'max_switch_count': max(
        [as_int(r.get('sequential_switch_count'), 0) for r in records] or [0]),
    'max_retained_count': max(
        [as_int(r.get('sequential_retained_count'), 0) for r in records] or [0]),
    'max_pending_region_count': max(
        [as_int(r.get('sequential_pending_region_count'), 0) for r in records] or [0]),
    'last_selection_reason': (
        records[-1].get('sequential_selection_reason') if records else None),
}
with open(dst, 'w') as f:
    json.dump(payload, f, indent=2)
print(json.dumps(payload, indent=2))
PY

cd "${REPO}"
zip -r "${ZIP_PATH}" \
  "${OUT#${REPO}/}" \
  "${LOG#${REPO}/}"

echo ""
echo "[C4.5 COMPLETE]"
echo "[ZIP] ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
