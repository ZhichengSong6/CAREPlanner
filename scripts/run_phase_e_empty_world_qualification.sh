#!/usr/bin/env bash
set -euo pipefail

# Phase-E empty-world qualification.
#
# Purpose:
#   Run the exact Phase-E stack on the exact obstacle-selected goals while
#   changing only the Gazebo environment from maixsense_obstacles.world to
#   maixsense_empty.world.
#
# Kept identical to Phase E:
#   - real ToF ray integration
#   - UNKNOWN/OCCUPIED map semantics
#   - final GCDF + exact VBC certification
#   - measured-state E5 execution collision audit
#   - planner / tracker / VisCDF logic
#
# Changed:
#   - Gazebo world contains no external obstacles.
#
# Qualification rule:
#   every selected case must reach the EE goal before the watchdog timeout.
#   EARLY_STOP_ON_GOAL=true means successful cases terminate immediately.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
DEFAULT_CASE_FILE="${REPO}/src/egocentric_arm_planner/config/phase_e_obstacle_core30_v1.json"
CASE_FILE="${CASE_FILE:-${DEFAULT_CASE_FILE}}"
RUN_SECONDS="${RUN_SECONDS:-45}"
EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-true}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
EXPECTED_CASE_COUNT="${EXPECTED_CASE_COUNT:-30}"
KEEP_CASE_ZIPS="${KEEP_CASE_ZIPS:-0}"

WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"

cd "${REPO}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "[ERROR] Phase-E qualification case file not found:"
  echo "        ${CASE_FILE}"
  echo ""
  echo "The repository currently may only contain the frozen Core-12 file."
  echo "Point CASE_FILE at the frozen obstacle-selected 30-goal JSON, e.g.:"
  echo "  CASE_FILE=/path/to/phase_e_obstacle_core30_v1.json \\"
  echo "  bash scripts/run_phase_e_empty_world_qualification.sh"
  echo ""
  echo "For a Core-12 smoke only:"
  echo "  CASE_FILE=${REPO}/src/egocentric_arm_planner/config/phase_e_obstacle_core12_v1.json \\"
  echo "  EXPECTED_CASE_COUNT=12 \\"
  echo "  bash scripts/run_phase_e_empty_world_qualification.sh"
  exit 2
fi

if [[ ! -f "${WORLD_FILE}" ]]; then
  echo "[ERROR] empty Gazebo world not found: ${WORLD_FILE}"
  exit 3
fi

mapfile -t CASES < <(
  python3 - "${CASE_FILE}" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as f:
    db = json.load(f)
cases = db.get("cases", [])
for c in cases:
    cid = str(c.get("case_id", "")).strip()
    if cid:
        print(cid)
PY
)

if [[ "${#CASES[@]}" -eq 0 ]]; then
  echo "[ERROR] no cases found in ${CASE_FILE}"
  exit 4
fi

if [[ "${EXPECTED_CASE_COUNT}" -gt 0 && "${#CASES[@]}" -ne "${EXPECTED_CASE_COUNT}" ]]; then
  echo "[ERROR] expected ${EXPECTED_CASE_COUNT} cases, but ${CASE_FILE} contains ${#CASES[@]}."
  echo "        Refusing to silently run the wrong qualification set."
  echo "        If this is intentional, set EXPECTED_CASE_COUNT=${#CASES[@]}."
  exit 5
fi

STAMP="${BATCH_STAMP:-$(date +%Y%m%d-%H%M%S)}"
GIT_SHORT="$(git rev-parse --short=8 HEAD)"
BATCH_ID="${BATCH_ID:-phase_e_empty_qualification_${STAMP}_${GIT_SHORT}}"
ROOT="${REPO}/outputs/phase_e_empty_world_qualification/${BATCH_ID}"
SUMMARY_DIR="${ROOT}/case_summaries"
ARTIFACT_DIR="${ROOT}/case_artifacts"
LOG_DIR="${ROOT}/logs"
FINAL_JSON="${ROOT}/qualification_summary.json"
FINAL_CSV="${ROOT}/qualification_summary.csv"
FINAL_ZIP="${REPO}/CAREPlanner_PHASE_E_EMPTY_QUALIFICATION_${BATCH_ID}.zip"

rm -rf "${ROOT}"
rm -f "${FINAL_ZIP}"
mkdir -p "${SUMMARY_DIR}" "${ARTIFACT_DIR}" "${LOG_DIR}"

cat > "${ROOT}/qualification_metadata.txt" <<EOF
benchmark=phase_e_empty_world_qualification
git_head=$(git rev-parse HEAD)
git_branch=$(git branch --show-current)
case_file=${CASE_FILE}
case_count=${#CASES[@]}
world_file=${WORLD_FILE}
confidence_map_config=${CONFIDENCE_MAP_CONFIG_FILE}
tof_fusion_enabled=true
execution_gcdf_audit_enabled=true
gcdf_body_inflation_m=0.015
run_seconds_watchdog=${RUN_SECONDS}
early_stop_on_goal=${EARLY_STOP_ON_GOAL}
qualification_rule=all_cases_task_success
cases=${CASES[*]}
EOF

cleanup_ros() {
  local n
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  sleep 0.5
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${n}" 2>/dev/null || true
  done
  rm -f /tmp/care_collision_cdf_gpu_c5_5.sock \
        /tmp/care_collision_cdf_gpu_c5_4.sock 2>/dev/null || true
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  [[ -f "${src}" ]] && cp -f "${src}" "${dst}"
}

echo "================================================================"
echo "PHASE-E EMPTY-WORLD QUALIFICATION"
echo "cases       : ${#CASES[@]}"
echo "case file   : ${CASE_FILE}"
echo "world       : ${WORLD_FILE}"
echo "map         : real ToF ray"
echo "E5 audit    : enabled"
echo "watchdog    : ${RUN_SECONDS}s"
echo "early stop  : ${EARLY_STOP_ON_GOAL}"
echo "================================================================"

for CASE_ID in "${CASES[@]}"; do
  echo ""
  echo "================================================================"
  echo "[QUALIFY] ${CASE_ID}"
  echo "================================================================"

  cleanup_ros

  RUN_ID="${CASE_ID}_empty_qualification_${STAMP}_${GIT_SHORT}"
  RUN_ROOT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
  CASE_ZIP="${REPO}/CAREPlanner_C5_RESULT_${RUN_ID}.zip"
  CASE_ART="${ARTIFACT_DIR}/${CASE_ID}"
  mkdir -p "${CASE_ART}"

  set +e
  (
    CASE_FILE="${CASE_FILE}" \
    CASE_ID="${CASE_ID}" \
    RUN_ID="${RUN_ID}" \
    RUN_SECONDS="${RUN_SECONDS}" \
    WORLD_FILE="${WORLD_FILE}" \
    CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE}" \
    TOF_FUSION_ENABLED=true \
    EXECUTION_GCDF_AUDIT_ENABLED=true \
    GCDF_BODY_INFLATION_M=0.015 \
    EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL}" \
    GAZEBO_GUI="${GAZEBO_GUI}" \
    USE_RVIZ="${USE_RVIZ}" \
    bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
  ) > >(tee "${LOG_DIR}/${CASE_ID}.log") 2>&1
  RUN_RC=$?
  set -e

  EVAL_RC=0
  if [[ -d "${RUN_ROOT}/run" ]]; then
    set +e
    python3 scripts/evaluate_phase_d_run.py \
      --repo "${REPO}" \
      --run-dir "${RUN_ROOT}/run" \
      --cases-json "${CASE_FILE}" \
      --case-id "${CASE_ID}" \
      --method "phase_e_empty_world_qualification" \
      --trial-id "${BATCH_ID}" \
      --output-json "${SUMMARY_DIR}/${CASE_ID}.json" \
      >> "${LOG_DIR}/${CASE_ID}.log" 2>&1
    EVAL_RC=$?
    set -e

    for name in \
      joint_states.csv \
      task_trajectory.csv \
      committed_trajectory.csv \
      regime_summary.csv \
      visibility_acquisition_summary.csv \
      candidate_vbc_summary.csv \
      execution_vbc_summary.csv \
      verification_outcome.csv \
      commit_summary.csv \
      tracker_summary.csv \
      local_planner_summary.csv \
      nominal_progress_summary.csv \
      blocker_stack_summary.csv \
      waypoint_schedule_summary.csv \
      e3_summary.csv \
      tof_fusion_summary.csv \
      execution_gcdf_selector_summary.csv \
      execution_gcdf_safety_summary.csv \
      execution_gcdf_hard_hold.csv \
      goal_stop_status.json \
      tracker_execution_breakdown.json; do
      copy_if_exists "${RUN_ROOT}/run/${name}" "${CASE_ART}/${name}"
    done
    copy_if_exists "${RUN_ROOT}/c5_4_local_sparse_scp_summary.json" \
      "${CASE_ART}/c5_4_local_sparse_scp_summary.json"
  else
    EVAL_RC=2
  fi

  if [[ ! -f "${SUMMARY_DIR}/${CASE_ID}.json" ]]; then
    python3 - "${SUMMARY_DIR}/${CASE_ID}.json" "${CASE_ID}" "${RUN_RC}" "${EVAL_RC}" <<'PY'
import json, sys
path, case_id, run_rc, eval_rc = sys.argv[1:]
json.dump({
    "phase": "E.empty_qualification",
    "case_id": case_id,
    "task_success": False,
    "overall_safe": False,
    "benchmark_runner_failure": True,
    "runner_return_code": int(run_rc),
    "evaluator_return_code": int(eval_rc),
}, open(path, "w"), indent=2)
PY
  fi

  if [[ "${KEEP_CASE_ZIPS}" != "1" ]]; then
    rm -f "${CASE_ZIP}"
  fi
done

python3 - "${SUMMARY_DIR}" "${FINAL_JSON}" "${FINAL_CSV}" "${CASE_FILE}" "${WORLD_FILE}" <<'PY'
import csv
import glob
import json
import math
import os
import sys

summary_dir, out_json, out_csv, case_file, world_file = sys.argv[1:]
rows = []
for path in sorted(glob.glob(os.path.join(summary_dir, "*.json"))):
    try:
        with open(path) as f:
            row = json.load(f)
        if row.get("case_id"):
            rows.append(row)
    except Exception:
        pass

def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def nested(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur

rows.sort(key=lambda r: r.get("case_id", ""))
success = [r for r in rows if bool(r.get("task_success"))]
safe = [r for r in rows if bool(r.get("overall_safe"))]
failures = [r.get("case_id") for r in rows if not bool(r.get("task_success"))]
unsafe = [r.get("case_id") for r in rows if not bool(r.get("overall_safe"))]

report = {
    "benchmark": "phase_e_empty_world_qualification",
    "case_file": case_file,
    "world_file": world_file,
    "case_count": len(rows),
    "task_success_count": len(success),
    "task_success_rate": (len(success) / len(rows)) if rows else None,
    "overall_safe_count": len(safe),
    "overall_safe_rate": (len(safe) / len(rows)) if rows else None,
    "qualification_pass": bool(rows) and len(success) == len(rows),
    "strict_safe_qualification_pass": (
        bool(rows) and len(success) == len(rows) and len(safe) == len(rows)),
    "failed_case_ids": failures,
    "unsafe_case_ids": unsafe,
    "cases": rows,
}
with open(out_json, "w") as f:
    json.dump(report, f, indent=2)

fields = [
    "case_id", "task_success", "overall_safe", "time_to_success_s",
    "final_position_error_m", "best_position_error_m",
    "final_orientation_error_rad", "repair_count", "probe_count",
    "commit_count", "candidate_vbc_records", "candidate_vbc_unsafe_records",
    "execution_vbc_records", "execution_vbc_unsafe_records",
    "max_remaining_obligation_count", "obligation_clear_events",
    "tracking_error_max_rad",
]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({
            "case_id": r.get("case_id"),
            "task_success": int(bool(r.get("task_success"))),
            "overall_safe": int(bool(r.get("overall_safe"))),
            "time_to_success_s": r.get("time_to_success_s"),
            "final_position_error_m": r.get("final_position_error_m"),
            "best_position_error_m": r.get("best_position_error_m"),
            "final_orientation_error_rad": r.get("final_orientation_error_rad"),
            "repair_count": r.get("repair_count"),
            "probe_count": r.get("probe_count"),
            "commit_count": r.get("commit_count"),
            "candidate_vbc_records": r.get("candidate_vbc_records"),
            "candidate_vbc_unsafe_records": r.get("candidate_vbc_unsafe_records"),
            "execution_vbc_records": r.get("execution_vbc_records"),
            "execution_vbc_unsafe_records": r.get("execution_vbc_unsafe_records"),
            "max_remaining_obligation_count": r.get("max_remaining_obligation_count"),
            "obligation_clear_events": r.get("obligation_clear_events"),
            "tracking_error_max_rad": nested(r, "tracking_error_inf", "max"),
        })

print(json.dumps({
    "case_count": report["case_count"],
    "task_success_count": report["task_success_count"],
    "task_success_rate": report["task_success_rate"],
    "overall_safe_count": report["overall_safe_count"],
    "qualification_pass": report["qualification_pass"],
    "strict_safe_qualification_pass": report["strict_safe_qualification_pass"],
    "failed_case_ids": report["failed_case_ids"],
    "unsafe_case_ids": report["unsafe_case_ids"],
}, indent=2))
PY

python3 - "${ROOT}" "${FINAL_ZIP}" <<'PY'
import os, sys, zipfile
root, dst = sys.argv[1:]
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for base, _, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            z.write(path, os.path.relpath(path, root))
print(dst)
PY

echo ""
echo "=============== EMPTY-WORLD QUALIFICATION COMPLETE ==============="
echo "[SUMMARY JSON] ${FINAL_JSON}"
echo "[SUMMARY CSV]  ${FINAL_CSV}"
echo "[UPLOAD ZIP]   ${FINAL_ZIP}"
ls -lh "${FINAL_ZIP}"
