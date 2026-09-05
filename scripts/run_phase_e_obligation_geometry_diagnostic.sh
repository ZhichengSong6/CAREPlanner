#!/usr/bin/env bash
set -euo pipefail

# Phase-E obligation geometry / confidence-query diagnostic.
#
# Focused Phase-E validation for obligation identity coherence. The run uses
# blocker-aware C5.41 plus q_vis-compatible obligation matching and records:
#   1) confidence-query transport / response validity,
#   2) accepted geometry refreshes relative to q_vis provenance, and
#   3) stale-identity rejections that force a new geometry-specific q_vis.
#
# Default representative case: phase_e_goal_014, empty world, 15 s.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
CASE_ID="${CASE_ID:-phase_e_goal_014}"
RUN_SECONDS="${RUN_SECONDS:-15}"
EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-true}"
GAZEBO_GUI="${GAZEBO_GUI:-true}"
USE_RVIZ="${USE_RVIZ:-false}"

WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"

cd "${REPO}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "[ERROR] case file not found: ${CASE_FILE}"
  exit 2
fi

python3 - "${CASE_FILE}" "${CASE_ID}" <<'PY'
import json, sys
path, cid = sys.argv[1:]
with open(path) as f:
    db = json.load(f)
ids = [str(c.get("case_id","")) for c in db.get("cases",[])]
if cid not in ids:
    raise SystemExit(f"[ERROR] {cid} not found in {path}")
print(f"[CASE CHECK] found {cid}")
PY

# Keep ROS/Gazebo and this offline parser on system Python. GPU/NCDF workers
# activate their own environment inside the normal Phase-E runner.
if [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); then
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  _before="${CONDA_DEFAULT_ENV:-unknown}"
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); do
    conda deactivate || break
  done
  echo "[DIAG ENV] sanitized inherited conda env (was ${_before})"
fi

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SHORT="$(git rev-parse --short=8 HEAD)"
RUN_ID="${CASE_ID}_obligation_geometry_diag_${STAMP}_${SHORT}"
RUN_ROOT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
DIAG_ROOT="${REPO}/outputs/phase_e_obligation_geometry_diagnostic/${RUN_ID}"
LOG_FILE="${DIAG_ROOT}/runner.log"
SUMMARY_JSON="${DIAG_ROOT}/obligation_geometry_diagnostic.json"
UPLOAD_ZIP="${REPO}/CAREPlanner_PHASE_E_OBLIGATION_GEOMETRY_DIAG_${RUN_ID}.zip"

mkdir -p "${DIAG_ROOT}/artifacts" "${DIAG_ROOT}/obligation_traces"

cleanup_ros() {
  local n
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  sleep 0.6
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${n}" 2>/dev/null || true
  done
  rm -f /tmp/care_collision_cdf_gpu_c5_5.sock \
        /tmp/care_collision_cdf_gpu_c5_4.sock 2>/dev/null || true
}
trap cleanup_ros EXIT INT TERM

echo "================================================================"
echo "PHASE-E OBLIGATION GEOMETRY DIAGNOSTIC"
echo "case       : ${CASE_ID}"
echo "world      : ${WORLD_FILE}"
echo "runtime    : ${RUN_SECONDS}s"
echo "planner    : C5.41 + q_vis-compatible obligation identity"
echo "diagnostic : confidence validity + geometry/q_vis coherence"
echo "================================================================"

cleanup_ros
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
) > >(tee "${LOG_FILE}") 2>&1
RUN_RC=$?
set -e

RUN_DIR="${RUN_ROOT}/run"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] expected run directory missing: ${RUN_DIR}"
  exit 3
fi

for f in \
  visibility_acquisition_summary.csv \
  blocker_stack_summary.csv \
  candidate_vbc_summary.csv \
  execution_vbc_summary.csv \
  verification_outcome.csv \
  commit_summary.csv \
  tracker_summary.csv \
  regime_summary.csv \
  e3_summary.csv \
  tof_fusion_summary.csv \
  local_planner_summary.csv \
  goal_stop_status.json; do
  [[ -f "${RUN_DIR}/${f}" ]] && cp -f "${RUN_DIR}/${f}" "${DIAG_ROOT}/artifacts/${f}"
done

# Preserve the projector's per-obligation creation traces wherever the normal
# runner placed them.
while IFS= read -r -d '' p; do
  cp -f "${p}" "${DIAG_ROOT}/obligation_traces/$(basename "${p}")"
done < <(find "${RUN_ROOT}" -type f -name 'c46_obligation_*.json' -print0 2>/dev/null)

python3 - "${RUN_DIR}" "${SUMMARY_JSON}" "${CASE_ID}" "${RUN_RC}" <<'PY'
import csv
import json
import math
import os
import re
import sys

run_dir, out_json, case_id, run_rc = sys.argv[1:]
TOKEN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

def token_rows(name):
    path = os.path.join(run_dir, name)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, newline="", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        if not header:
            return out
        time_i = header.index("%time") if "%time" in header else 0
        data_i = header.index("field.data") if "field.data" in header else 1
        for row in rd:
            if len(row) <= data_i:
                continue
            # rostopic CSV may split field.data at commas embedded in token
            # values (xyz, q vectors). The normal C5 digest deliberately joins
            # every column from field.data onward; keep this parser identical.
            text = ",".join(row[data_i:])
            d = dict(TOKEN.findall(text))
            if d:
                try:
                    d["_t"] = float(row[time_i]) / 1e9
                except Exception:
                    d["_t"] = math.nan
                out.append(d)
    return out

def num(v, default=math.nan):
    try:
        x = float(v)
        return x
    except Exception:
        return default

def integer(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

acq = token_rows("visibility_acquisition_summary.csv")
blk = token_rows("blocker_stack_summary.csv")
cand = token_rows("candidate_vbc_summary.csv")
reg = token_rows("regime_summary.csv")

valid_rows = [
    r for r in acq
    if integer(r.get("active_obligation_id"), -1) >= 0
]
outside_rows = [
    r for r in valid_rows
    if r.get("active_query_status") == "ok" and
       integer(r.get("active_point_count"), 0) > 0 and
       integer(r.get("active_inside_map_count"), 0) == 0
]
length_mismatch_rows = [
    r for r in valid_rows
    if r.get("active_query_status") == "length_mismatch"
]
service_exception_rows = [
    r for r in valid_rows
    if r.get("active_query_status") == "service_exception"
]
service_exception_errors = sorted({
    r.get("active_query_error", "unknown")
    for r in service_exception_rows
})
nonfinite_rows = [
    r for r in valid_rows
    if integer(r.get("active_nonfinite_confidence_count"), 0) > 0
]
geometry_changed_rows = [
    r for r in valid_rows
    if integer(r.get("active_geometry_changed_since_qvis"), 0) == 1
]

max_source_shift = max(
    [num(r.get("active_centroid_shift_from_qvis_m")) for r in valid_rows
     if math.isfinite(num(r.get("active_centroid_shift_from_qvis_m")))] or [0.0])
max_historical_source_shift = max(
    [num(r.get("active_max_centroid_shift_from_qvis_m")) for r in valid_rows
     if math.isfinite(num(r.get("active_max_centroid_shift_from_qvis_m")))] or [0.0])
max_match_updates = max(
    [integer(r.get("active_geometry_match_update_count"), 0)
     for r in valid_rows] or [0])
max_match_changes = max(
    [integer(r.get("active_geometry_match_change_count"), 0)
     for r in valid_rows] or [0])

first_outside = outside_rows[0] if outside_rows else {}
last_acq = acq[-1] if acq else {}
last_blk = blk[-1] if blk else {}
last_reg = reg[-1] if reg else {}

unsafe_pred = [
    r for r in cand
    if r.get("trajectory_source") == "predicted" and
       r.get("has_violation") == "1"
]

if length_mismatch_rows or service_exception_rows:
    primary = "CONFIDENCE_QUERY_TRANSPORT_OR_RESPONSE_FAILURE"
elif outside_rows:
    primary = "ACTIVE_OBLIGATION_ALL_POINTS_OUTSIDE_MAP"
elif nonfinite_rows:
    primary = "INSIDE_MAP_CONFIDENCE_NONFINITE"
elif geometry_changed_rows and max_historical_source_shift > 0.05:
    primary = "OBLIGATION_GEOMETRY_DRIFT_LARGE"
elif geometry_changed_rows:
    primary = "OBLIGATION_GEOMETRY_DRIFT_PRESENT"
else:
    primary = "NO_CONFIDENCE_OR_GEOMETRY_FAILURE_OBSERVED"

report = {
    "test": "phase_e_obligation_geometry_diagnostic",
    "case_id": case_id,
    "runner_return_code": int(run_rc),
    "primary_diagnostic": primary,
    "acquisition_record_count": len(acq),
    "active_obligation_record_count": len(valid_rows),
    "ever_all_active_points_outside_map": bool(outside_rows),
    "all_active_points_outside_map_record_count": len(outside_rows),
    "first_all_outside": {
        "t": first_outside.get("_t"),
        "obligation_id": integer(first_outside.get("active_obligation_id"), -1),
        "point_count": integer(first_outside.get("active_point_count"), 0),
        "xyz_min": first_outside.get("active_xyz_min"),
        "xyz_max": first_outside.get("active_xyz_max"),
        "current_centroid": first_outside.get("active_current_centroid"),
        "q_vis_source_centroid": first_outside.get(
            "active_q_vis_source_centroid"),
        "centroid_shift_from_qvis_m": num(
            first_outside.get("active_centroid_shift_from_qvis_m")),
        "match_update_count": integer(
            first_outside.get("active_geometry_match_update_count"), 0),
    } if first_outside else None,
    "length_mismatch_record_count": len(length_mismatch_rows),
    "service_exception_record_count": len(service_exception_rows),
    "service_exception_first_t": (
        service_exception_rows[0].get("_t")
        if service_exception_rows else None),
    "service_exception_last_t": (
        service_exception_rows[-1].get("_t")
        if service_exception_rows else None),
    "service_exception_errors": service_exception_errors,
    "inside_map_nonfinite_confidence_record_count": len(nonfinite_rows),
    "geometry_changed_since_qvis_record_count": len(geometry_changed_rows),
    "max_current_centroid_shift_from_qvis_m": max_source_shift,
    "max_historical_centroid_shift_from_qvis_m": max_historical_source_shift,
    "max_geometry_match_update_count": max_match_updates,
    "max_geometry_match_change_count": max_match_changes,
    "predicted_vbc_unsafe_records": len(unsafe_pred),
    "final": {
        "query_status": last_acq.get("active_query_status"),
        "active_obligation_id": integer(
            last_acq.get("active_obligation_id"), -1),
        "remaining_ids": last_acq.get("remaining_ids"),
        "seen_obligation_count": integer(
            last_acq.get("seen_obligation_count"), 0),
        "inside_map_count": integer(
            last_acq.get("active_inside_map_count"), 0),
        "outside_map_count": integer(
            last_acq.get("active_outside_map_count"), 0),
        "finite_confidence_count": integer(
            last_acq.get("active_finite_confidence_count"), 0),
        "active_xyz_min": last_acq.get("active_xyz_min"),
        "active_xyz_max": last_acq.get("active_xyz_max"),
        "current_centroid": last_acq.get("active_current_centroid"),
        "q_vis_source_centroid": last_acq.get(
            "active_q_vis_source_centroid"),
        "centroid_shift_from_qvis_m": num(
            last_acq.get("active_centroid_shift_from_qvis_m")),
        "geometry_match_update_count": integer(
            last_acq.get("active_geometry_match_update_count"), 0),
        "geometry_changed_since_qvis": integer(
            last_acq.get("active_geometry_changed_since_qvis"), 0),
        "cycle_block_count": integer(
            last_blk.get("cycle_block_count"), 0),
        "qvis_match_check_count": integer(
            last_blk.get("qvis_match_check_count"), 0),
        "qvis_match_accept_count": integer(
            last_blk.get("qvis_match_accept_count"), 0),
        "qvis_match_reject_count": integer(
            last_blk.get("qvis_match_reject_count"), 0),
        "qvis_match_error_count": integer(
            last_blk.get("qvis_match_error_count"), 0),
        "qvis_match_last_obligation_id": integer(
            last_blk.get("qvis_match_last_obligation_id"), -1),
        "qvis_match_last_f_min": num(
            last_blk.get("qvis_match_last_f_min")),
        "qvis_match_last_reason": last_blk.get("qvis_match_last_reason"),
        "adaptive_refinement_enabled": integer(
            last_blk.get("adaptive_refinement_enabled"), 0),
        "adaptive_refinement_trigger_count": integer(
            last_blk.get("adaptive_refinement_trigger_count"), 0),
        "adaptive_refinement_success_count": integer(
            last_blk.get("adaptive_refinement_success_count"), 0),
        "adaptive_refinement_failure_count": integer(
            last_blk.get("adaptive_refinement_failure_count"), 0),
        "adaptive_refinement_skip_count": integer(
            last_blk.get("adaptive_refinement_skip_count"), 0),
        "adaptive_refinement_parent_count": integer(
            last_blk.get("adaptive_refinement_parent_count"), 0),
        "adaptive_refinement_child_count": integer(
            last_blk.get("adaptive_refinement_child_count"), 0),
        "adaptive_refinement_last_parents": last_blk.get(
            "adaptive_refinement_last_parents"),
        "adaptive_refinement_last_children": last_blk.get(
            "adaptive_refinement_last_children"),
        "adaptive_refinement_cross_f_ab": num(
            last_blk.get("adaptive_refinement_cross_f_ab")),
        "adaptive_refinement_cross_f_ba": num(
            last_blk.get("adaptive_refinement_cross_f_ba")),
        "adaptive_refinement_refined_zone_count": integer(
            last_blk.get("adaptive_refinement_refined_zone_count"), 0),
        "adaptive_refinement_family_route_count": integer(
            last_blk.get("adaptive_refinement_family_route_count"), 0),
        "adaptive_refinement_absorb_count": integer(
            last_blk.get("adaptive_refinement_absorb_count"), 0),
        "adaptive_refinement_absorbed_point_count": integer(
            last_blk.get("adaptive_refinement_absorbed_point_count"), 0),
        "adaptive_refinement_qvis_reuse_count": integer(
            last_blk.get("adaptive_refinement_qvis_reuse_count"), 0),
        "adaptive_refinement_qvis_regen_count": integer(
            last_blk.get("adaptive_refinement_qvis_regen_count"), 0),
        "adaptive_refinement_qvis_regen_failure_count": integer(
            last_blk.get("adaptive_refinement_qvis_regen_failure_count"), 0),
        "adaptive_refinement_last_family_id": integer(
            last_blk.get("adaptive_refinement_last_family_id"), -1),
        "adaptive_refinement_last_child_id": integer(
            last_blk.get("adaptive_refinement_last_child_id"), -1),
        "adaptive_refinement_last_absorb_f_min": num(
            last_blk.get("adaptive_refinement_last_absorb_f_min")),
        "adaptive_refinement_reason": last_blk.get(
            "adaptive_refinement_reason"),
        "blocker_stack": last_blk.get("stack"),
        "blocker_switch_reason": last_blk.get("switch_reason"),
        "regime_state": last_reg.get("state"),
        "commit_count": integer(last_reg.get("commit_count"), 0),
        "probe_entry_count": integer(last_reg.get("probe_entry_count"), 0),
    },
}
with open(out_json, "w") as f:
    json.dump(report, f, indent=2, allow_nan=True)

print("\n========== OBLIGATION GEOMETRY DIAGNOSTIC ==========")
print(json.dumps(report, indent=2, allow_nan=True))
print("====================================================")
PY

cat > "${DIAG_ROOT}/metadata.txt" <<EOF
git_head=$(git rev-parse HEAD)
case_file=${CASE_FILE}
case_id=${CASE_ID}
world_file=${WORLD_FILE}
confidence_map_config=${CONFIDENCE_MAP_CONFIG_FILE}
run_seconds=${RUN_SECONDS}
planner_semantics=persistent_spatial_partition_visibility_obligations
EOF

rm -f "${UPLOAD_ZIP}"
python3 - "${DIAG_ROOT}" "${UPLOAD_ZIP}" <<'PY'
import os, sys, zipfile
root, dst = sys.argv[1:]
with zipfile.ZipFile(
        dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for base, _, files in os.walk(root):
        for name in files:
            p = os.path.join(base, name)
            z.write(p, os.path.relpath(p, root))
print(dst)
PY

echo ""
echo "[SUMMARY]    ${SUMMARY_JSON}"
echo "[UPLOAD ZIP] ${UPLOAD_ZIP}"
ls -lh "${UPLOAD_ZIP}"
