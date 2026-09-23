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
DEFAULT_CASE_FILE="${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json"
CASE_FILE="${CASE_FILE:-${DEFAULT_CASE_FILE}}"
RUN_SECONDS="${RUN_SECONDS:-45}"
EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-true}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
EXPECTED_CASE_COUNT="${EXPECTED_CASE_COUNT:-30}"
KEEP_CASE_ZIPS="${KEEP_CASE_ZIPS:-0}"
# Scientific control for the historical qualification: these 30 cases are the
# common home-pose experiment, not the later random-q0 diagnostic.
REQUIRE_NEAR_ZERO_INITIAL_Q="${REQUIRE_NEAR_ZERO_INITIAL_Q:-true}"
NEAR_ZERO_INITIAL_Q_TOL_RAD="${NEAR_ZERO_INITIAL_Q_TOL_RAD:-0.0001}"

WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"
if [[ "${VBC_GEOMETRY_BACKEND:-${GEOMETRY_BACKEND:-primitive}}" == "primitive" ]]; then
  VBC_SWEPT_VOLUME_MARGIN_M="${VBC_SWEPT_VOLUME_MARGIN_M:-0.010}"
else
  VBC_SWEPT_VOLUME_MARGIN_M="${VBC_SWEPT_VOLUME_MARGIN_M:-0.0}"
fi

# Optional scalar + per-sensor visibility branch.  Defaults mirror the online
# runner and remain disabled globally; named H9 regressions opt in explicitly.
PER_SENSOR_HYBRID_ENABLED="${PER_SENSOR_HYBRID_ENABLED:-false}"
PER_SENSOR_CHECKPOINT="${PER_SENSOR_CHECKPOINT:-${REPO}/src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt}"
PER_SENSOR_SELF_FILTER_URDF="${PER_SENSOR_SELF_FILTER_URDF:-${REPO}/src/arm_description/urdf/Arm_with_self_filter_collision.urdf}"
PER_SENSOR_BRANCH_ASCENT_STEPS="${PER_SENSOR_BRANCH_ASCENT_STEPS:-1}"
PER_SENSOR_BRANCH_STEP_SIZE="${PER_SENSOR_BRANCH_STEP_SIZE:-0.05}"
PER_SENSOR_BRANCH_MAX_STEP_NORM="${PER_SENSOR_BRANCH_MAX_STEP_NORM:-0.25}"
PER_SENSOR_MAX_BRANCH_ATTEMPTS="${PER_SENSOR_MAX_BRANCH_ATTEMPTS:-4}"
PER_SENSOR_MIN_CONSERVATIVE_G="${PER_SENSOR_MIN_CONSERVATIVE_G:-0.0}"
PER_SENSOR_REQUIRE_PRIMITIVE_LOS="${PER_SENSOR_REQUIRE_PRIMITIVE_LOS:-true}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
NCDF_DEVICE="${NCDF_DEVICE:-cpu}"
GPU_ENV="${GPU_ENV:-viscdf}"
GPU_DEVICE="${GPU_DEVICE:-cuda}"

cd "${REPO}"

if [ -z "${DISPLAY:-}" ]; then
  echo "[ERROR] Missing DISPLAY; real ToF depth rendering is required before any qualification case" >&2
  exit 2
fi

# Keep the whole batch, including the offline evaluator, on the system ROS
# Python. Phase-E GPU/NCDF workers explicitly activate their own conda envs.
if [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); then
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  _qualification_conda_before="${CONDA_DEFAULT_ENV:-unknown}"
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); do
    conda deactivate || break
  done
  echo "[QUALIFICATION ENV] sanitized inherited conda env (was ${_qualification_conda_before}); batch uses system ROS/Python"
fi

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "[ERROR] Phase-E qualification case file not found:"
  echo "        ${CASE_FILE}"
  echo ""
  echo "The historical Phase-E 30-goal qualification file is expected at:"
  echo "  ${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json"
  echo "This is the same 30-goal pool used by the previous 0/30 qualification."
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

PER_SENSOR_CHECKPOINT_SHA256="not_enabled"
if [[ "${PER_SENSOR_HYBRID_ENABLED}" == "true" || "${PER_SENSOR_HYBRID_ENABLED}" == "1" ]]; then
  if [[ ! -f "${PER_SENSOR_CHECKPOINT}" ]]; then
    echo "[ERROR] per-sensor checkpoint not found: ${PER_SENSOR_CHECKPOINT}" >&2
    exit 7
  fi
  if [[ ! -f "${PER_SENSOR_SELF_FILTER_URDF}" ]]; then
    echo "[ERROR] per-sensor self-filter URDF not found: ${PER_SENSOR_SELF_FILTER_URDF}" >&2
    exit 8
  fi
  PER_SENSOR_CHECKPOINT_SHA256="$(sha256sum "${PER_SENSOR_CHECKPOINT}" | awk '{print $1}')"
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

if [[ "${REQUIRE_NEAR_ZERO_INITIAL_Q}" == "true" || "${REQUIRE_NEAR_ZERO_INITIAL_Q}" == "1" ]]; then
  python3 - "${CASE_FILE}" "${NEAR_ZERO_INITIAL_Q_TOL_RAD}" <<'PY'
import json,sys
path,tol_s=sys.argv[1:]
tol=float(tol_s)
d=json.load(open(path))
bad=[]
worst=0.0
for case in d.get("cases",[]):
    q=case.get("initial_q",[0.0]*7)
    if len(q)!=7:
        raise SystemExit("ERROR: invalid initial_q length for "+str(case.get("case_id")))
    m=max(abs(float(x)) for x in q)
    worst=max(worst,m)
    if m>tol:
        bad.append((case.get("case_id"),m,q))
if bad:
    print("[ERROR] qualification requires historical near-zero q0; found nonzero cases:")
    for cid,m,q in bad[:10]:
        print("  ",cid,"max_abs_q=",m,"q=",q)
    raise SystemExit(6)
print(f"[Q0 PREFLIGHT OK] all {len(d.get('cases',[]))} cases near zero; max |q0|={worst:.3e} rad <= {tol:.3e}")
PY
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

if [[ -e "${ROOT}" || -e "${FINAL_ZIP}" ]]; then
  echo "[ERROR] batch output already exists; choose a new BATCH_ID (no overwrite)." >&2
  exit 9
fi
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
vbc_swept_volume_margin_m=${VBC_SWEPT_VOLUME_MARGIN_M}
startup_bootstrap_policy=per_link_50ms_max_plus_gcdf_query_footprint
startup_bootstrap_config=${CONFIDENCE_MAP_CONFIG_FILE}
require_near_zero_initial_q=${REQUIRE_NEAR_ZERO_INITIAL_Q}
near_zero_initial_q_tol_rad=${NEAR_ZERO_INITIAL_Q_TOL_RAD}
force_exact_zero_initial_q=true
run_seconds_watchdog=${RUN_SECONDS}
early_stop_on_goal=${EARLY_STOP_ON_GOAL}
qualification_rule=all_cases_task_success
task_success_authority=measured_joint_states_fk_to_requested_ee_goal
legacy_nominal_progress_role=diagnostic_only_non_authoritative_after_active_sensing_replans
per_sensor_hybrid_enabled=${PER_SENSOR_HYBRID_ENABLED}
per_sensor_checkpoint=${PER_SENSOR_CHECKPOINT}
per_sensor_checkpoint_sha256=${PER_SENSOR_CHECKPOINT_SHA256}
per_sensor_self_filter_urdf=${PER_SENSOR_SELF_FILTER_URDF}
per_sensor_branch_solver=projection_root_ascent
per_sensor_projection_iters=10
per_sensor_projection_damping=0.5
per_sensor_projection_epsilon_f=0.03
per_sensor_projection_max_step_norm=0.25
per_sensor_root_refine_iters=12
per_sensor_root_tolerance_f=0.002
per_sensor_branch_ascent_steps=${PER_SENSOR_BRANCH_ASCENT_STEPS}
per_sensor_branch_step_size=${PER_SENSOR_BRANCH_STEP_SIZE}
per_sensor_branch_max_step_norm=${PER_SENSOR_BRANCH_MAX_STEP_NORM}
per_sensor_max_branch_attempts=${PER_SENSOR_MAX_BRANCH_ATTEMPTS}
per_sensor_min_conservative_g=${PER_SENSOR_MIN_CONSERVATIVE_G}
per_sensor_require_primitive_los=${PER_SENSOR_REQUIRE_PRIMITIVE_LOS}
ncdf_env=${NCDF_ENV}
ncdf_device=${NCDF_DEVICE}
collision_gpu_env=${GPU_ENV}
collision_gpu_device=${GPU_DEVICE}
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
  if [[ -f "${src}" ]]; then
    cp -f "${src}" "${dst}"
  fi
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
echo "q0 control  : near-zero required=${REQUIRE_NEAR_ZERO_INITIAL_Q}, tol=${NEAR_ZERO_INITIAL_Q_TOL_RAD} rad"
echo "bootstrap   : per-link 50-ms MAX + 1.5cm body inflation + 2.5cm selector band"
echo "per-sensor : enabled=${PER_SENSOR_HYBRID_ENABLED} checkpoint=${PER_SENSOR_CHECKPOINT}"
echo "sensor SHA : ${PER_SENSOR_CHECKPOINT_SHA256}"
echo "NCDF        : env=${NCDF_ENV} device=${NCDF_DEVICE}"
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
    REQUIRE_REAL_TOF_READINESS=true \
    EXECUTION_GCDF_AUDIT_ENABLED=true \
    GCDF_BODY_INFLATION_M=0.015 \
    VBC_SWEPT_VOLUME_MARGIN_M="${VBC_SWEPT_VOLUME_MARGIN_M}" \
    FORCE_ZERO_INITIAL_Q=true \
    EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL}" \
    GAZEBO_GUI="${GAZEBO_GUI}" \
    USE_RVIZ="${USE_RVIZ}" \
    PER_SENSOR_HYBRID_ENABLED="${PER_SENSOR_HYBRID_ENABLED}" \
    PER_SENSOR_CHECKPOINT="${PER_SENSOR_CHECKPOINT}" \
    PER_SENSOR_SELF_FILTER_URDF="${PER_SENSOR_SELF_FILTER_URDF}" \
    PER_SENSOR_BRANCH_ASCENT_STEPS="${PER_SENSOR_BRANCH_ASCENT_STEPS}" \
    PER_SENSOR_BRANCH_STEP_SIZE="${PER_SENSOR_BRANCH_STEP_SIZE}" \
    PER_SENSOR_BRANCH_MAX_STEP_NORM="${PER_SENSOR_BRANCH_MAX_STEP_NORM}" \
    PER_SENSOR_MAX_BRANCH_ATTEMPTS="${PER_SENSOR_MAX_BRANCH_ATTEMPTS}" \
    PER_SENSOR_MIN_CONSERVATIVE_G="${PER_SENSOR_MIN_CONSERVATIVE_G}" \
    PER_SENSOR_REQUIRE_PRIMITIVE_LOS="${PER_SENSOR_REQUIRE_PRIMITIVE_LOS}" \
    NCDF_ENV="${NCDF_ENV}" \
    NCDF_DEVICE="${NCDF_DEVICE}" \
    GPU_ENV="${GPU_ENV}" \
    GPU_DEVICE="${GPU_DEVICE}" \
    bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
  ) > >(tee "${LOG_DIR}/${CASE_ID}.log") 2>&1
  RUN_RC=$?
  set -e

  EVAL_RC=0
  if [[ -d "${RUN_ROOT}/run" ]]; then
    WINDOW_ARGS=()
    if [[ "${EARLY_STOP_ON_GOAL}" == "true" || "${EARLY_STOP_ON_GOAL}" == "1" ]]; then
      WINDOW_ARGS+=(--require-benchmark-window)
    fi
    set +e
    python3 scripts/evaluate_phase_d_run.py \
      --repo "${REPO}" \
      --run-dir "${RUN_ROOT}/run" \
      --cases-json "${CASE_FILE}" \
      --case-id "${CASE_ID}" \
      --method "phase_e_empty_world_qualification" \
      --trial-id "${BATCH_ID}" \
      "${WINDOW_ARGS[@]}" \
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
      local_witness_diagnostics.csv \
      local_witness_selector_diagnostics.csv \
      nominal_progress_summary.csv \
      blocker_stack_summary.csv \
      waypoint_schedule_summary.csv \
      e3_summary.csv \
      tof_fusion_summary.csv \
      execution_gcdf_selector_summary.csv \
      execution_gcdf_safety_summary.csv \
      execution_gcdf_hard_hold.csv \
      goal_stop_status.json \
      perception_raw_ready.json \
      perception_ready.json \
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

  # An unavailable sensor pipeline invalidates the experiment setup. Preserve
  # this case's failure evidence and stop instead of launching the same broken
  # setup for every remaining case. Never replace/retry the failed attempt.
  if python3 - "${RUN_ROOT}/run" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
failed = any(p.is_file() and not json.loads(p.read_text()).get('ready', False)
             for p in (root/'perception_raw_ready.json', root/'perception_ready.json'))
raise SystemExit(0 if failed else 1)
PY
  then
    echo "[ERROR] Perception prerequisite failed; aborting batch without replacement runs"
    cleanup_ros
    exit 2
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
    "tracking_error_max_rad", "spatial_tracking_error_max_rad",
    "spatial_tracking_bound_max_m", "same_phase_tracking_bound_max_m",
    "tracking_phase_lag_max_abs_s",
    "task_progress_authority", "legacy_nominal_progress_stale",
    "legacy_nominal_progress_phase_s",
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
            "spatial_tracking_error_max_rad": nested(
                r, "spatial_tracking_error_inf", "max"),
            "spatial_tracking_bound_max_m": nested(
                r, "spatial_tracking_bound_m", "max"),
            "same_phase_tracking_bound_max_m": nested(
                r, "primitive_tracking_same_phase_bound_m", "max"),
            "tracking_phase_lag_max_abs_s": nested(
                r, "tracking_phase_abs_lag_s", "max"),
            "task_progress_authority": r.get("task_progress_authority"),
            "legacy_nominal_progress_stale": int(bool(
                r.get("legacy_nominal_progress_stale"))),
            "legacy_nominal_progress_phase_s": r.get(
                "legacy_nominal_progress_phase_s"),
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
