#!/usr/bin/env bash
set -euo pipefail

# Phase-E random feasible-q0 cross-pair experiment.
#
# Scientific question:
#   Does the strong Phase-E deadlock observed from the common zero/home pose
#   persist when the SAME 30 target goals are started from obstacle-feasible
#   configurations drawn from the same obstacle-aware goal pool?
#
# Design:
#   - same 30 target EE goals as the original obstacle-selected pool
#   - every target used exactly once
#   - q0 pool regenerated directly in joint space in the obstacle world
#   - 30 distinct q0 samples assigned one-to-one to the 30 targets
#   - q0 obstacle clearance is recomputed before the batch starts
#   - obstacle world is used for BOTH q0 feasibility and execution
#
# This benchmark intentionally changes only initial configuration relative to
# the target set. The current corrected raw-body VBC margin is kept at 0.
# The earlier full-q_vis terminal-horizon experiment is NOT used here; the
# canonical local frontier/q_vis steering path is restored for this diagnostic.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
RUN_SECONDS="${RUN_SECONDS:-45}"
EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-true}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
SEED="${SEED:-20260906}"
MIN_START_GOAL_EE_DISTANCE_M="${MIN_START_GOAL_EE_DISTANCE_M:-0.15}"
Q0_REQUIRED_CLEARANCE_M="${Q0_REQUIRED_CLEARANCE_M:-0.06}"
Q0_BODY_INFLATION_M="${Q0_BODY_INFLATION_M:-0.015}"
# This diagnostic intentionally removes the extra trusted-free shell. The
# robot's raw occupied body is still marked known-free at startup.
STARTUP_PRIOR_INFLATION_M="${STARTUP_PRIOR_INFLATION_M:-0.0}"
STARTUP_PRIOR_REQUIRED_CLEARANCE_M="${STARTUP_PRIOR_REQUIRED_CLEARANCE_M:-0.0}"
Q0_POOL_TARGET_COUNT="${Q0_POOL_TARGET_COUNT:-80}"
Q0_POOL_MAX_SAMPLES="${Q0_POOL_MAX_SAMPLES:-100000}"
OFFLINE_GEOMETRY_CONDA_ENV="${OFFLINE_GEOMETRY_CONDA_ENV:-viscdf}"
BUILD_ONLY="${BUILD_ONLY:-false}"

WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_obstacles.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"

cd "${REPO}"

# Keep the original 30 targets fixed. Initial configurations are regenerated
# directly in joint space in the SAME obstacle world; they are not recycled
# from goal IK solutions.
if [[ -z "${TARGET_POOL:-}" ]]; then
  TARGET_POOL="${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json"
fi

if [[ ! -f "${TARGET_POOL}" ]]; then
  echo "[ERROR] target pool not found: ${TARGET_POOL}"
  exit 2
fi

if ! python3 - "${TARGET_POOL}" <<'PY'
import json,sys
t=json.load(open(sys.argv[1])).get("cases",[])
if len(t)!=30:
    raise SystemExit(f"target pool must contain 30 cases, got {len(t)}")
print(f"[TARGET POOL OK] targets={len(t)}")
PY
then
  echo "[ERROR] target pool validation failed"
  exit 2
fi

if [[ ! -f "${WORLD_FILE}" ]]; then
  echo "[ERROR] obstacle world not found: ${WORLD_FILE}"
  exit 3
fi

# The offline q0 builder uses Pinocchio. Historical Phase-E geometry
# generation was run in the 'viscdf' conda environment (Pinocchio 4.1.0).
# Resolve conda explicitly and preflight the dependency before any batch work.
if command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
  CONDA_EXE="${HOME}/anaconda3/bin/conda"
elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  CONDA_EXE="${HOME}/miniconda3/bin/conda"
else
  echo "[ERROR] conda executable not found"
  exit 5
fi

echo "[PREFLIGHT] offline geometry env=${OFFLINE_GEOMETRY_CONDA_ENV}"
if ! "${CONDA_EXE}" run -n "${OFFLINE_GEOMETRY_CONDA_ENV}" \
  python -c 'import pinocchio as pin; print("[PREFLIGHT OK] pinocchio", pin.__version__)'
then
  echo "[ERROR] Pinocchio preflight failed in conda env: ${OFFLINE_GEOMETRY_CONDA_ENV}"
  echo "        Historical Phase-E geometry scripts were run in env 'viscdf'."
  exit 6
fi

# Keep ROS/Gazebo on system Python; GPU/NCDF workers activate their own envs.
if [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); then
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); do
    conda deactivate || break
  done
fi

STAMP="${BATCH_STAMP:-$(date +%Y%m%d-%H%M%S)}"
SHORT="$(git rev-parse --short=8 HEAD)"
BATCH_ID="${BATCH_ID:-phase_e_random_q0_cross_pairs_${STAMP}_${SHORT}}"
ROOT="${REPO}/outputs/phase_e_random_q0_cross_pairs/${BATCH_ID}"
CASE_FILE="${CASE_FILE:-${ROOT}/random_q0_cross_pairs.json}"
SUMMARY_DIR="${ROOT}/case_summaries"
ARTIFACT_DIR="${ROOT}/case_artifacts"
LOG_DIR="${ROOT}/logs"
FINAL_JSON="${ROOT}/experiment_summary.json"
FINAL_CSV="${ROOT}/experiment_summary.csv"
FINAL_ZIP="${REPO}/CAREPlanner_PHASE_E_RANDOM_Q0_CROSS_PAIRS_${BATCH_ID}.zip"
Q0_SOURCE_POOL="${ROOT}/random_initial_q_pool.json"
EXPERIMENT_CONFIDENCE_MAP_CONFIG_FILE="${ROOT}/confidence_map_phase_e_random_q0.yaml"

rm -rf "${ROOT}"
rm -f "${FINAL_ZIP}"
mkdir -p "${ROOT}" "${SUMMARY_DIR}" "${ARTIFACT_DIR}" "${LOG_DIR}"

echo "================================================================"
echo "GENERATING DIRECT RANDOM q0 POOL"
echo "================================================================"
echo "target pool : ${TARGET_POOL}"
echo "world       : ${WORLD_FILE}"
echo "seed        : ${SEED}"
echo "startup prior extra inflation : ${STARTUP_PRIOR_INFLATION_M} m"
echo

# Build an experiment-local confidence-map config so the formal Phase-E config
# remains untouched. Only current_body_prior/inflation_radius is changed.
python3 - "${CONFIDENCE_MAP_CONFIG_FILE}" "${EXPERIMENT_CONFIDENCE_MAP_CONFIG_FILE}" "${STARTUP_PRIOR_INFLATION_M}" <<'PY'
import sys,yaml
src,dst,inflate=sys.argv[1:]
d=yaml.safe_load(open(src))
d.setdefault("current_body_prior",{})["inflation_radius"]=float(inflate)
with open(dst,"w") as f:
    yaml.safe_dump(d,f,sort_keys=False)
print("[CONFIG] experiment confidence map:",dst)
print("[CONFIG] startup prior inflation:",float(inflate))
PY
CONFIDENCE_MAP_CONFIG_FILE="${EXPERIMENT_CONFIDENCE_MAP_CONFIG_FILE}"

"${CONDA_EXE}" run -n "${OFFLINE_GEOMETRY_CONDA_ENV}" \
  python scripts/generate_phase_e_random_initial_q_pool.py \
  --repo "${REPO}" \
  --world "${WORLD_FILE}" \
  --target-count "${Q0_POOL_TARGET_COUNT}" \
  --max-samples "${Q0_POOL_MAX_SAMPLES}" \
  --seed "${SEED}" \
  --risk-body-inflation "${Q0_BODY_INFLATION_M}" \
  --min-obstacle-clearance "${Q0_REQUIRED_CLEARANCE_M}" \
  --startup-prior-inflation "${STARTUP_PRIOR_INFLATION_M}" \
  --min-startup-prior-clearance "${STARTUP_PRIOR_REQUIRED_CLEARANCE_M}" \
  --output-json "${Q0_SOURCE_POOL}"

echo
echo "================================================================"
echo "BUILDING 30 TARGET × RANDOM-q0 CROSS PAIRS"
echo "================================================================"

"${CONDA_EXE}" run -n "${OFFLINE_GEOMETRY_CONDA_ENV}" \
  python scripts/build_phase_e_random_q0_cross_pairs.py \
  --repo "${REPO}" \
  --target-pool "${TARGET_POOL}" \
  --q0-source-pool "${Q0_SOURCE_POOL}" \
  --world "${WORLD_FILE}" \
  --body-inflation "${Q0_BODY_INFLATION_M}" \
  --required-q0-clearance "${Q0_REQUIRED_CLEARANCE_M}" \
  --startup-prior-inflation "${STARTUP_PRIOR_INFLATION_M}" \
  --required-startup-prior-clearance "${STARTUP_PRIOR_REQUIRED_CLEARANCE_M}" \
  --min-start-goal-ee-distance "${MIN_START_GOAL_EE_DISTANCE_M}" \
  --seed "${SEED}" \
  --output-json "${CASE_FILE}"

mapfile -t CASES < <(
  python3 - "${CASE_FILE}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for c in d.get("cases",[]):
    print(c["case_id"])
PY
)

if [[ "${#CASES[@]}" -ne 30 ]]; then
  echo "[ERROR] generated case file has ${#CASES[@]} cases, expected 30"
  exit 4
fi

if [[ "${BUILD_ONLY}" == "true" || "${BUILD_ONLY}" == "1" ]]; then
  echo
  echo "================================================================"
  echo "BUILD-ONLY PREFLIGHT COMPLETE"
  echo "================================================================"
  echo "[OK] Pinocchio environment verified"
  echo "[OK] direct random-q0 pool generated in obstacle world"
  echo "[OK] 30 distinct q0 configurations assigned to 30 targets"
  echo "[CASE FILE] ${CASE_FILE}"
  echo "No Gazebo/ROS experiment was started."
  echo "================================================================"
  exit 0
fi

cat > "${ROOT}/experiment_metadata.txt" <<EOF
benchmark=phase_e_random_feasible_q0_cross_pairs
git_head=$(git rev-parse HEAD)
git_branch=$(git branch --show-current)
target_pool=${TARGET_POOL}
q0_source_pool=${Q0_SOURCE_POOL}
q0_sampling=direct_joint_space_random
case_file=${CASE_FILE}
case_count=${#CASES[@]}
world_file=${WORLD_FILE}
confidence_map_config=${CONFIDENCE_MAP_CONFIG_FILE}
seed=${SEED}
offline_geometry_conda_env=${OFFLINE_GEOMETRY_CONDA_ENV}
q0_body_inflation_m=${Q0_BODY_INFLATION_M}
q0_required_obstacle_clearance_m=${Q0_REQUIRED_CLEARANCE_M}
startup_prior_inflation_m=${STARTUP_PRIOR_INFLATION_M}
startup_prior_required_clearance_m=${STARTUP_PRIOR_REQUIRED_CLEARANCE_M}
min_start_goal_ee_distance_m=${MIN_START_GOAL_EE_DISTANCE_M}
run_seconds_watchdog=${RUN_SECONDS}
early_stop_on_goal=${EARLY_STOP_ON_GOAL}
tof_fusion_enabled=true
execution_gcdf_audit_enabled=true
vbc_swept_volume_margin_m=0.0
frontier_steering_enabled=true
vbc_gated_frontier_step_enabled=true
startup_body_prior_semantics=static_once_from_actual_random_q0
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
  local src="$1" dst="$2"
  [[ -f "${src}" ]] && cp -f "${src}" "${dst}"
}

FILES=(
  joint_states.csv
  task_trajectory.csv
  committed_trajectory.csv
  regime_summary.csv
  visibility_acquisition_summary.csv
  candidate_vbc_summary.csv
  execution_vbc_summary.csv
  verification_outcome.csv
  commit_summary.csv
  tracker_summary.csv
  local_planner_summary.csv
  nominal_progress_summary.csv
  blocker_stack_summary.csv
  waypoint_schedule_summary.csv
  visibility_frontier_summary.csv
  e3_summary.csv
  tof_fusion_summary.csv
  execution_gcdf_selector_summary.csv
  execution_gcdf_safety_summary.csv
  execution_gcdf_hard_hold.csv
  goal_stop_status.json
  tracker_execution_breakdown.json
  runtime_semantics.txt
)

echo
echo "================================================================"
echo "PHASE-E RANDOM FEASIBLE q0 CROSS-PAIR EXPERIMENT"
echo "================================================================"
echo "cases       : ${#CASES[@]}"
echo "world       : obstacle world"
echo "q0 source   : direct joint-space random feasible samples"
echo "VBC margin  : raw body, +0.0 m"
echo "steering    : canonical local frontier/q_vis (not full-q_vis experiment)"
echo "watchdog    : ${RUN_SECONDS}s"
echo "================================================================"

for CASE_ID in "${CASES[@]}"; do
  echo
  echo "================================================================"
  echo "[RUN] ${CASE_ID}"
  python3 - "${CASE_FILE}" "${CASE_ID}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
c=next(x for x in d["cases"] if x["case_id"]==sys.argv[2])
print("target case :", c["case_id"])
print("q0 source   :", c["initial_q_source_case_id"])
print("initial_q   :", c["initial_q"])
print("start-goal EE distance:", c["cross_pair_start_goal_ee_distance_m"])
print("q0 collision clearance:", c["initial_q_source_collision_clearance_m"])
print("startup prior clearance:", c["initial_q_source_startup_prior_clearance_m"])
PY
  echo "================================================================"

  cleanup_ros

  RUN_ID="${CASE_ID}_random_q0_${STAMP}_${SHORT}"
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
    EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL}" \
    GAZEBO_GUI="${GAZEBO_GUI}" \
    USE_RVIZ="${USE_RVIZ}" \
    VBC_SWEPT_VOLUME_MARGIN_M=0.0 \
    FRONTIER_STEERING_ENABLED=true \
    VBC_GATED_FRONTIER_STEP_ENABLED=true \
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
      --method "phase_e_random_q0_cross_pairs" \
      --trial-id "${BATCH_ID}" \
      --output-json "${SUMMARY_DIR}/${CASE_ID}.json" \
      >> "${LOG_DIR}/${CASE_ID}.log" 2>&1
    EVAL_RC=$?
    set -e

    for name in "${FILES[@]}"; do
      copy_if_exists "${RUN_ROOT}/run/${name}" "${CASE_ART}/${name}"
    done
    copy_if_exists "${RUN_ROOT}/c5_4_local_sparse_scp_summary.json" \
      "${CASE_ART}/c5_4_local_sparse_scp_summary.json"
  else
    EVAL_RC=2
  fi

  if [[ ! -f "${SUMMARY_DIR}/${CASE_ID}.json" ]]; then
    python3 - "${SUMMARY_DIR}/${CASE_ID}.json" "${CASE_ID}" "${RUN_RC}" "${EVAL_RC}" <<'PY'
import json,sys
path,cid,rr,er=sys.argv[1:]
json.dump({
  "case_id":cid,
  "task_success":False,
  "overall_safe":False,
  "benchmark_runner_failure":True,
  "runner_return_code":int(rr),
  "evaluator_return_code":int(er),
},open(path,"w"),indent=2)
PY
  fi

  rm -f "${CASE_ZIP}"
done

# Aggregate ordinary task/safety results and attach q0-source metadata.
python3 - "${SUMMARY_DIR}" "${CASE_FILE}" "${FINAL_JSON}" "${FINAL_CSV}" <<'PY'
import csv,glob,json,os,sys
summary_dir,case_file,out_json,out_csv=sys.argv[1:]
pairs=json.load(open(case_file))
case_map={c["case_id"]:c for c in pairs["cases"]}
rows=[]
for p in sorted(glob.glob(os.path.join(summary_dir,"*.json"))):
    r=json.load(open(p))
    cid=r.get("case_id")
    if cid not in case_map: continue
    c=case_map[cid]
    row=dict(r)
    row["initial_q_source_case_id"]=c["initial_q_source_case_id"]
    row["start_goal_ee_distance_m"]=c["cross_pair_start_goal_ee_distance_m"]
    row["q0_obstacle_clearance_m"]=c["initial_q_source_collision_clearance_m"]
    row["q0_startup_prior_clearance_m"]=c["initial_q_source_startup_prior_clearance_m"]
    row["initial_q"]=c["initial_q"]
    rows.append(row)
rows.sort(key=lambda r:r["case_id"])
report={
  "benchmark":"phase_e_random_feasible_q0_cross_pairs",
  "case_count":len(rows),
  "task_success_count":sum(bool(r.get("task_success")) for r in rows),
  "overall_safe_count":sum(bool(r.get("overall_safe")) for r in rows),
  "task_success_rate":(
      sum(bool(r.get("task_success")) for r in rows)/len(rows) if rows else None),
  "cases":rows,
}
json.dump(report,open(out_json,"w"),indent=2)
fields=[
  "case_id","initial_q_source_case_id","start_goal_ee_distance_m",
  "q0_obstacle_clearance_m","q0_startup_prior_clearance_m","task_success","overall_safe","time_to_success_s",
  "final_position_error_m","best_position_error_m","commit_count",
  "candidate_vbc_records","candidate_vbc_unsafe_records",
  "execution_vbc_records","execution_vbc_unsafe_records",
  "max_remaining_obligation_count","obligation_clear_events",
]
with open(out_csv,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows:
        w.writerow({k:r.get(k) for k in fields})
print(json.dumps({
  "case_count":report["case_count"],
  "task_success_count":report["task_success_count"],
  "task_success_rate":report["task_success_rate"],
  "overall_safe_count":report["overall_safe_count"],
},indent=2))
PY

# Run the same strict blocker concentration analysis automatically.
python3 scripts/analyze_phase_e_qualification_blockers.py \
  --repo "${REPO}" \
  --input "${ROOT}" \
  --output-dir "${ROOT}/blocker_analysis" \
  > "${ROOT}/blocker_analysis_terminal.txt" 2>&1 || true

python3 - "${ROOT}" "${FINAL_ZIP}" <<'PY'
import os,sys,zipfile
root,dst=sys.argv[1:]
with zipfile.ZipFile(dst,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for base,_,files in os.walk(root):
        for name in sorted(files):
            p=os.path.join(base,name)
            z.write(p,os.path.relpath(p,root))
print(dst)
PY

echo
echo "================================================================"
echo "RANDOM-q0 CROSS-PAIR EXPERIMENT COMPLETE"
echo "================================================================"
echo "[CASE FILE]    ${CASE_FILE}"
echo "[SUMMARY JSON] ${FINAL_JSON}"
echo "[SUMMARY CSV]  ${FINAL_CSV}"
echo "[BLOCKERS]     ${ROOT}/blocker_analysis/"
echo "[UPLOAD ZIP]   ${FINAL_ZIP}"
ls -lh "${FINAL_ZIP}"
