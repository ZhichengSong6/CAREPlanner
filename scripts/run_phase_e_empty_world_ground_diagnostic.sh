#!/usr/bin/env bash
set -euo pipefail

# Repeated-capture diagnostic for the five Phase-E empty-world cases that
# previously entered execution-GCDF OCCUPIED HARD_HOLD.
#
# No mapping/planning/safety semantics are changed. Each case is rerun from
# q0=0 until the first [EXECUTION_GCDF_OCCUPIED_BLOCKER] is captured or the
# per-case trial budget is exhausted. Once captured, later trials for that case
# are skipped.
#
# Primary question:
#   Are the intermittent occupied blockers bottom-layer / ground-plane voxels?

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"
RUN_SECONDS="${RUN_SECONDS:-20}"
MAX_TRIALS_PER_CASE="${MAX_TRIALS_PER_CASE:-5}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
CASES=(phase_e_goal_004 phase_e_goal_015 phase_e_goal_021 phase_e_goal_022 phase_e_goal_027)

cd "${REPO}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "[ERROR] case file missing: ${CASE_FILE}" >&2
  exit 2
fi
if [[ ! -f "${WORLD_FILE}" ]]; then
  echo "[ERROR] world missing: ${WORLD_FILE}" >&2
  exit 3
fi
if ! [[ "${MAX_TRIALS_PER_CASE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] MAX_TRIALS_PER_CASE must be a positive integer" >&2
  exit 4
fi

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
GIT_SHORT="$(git rev-parse --short=8 HEAD)"
DIAG_ID="phase_e_empty_ground_repeat_${STAMP}_${GIT_SHORT}"
ROOT="${REPO}/outputs/phase_e_empty_ground_diagnostic/${DIAG_ID}"
LOG_DIR="${ROOT}/logs"
ART_DIR="${ROOT}/artifacts"
SUMMARY="${ROOT}/occupied_blockers.txt"
CAPTURE_CSV="${ROOT}/capture_summary.csv"
ZIP="${REPO}/CAREPlanner_PHASE_E_EMPTY_GROUND_DIAGNOSTIC_${DIAG_ID}.zip"

rm -rf "${ROOT}"
rm -f "${ZIP}"
mkdir -p "${LOG_DIR}" "${ART_DIR}"
: > "${SUMMARY}"

cleanup_ros() {
  local n
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  sleep 0.5
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${n}" 2>/dev/null || true
  done
  rm -f /tmp/care_collision_cdf_gpu_c5_5.sock         /tmp/care_collision_cdf_gpu_c5_4.sock 2>/dev/null || true
}

cat > "${ROOT}/metadata.txt" <<EOF
diagnostic=phase_e_empty_world_ground_blocker_repeated_capture
git_head=$(git rev-parse HEAD)
git_branch=$(git branch --show-current)
case_file=${CASE_FILE}
world_file=${WORLD_FILE}
confidence_map_config=${CONFIDENCE_MAP_CONFIG_FILE}
run_seconds_per_trial=${RUN_SECONDS}
max_trials_per_case=${MAX_TRIALS_PER_CASE}
force_exact_zero_initial_q=true
ray_persistent_occupied_expected=true
cases=${CASES[*]}
question=is intermittent execution-GCDF OCCUPIED HARD_HOLD caused by ground/bottom-layer voxels
EOF

echo "================================================================"
echo "PHASE-E EMPTY-WORLD OCCUPIED BLOCKER REPEATED CAPTURE"
echo "cases              : ${#CASES[@]}"
echo "max trials / case  : ${MAX_TRIALS_PER_CASE}"
echo "run seconds / trial: ${RUN_SECONDS}"
echo "world              : ${WORLD_FILE}"
echo "================================================================"

for CASE_ID in "${CASES[@]}"; do
  echo ""
  echo "################################################################"
  echo "[CASE] ${CASE_ID}"
  echo "################################################################"

  CAPTURED=0
  CASE_ART_ROOT="${ART_DIR}/${CASE_ID}"
  mkdir -p "${CASE_ART_ROOT}"

  for TRIAL in $(seq 1 "${MAX_TRIALS_PER_CASE}"); do
    echo ""
    echo "================================================================"
    echo "[GROUND REPEAT] case=${CASE_ID} trial=${TRIAL}/${MAX_TRIALS_PER_CASE}"
    echo "================================================================"

    cleanup_ros

    TRIAL_TAG=$(printf "trial_%02d" "${TRIAL}")
    RUN_ID="${CASE_ID}_ground_repeat_${TRIAL_TAG}_${STAMP}_${GIT_SHORT}"
    RUN_ROOT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
    CASE_ZIP="${REPO}/CAREPlanner_C5_RESULT_${RUN_ID}.zip"
    LOG="${LOG_DIR}/${CASE_ID}_${TRIAL_TAG}.log"
    CASE_ART="${CASE_ART_ROOT}/${TRIAL_TAG}"
    mkdir -p "${CASE_ART}"

    set +e
    (
      CASE_FILE="${CASE_FILE}"       CASE_ID="${CASE_ID}"       RUN_ID="${RUN_ID}"       RUN_SECONDS="${RUN_SECONDS}"       WORLD_FILE="${WORLD_FILE}"       CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE}"       TOF_FUSION_ENABLED=true       EXECUTION_GCDF_AUDIT_ENABLED=true       GCDF_BODY_INFLATION_M=0.015       FORCE_ZERO_INITIAL_Q=true       EARLY_STOP_ON_GOAL=false       GAZEBO_GUI="${GAZEBO_GUI}"       USE_RVIZ="${USE_RVIZ}"       bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
    ) > >(tee "${LOG}") 2>&1
    RC=$?
    set -e

    BLOCKER_LINE="$(grep -F -m1 "[EXECUTION_GCDF_OCCUPIED_BLOCKER]" "${LOG}" || true)"

    {
      echo "case=${CASE_ID} trial=${TRIAL} runner_rc=${RC} captured=$([[ -n "${BLOCKER_LINE}" ]] && echo 1 || echo 0)"
      if [[ -n "${BLOCKER_LINE}" ]]; then
        echo "${BLOCKER_LINE}"
      fi
      echo ""
    } >> "${SUMMARY}"

    if [[ -d "${RUN_ROOT}/run" ]]; then
      for name in         execution_gcdf_safety_summary.csv         execution_gcdf_selector_summary.csv         execution_gcdf_hard_hold.csv         e3_summary.csv         tof_fusion_summary.csv         joint_states.csv         regime_summary.csv         local_planner_summary.csv; do
        [[ -f "${RUN_ROOT}/run/${name}" ]] &&           cp -f "${RUN_ROOT}/run/${name}" "${CASE_ART}/${name}"
      done
    fi

    rm -f "${CASE_ZIP}"

    if [[ -n "${BLOCKER_LINE}" ]]; then
      echo "[CAPTURED] ${CASE_ID} on trial ${TRIAL}"
      CAPTURED=1
      break
    fi

    echo "[NO CAPTURE] ${CASE_ID} trial ${TRIAL}"
  done

  if [[ "${CAPTURED}" -eq 0 ]]; then
    echo "[EXHAUSTED] ${CASE_ID}: no HARD_HOLD in ${MAX_TRIALS_PER_CASE} trials"
  fi
done

cleanup_ros

python3 - "${SUMMARY}" "${CAPTURE_CSV}" "${MAX_TRIALS_PER_CASE}" <<'PY'
import csv
import re
import sys

summary_path, out_csv, max_trials = sys.argv[1:]
max_trials = int(max_trials)

case_re = re.compile(r"^case=(\S+) trial=(\d+) runner_rc=(-?\d+) captured=(\d+)")
block_re = re.compile(
    r"voxel=\[([^,]+),([^,]+),([^\]]+)\].*?"
    r"pair_index=(\d+).*?"
    r"timestep=(-?\d+).*?"
    r"raw_center_clearance=([^ ]+).*?"
    r"voxel_volume_clearance=([^ ]+).*?"
    r"learned_d=([^ ]+).*?"
    r"ground_band_candidate=(\d+).*?"
    r"hard_occupied_voxel_count=(\d+)"
)

rows = {}
pending = None
with open(summary_path, errors="replace") as f:
    for raw in f:
        line = raw.strip()
        m = case_re.match(line)
        if m:
            cid, trial, rc, captured = m.groups()
            trial = int(trial)
            rec = rows.setdefault(cid, {
                "case_id": cid,
                "trials_run": 0,
                "captured": 0,
                "capture_trial": None,
                "runner_rc": None,
                "voxel_x": None,
                "voxel_y": None,
                "voxel_z": None,
                "pair_index": None,
                "timestep": None,
                "raw_center_clearance_m": None,
                "voxel_volume_clearance_m": None,
                "learned_d": None,
                "ground_band_candidate": None,
                "hard_occupied_voxel_count": None,
            })
            rec["trials_run"] = max(rec["trials_run"], trial)
            rec["runner_rc"] = int(rc)
            pending = rec if int(captured) else None
            continue

        if pending is not None and "[EXECUTION_GCDF_OCCUPIED_BLOCKER]" in line:
            b = block_re.search(line)
            if b:
                (x, y, z, pair, timestep, rawc, volc, learned,
                 ground, count) = b.groups()
                pending.update({
                    "captured": 1,
                    "capture_trial": pending["trials_run"],
                    "voxel_x": float(x),
                    "voxel_y": float(y),
                    "voxel_z": float(z),
                    "pair_index": int(pair),
                    "timestep": int(timestep),
                    "raw_center_clearance_m": float(rawc),
                    "voxel_volume_clearance_m": float(volc),
                    "learned_d": float(learned),
                    "ground_band_candidate": int(ground),
                    "hard_occupied_voxel_count": int(count),
                })
            pending = None

fields = [
    "case_id", "trials_run", "captured", "capture_trial", "runner_rc",
    "voxel_x", "voxel_y", "voxel_z", "pair_index", "timestep",
    "raw_center_clearance_m", "voxel_volume_clearance_m", "learned_d",
    "ground_band_candidate", "hard_occupied_voxel_count",
]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for cid in sorted(rows):
        w.writerow(rows[cid])

captured = [r for r in rows.values() if r["captured"]]
ground = [r for r in captured if r["ground_band_candidate"] == 1]
print("==============================================================")
print("[CAPTURE SUMMARY]")
print("cases:", len(rows))
print("captured:", len(captured))
print("ground_band:", len(ground))
print("non_ground:", len(captured) - len(ground))
for r in sorted(rows.values(), key=lambda x: x["case_id"]):
    if r["captured"]:
        print(
            "{case_id}: captured trial={capture_trial} "
            "voxel=[{voxel_x:.3f},{voxel_y:.3f},{voxel_z:.3f}] "
            "ground={ground_band_candidate}".format(**r))
    else:
        print(
            "{}: no capture after {} trials".format(
                r["case_id"], r["trials_run"]))
PY

python3 - "${ROOT}" "${ZIP}" <<'PY'
import os, sys, zipfile
root, dst = sys.argv[1:]
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for base, _, files in os.walk(root):
        for name in files:
            p = os.path.join(base, name)
            z.write(p, os.path.relpath(p, root))
print(dst)
PY

echo ""
echo "================ REPEATED GROUND DIAGNOSTIC COMPLETE ================"
echo "[RAW BLOCKERS] ${SUMMARY}"
cat "${SUMMARY}"
echo "[CAPTURE CSV]  ${CAPTURE_CSV}"
cat "${CAPTURE_CSV}"
echo "[UPLOAD ZIP]   ${ZIP}"
ls -lh "${ZIP}"
