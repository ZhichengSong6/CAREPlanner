#!/usr/bin/env bash
set -euo pipefail

# Phase-E suspended OCCUPIED hotspot provenance diagnostic.
#
# Repeats the three cases that reproducibly produced non-ground execution
# HARD_HOLD blockers. The diagnostic configs change no planning, mapping, or
# self-filter thresholds. They only log:
#   [TOF_HOTSPOT_HIT]   raw organized ToF hit + nearest labeled self sphere
#   [HOTSPOT_MAP_*]     packet-level HIT/FREE evidence and occupancy transition
#   [EXECUTION_GCDF_OCCUPIED_BLOCKER] exact execution blocker voxel
#
# Stop each case after the first captured HARD_HOLD.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray_hotspot_diag.yaml}"
TOF_FUSION_CONFIG_FILE="${TOF_FUSION_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/tof_fusion_self_filter_hotspot_diag.yaml}"
RUN_SECONDS="${RUN_SECONDS:-20}"
MAX_TRIALS_PER_CASE="${MAX_TRIALS_PER_CASE:-5}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
CASES=(phase_e_goal_004 phase_e_goal_015 phase_e_goal_022)

cd "${REPO}"

for f in "${CASE_FILE}" "${WORLD_FILE}" "${CONFIDENCE_MAP_CONFIG_FILE}" "${TOF_FUSION_CONFIG_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] missing required file: ${f}" >&2
    exit 2
  fi
done
if ! [[ "${MAX_TRIALS_PER_CASE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] MAX_TRIALS_PER_CASE must be a positive integer" >&2
  exit 3
fi

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
GIT_SHORT="$(git rev-parse --short=8 HEAD)"
DIAG_ID="phase_e_hotspot_provenance_${STAMP}_${GIT_SHORT}"
ROOT="${REPO}/outputs/phase_e_hotspot_provenance/${DIAG_ID}"
ART_DIR="${ROOT}/artifacts"
SUMMARY="${ROOT}/capture_summary.txt"
ZIP="${REPO}/CAREPlanner_PHASE_E_HOTSPOT_PROVENANCE_${DIAG_ID}.zip"

rm -rf "${ROOT}"
rm -f "${ZIP}"
mkdir -p "${ART_DIR}"
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
diagnostic=phase_e_suspended_occupied_hotspot_provenance
git_head=$(git rev-parse HEAD)
case_file=${CASE_FILE}
world_file=${WORLD_FILE}
confidence_map_config=${CONFIDENCE_MAP_CONFIG_FILE}
tof_fusion_config=${TOF_FUSION_CONFIG_FILE}
run_seconds_per_trial=${RUN_SECONDS}
max_trials_per_case=${MAX_TRIALS_PER_CASE}
hotspot_x=[-0.025,0.075]
hotspot_y=[0.025,0.125]
hotspot_z=[0.225,0.425]
mapping_semantics_changed=false
self_filter_semantics_changed=false
EOF

echo "================================================================"
echo "PHASE-E HOTSPOT PROVENANCE DIAGNOSTIC"
echo "cases              : ${CASES[*]}"
echo "max trials / case  : ${MAX_TRIALS_PER_CASE}"
echo "run seconds / trial: ${RUN_SECONDS}"
echo "================================================================"

for CASE_ID in "${CASES[@]}"; do
  CAPTURED=0
  CASE_DIR="${ART_DIR}/${CASE_ID}"
  mkdir -p "${CASE_DIR}"

  for TRIAL in $(seq 1 "${MAX_TRIALS_PER_CASE}"); do
    cleanup_ros
    TRIAL_TAG=$(printf "trial_%02d" "${TRIAL}")
    RUN_ID="${CASE_ID}_hotspot_${TRIAL_TAG}_${STAMP}_${GIT_SHORT}"
    RUN_ROOT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
    RUN_LOG_ROOT="${REPO}/logs/c5_5_vbc_gcdf_regime/${RUN_ID}"
    CASE_ZIP="${REPO}/CAREPlanner_C5_RESULT_${RUN_ID}.zip"
    TRIAL_DIR="${CASE_DIR}/${TRIAL_TAG}"
    mkdir -p "${TRIAL_DIR}"

    echo ""
    echo "================================================================"
    echo "[HOTSPOT] case=${CASE_ID} trial=${TRIAL}/${MAX_TRIALS_PER_CASE}"
    echo "================================================================"

    set +e
    CASE_FILE="${CASE_FILE}"     CASE_ID="${CASE_ID}"     RUN_ID="${RUN_ID}"     RUN_SECONDS="${RUN_SECONDS}"     WORLD_FILE="${WORLD_FILE}"     CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE}"     TOF_FUSION_CONFIG_FILE="${TOF_FUSION_CONFIG_FILE}"     TOF_FUSION_ENABLED=true     EXECUTION_GCDF_AUDIT_ENABLED=true     GCDF_BODY_INFLATION_M=0.015     FORCE_ZERO_INITIAL_Q=true     EARLY_STOP_ON_GOAL=false     GAZEBO_GUI="${GAZEBO_GUI}"     USE_RVIZ="${USE_RVIZ}"       bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
    RC=$?
    set -e

    CONTROL_LOG="${RUN_LOG_ROOT}/run/controlled.log"
    BLOCKER_LINE=""
    if [[ -f "${CONTROL_LOG}" ]]; then
      BLOCKER_LINE="$(grep -F -m1 "[EXECUTION_GCDF_OCCUPIED_BLOCKER]" "${CONTROL_LOG}" || true)"
      cp -f "${CONTROL_LOG}" "${TRIAL_DIR}/controlled.log"

      grep -F "[TOF_HOTSPOT_HIT]" "${CONTROL_LOG}"         > "${TRIAL_DIR}/tof_hotspot_hits.log" || true
      grep -F "[HOTSPOT_MAP_PACKET]" "${CONTROL_LOG}"         > "${TRIAL_DIR}/hotspot_map_packets.log" || true
      grep -F "[HOTSPOT_MAP_HIT]" "${CONTROL_LOG}"         > "${TRIAL_DIR}/hotspot_map_hits.log" || true
      grep -F "[EXECUTION_GCDF_OCCUPIED_BLOCKER]" "${CONTROL_LOG}"         > "${TRIAL_DIR}/execution_blockers.log" || true
    fi

    if [[ -d "${RUN_ROOT}/run" ]]; then
      for name in         execution_gcdf_safety_summary.csv         execution_gcdf_hard_hold.csv         e3_summary.csv         tof_fusion_summary.csv         joint_states.csv         regime_summary.csv; do
        [[ -f "${RUN_ROOT}/run/${name}" ]] &&           cp -f "${RUN_ROOT}/run/${name}" "${TRIAL_DIR}/${name}"
      done
    fi

    TOF_COUNT=0
    MAP_PACKET_COUNT=0
    MAP_HIT_COUNT=0
    [[ -f "${TRIAL_DIR}/tof_hotspot_hits.log" ]] &&       TOF_COUNT="$(wc -l < "${TRIAL_DIR}/tof_hotspot_hits.log")"
    [[ -f "${TRIAL_DIR}/hotspot_map_packets.log" ]] &&       MAP_PACKET_COUNT="$(wc -l < "${TRIAL_DIR}/hotspot_map_packets.log")"
    [[ -f "${TRIAL_DIR}/hotspot_map_hits.log" ]] &&       MAP_HIT_COUNT="$(wc -l < "${TRIAL_DIR}/hotspot_map_hits.log")"

    {
      echo "case=${CASE_ID} trial=${TRIAL} rc=${RC} captured=$([[ -n "${BLOCKER_LINE}" ]] && echo 1 || echo 0) tof_hotspot_hits=${TOF_COUNT} map_packets=${MAP_PACKET_COUNT} map_hits=${MAP_HIT_COUNT}"
      [[ -n "${BLOCKER_LINE}" ]] && echo "${BLOCKER_LINE}"
      echo ""
    } >> "${SUMMARY}"

    rm -f "${CASE_ZIP}"

    if [[ -n "${BLOCKER_LINE}" ]]; then
      echo "[CAPTURED] ${CASE_ID} trial=${TRIAL}"
      CAPTURED=1
      break
    fi
    echo "[NO HARD_HOLD] ${CASE_ID} trial=${TRIAL}"
  done

  if [[ "${CAPTURED}" -eq 0 ]]; then
    echo "[EXHAUSTED] ${CASE_ID}: no HARD_HOLD captured"
  fi
done

cleanup_ros

# Build a compact chronological provenance report around every captured blocker.
python3 - "${ART_DIR}" "${ROOT}/provenance_report.txt" <<'PY'
import glob
import os
import re
import sys

art, out = sys.argv[1:]
stamp_re = re.compile(r"stamp=([0-9]+(?:\.[0-9]+)?)")

def stamp(line):
    m = stamp_re.search(line)
    return float(m.group(1)) if m else float("nan")

with open(out, "w") as dst:
    for blocker_path in sorted(glob.glob(
            os.path.join(art, "*", "trial_*", "execution_blockers.log"))):
        lines = open(blocker_path, errors="replace").read().splitlines()
        if not lines:
            continue
        trial_dir = os.path.dirname(blocker_path)
        case_id = os.path.basename(os.path.dirname(trial_dir))
        trial = os.path.basename(trial_dir)
        blocker = lines[0]
        tb = stamp(blocker)

        dst.write("=" * 78 + "\n")
        dst.write(f"{case_id} {trial}\n")
        dst.write("BLOCKER: " + blocker + "\n")

        sources = []
        for name in (
            "tof_hotspot_hits.log",
            "hotspot_map_packets.log",
            "hotspot_map_hits.log",
        ):
            p = os.path.join(trial_dir, name)
            if not os.path.isfile(p):
                continue
            for line in open(p, errors="replace"):
                t = stamp(line)
                # Keep the full pre-history plus a short post-event window.
                if t == t and tb == tb and t <= tb + 0.30:
                    sources.append((t, name, line.rstrip()))
        sources.sort(key=lambda x: x[0])

        dst.write("\nCHRONOLOGICAL HOTSPOT EVIDENCE:\n")
        for t, name, line in sources:
            dst.write(f"{t:10.6f} {name}: {line}\n")
        dst.write("\n")

print(out)
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
echo "================ HOTSPOT PROVENANCE COMPLETE ================"
cat "${SUMMARY}"
echo "[PROVENANCE] ${ROOT}/provenance_report.txt"
echo "[UPLOAD ZIP] ${ZIP}"
ls -lh "${ZIP}"
