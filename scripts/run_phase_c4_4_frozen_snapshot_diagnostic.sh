#!/usr/bin/env bash
set -u -o pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-8.0}"
CONFIG_FILE="${CONFIG_FILE:-${REPO}/src/egocentric_arm_planner/config/planner_phase1_c4_3_planner.yaml}"
BASE_OUT="${OUT:-${REPO}/outputs/phase_c4_4_frozen_snapshot_diagnostic/${CASE_ID}}"
BASE_LOG="${LOG:-${REPO}/logs/phase_c4_4_frozen_snapshot_diagnostic/${CASE_ID}}"
DIAG_OUT="${BASE_OUT}/frozen_snapshot"
DIAG_LOG="${BASE_LOG}/frozen_snapshot_diagnostic.log"

cd "${REPO}" || exit 1
source devel/setup.bash

# The underlying C4.4 runner owns Gazebo/ROS cleanup. It also removes BASE_OUT
# and BASE_LOG at startup, so do not create diagnostic files until it is ready.
CASE_ID="${CASE_ID}" \
RUN_SECONDS="${RUN_SECONDS}" \
CONFIG_FILE="${CONFIG_FILE}" \
OUT="${BASE_OUT}" \
LOG="${BASE_LOG}" \
bash scripts/run_phase_c4_4_verified_regime_smoke.sh &
RUNNER_PID=$!
DIAG_PID=""

cleanup_diag() {
  if [ -n "${DIAG_PID}" ] && kill -0 "${DIAG_PID}" 2>/dev/null; then
    kill -INT -- "-${DIAG_PID}" 2>/dev/null || true
    sleep 0.15
    kill -TERM -- "-${DIAG_PID}" 2>/dev/null || true
  fi
}
trap cleanup_diag EXIT INT TERM

READY=0
for _ in $(seq 1 400); do
  if ! kill -0 "${RUNNER_PID}" 2>/dev/null; then break; fi
  NODES="$(rosnode list 2>/dev/null || true)"
  if echo "${NODES}" | grep -q '^/velocity_qp_mpc_waypoint_node$' && \
     echo "${NODES}" | grep -q '^/c4_4_verified_regime_manager$' && \
     rostopic list 2>/dev/null | grep -q '^/care_planner/confidence_map'; then
    READY=1
    break
  fi
  sleep 0.10
done

if [ "${READY}" != "1" ]; then
  echo "[ERROR] main C4.4 system was not ready for frozen diagnostic"
  wait "${RUNNER_PID}" || true
  exit 1
fi

mkdir -p "${DIAG_OUT}" "${BASE_LOG}"
setsid roslaunch egocentric_arm_planner c4_4_frozen_snapshot_repair_diagnostic.launch \
  config_file:="${CONFIG_FILE}" \
  output_dir:="${DIAG_OUT}" \
  > "${DIAG_LOG}" 2>&1 &
DIAG_PID=$!

DIAG_READY=0
for _ in $(seq 1 200); do
  if ! kill -0 "${DIAG_PID}" 2>/dev/null; then break; fi
  NODES="$(rosnode list 2>/dev/null || true)"
  if echo "${NODES}" | grep -q '^/frozen_snapshot_repair_diagnostic$' && \
     echo "${NODES}" | grep -q '^/frozen_snapshot_vbc/trajectory_vbc_selector_node$'; then
    DIAG_READY=1
    break
  fi
  sleep 0.05
done

if [ "${DIAG_READY}" != "1" ]; then
  echo "[ERROR] frozen snapshot diagnostic nodes did not start"
  tail -n 160 "${DIAG_LOG}" 2>/dev/null || true
  wait "${RUNNER_PID}" || true
  exit 1
fi

echo "[DIAG] frozen snapshot diagnostic armed at first actual REPAIR MPC cycle"
echo "[DIAG] horizons: 0.5, 1.0, 1.5, 2.0 s; same dt/limits/PIQP repair objective"

RESULT="${DIAG_OUT}/frozen_snapshot_repair_results.json"
while kill -0 "${RUNNER_PID}" 2>/dev/null; do
  if [ -s "${RESULT}" ]; then break; fi
  if [ -n "${DIAG_PID}" ] && ! kill -0 "${DIAG_PID}" 2>/dev/null; then break; fi
  sleep 0.05
done

if [ -s "${RESULT}" ]; then
  echo "[DIAG RESULT]"
  cat "${RESULT}"
  cleanup_diag
  DIAG_PID=""
else
  echo "[WARN] frozen diagnostic result was not completed before the main run ended"
  tail -n 200 "${DIAG_LOG}" 2>/dev/null || true
fi

wait "${RUNNER_PID}"
RUNNER_STATUS=$?

if [ -s "${RESULT}" ]; then
  echo "[FROZEN SNAPSHOT] ${RESULT}"
  echo "[FROZEN CSV]      ${DIAG_OUT}/frozen_snapshot_repair_results.csv"
  echo "[SNAPSHOT]        ${DIAG_OUT}/frozen_snapshot.csv"
else
  echo "[ERROR] missing frozen snapshot result: ${RESULT}"
  exit 2
fi

exit "${RUNNER_STATUS}"
