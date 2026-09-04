#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"

# ROS Noetic/Gazebo scripts use /usr/bin/env python3. If CAREPlanner is
# launched from a research conda env (e.g. viscdf), spawn_model and controller
# spawner inherit that interpreter and can import rospy via ROS_PYTHONPATH while
# still missing system packages such as rospkg. Normalize the parent shell here
# so every C5 caller (Phase D, Phase E, direct single-case runs) starts the ROS/
# Gazebo stack from the system environment. GPU/NCDF subprocesses explicitly
# activate their own conda env later.
if [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); then
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  _care_c5_conda_before="${CONDA_DEFAULT_ENV:-unknown}"
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); do
    conda deactivate || break
  done
  echo "[C5 ENV] sanitized inherited conda env (was ${_care_c5_conda_before}); ROS/Gazebo use system Python"
fi

CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-30.0}"

# CASE_ID is the semantic controlled-trial scenario key and MUST remain an
# exact entry from the case JSON (e.g. "case_003").  RUN_ID is only an
# artifact/experiment identity.  Conflating them previously made the case
# lookup print "unknown case_id" and silently run with empty goal fields.
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S-%3N)}"
GIT_SHORT="$(git -C "${REPO}" rev-parse --short=8 HEAD)"
RUN_ID="${RUN_ID:-${CASE_ID}_${RUN_STAMP}_${GIT_SHORT}}"
export CASE_ID RUN_ID

export CONFIG_FILE="${REPO}/src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml"
export CARE_WEIGHT="${CARE_WEIGHT:-3000.0}"
export GPU_SOCKET="${GPU_SOCKET:-/tmp/care_collision_cdf_gpu_c5_5.sock}"
# Paths are intentionally derived from RUN_ID every time.  Do not inherit a
# stale ROOT_OUT/ROOT_LOG/ZIP_PATH from a previous shell session.
export ROOT_OUT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
export ROOT_LOG="${REPO}/logs/c5_5_vbc_gcdf_regime/${RUN_ID}"
export ZIP_PATH="${REPO}/CAREPlanner_C5_RESULT_${RUN_ID}.zip"
export INITIAL_GATE_MAX_TRIES="${INITIAL_GATE_MAX_TRIES:-100}"
export INITIAL_GATE_ECHO_TIMEOUT="${INITIAL_GATE_ECHO_TIMEOUT:-0.20}"

# C5.9 execution semantics: REPAIR/PROBE optimize over a 1 s local horizon but
# build a short executable prefix with the actual braking/hold tail. That exact
# executable view must pass final GCDF and exact VBC before a single commit.
# PROBE keeps single certified execution ownership. C5.40 may preplan one raw
# look-ahead candidate while the current certified prefix executes, but that raw
# candidate is buffered until prefix completion; only then is it rebased and
# sent through final GCDF + exact VBC. NORMAL remains full-horizon GCDF/VBC.
export REPAIR_PREFIX_VERIFY="${REPAIR_PREFIX_VERIFY:-1}"
# Smooth certified handoff: REPAIR gets enough useful-motion horizon for the
# next plan+GCDF+VBC to finish before its fallback brake tail. PROBE retains
# the frozen shorter single-flight semantics.
export REPAIR_PREFIX_S="${REPAIR_PREFIX_S:-0.25}"
export PROBE_PREFIX_S="${PROBE_PREFIX_S:-0.15}"
export SMOOTH_HANDOFF_ENABLED="${SMOOTH_HANDOFF_ENABLED:-true}"
export REPAIR_BRAKE_DT_S="${REPAIR_BRAKE_DT_S:-0.05}"
export REPAIR_HOLD_S="${REPAIR_HOLD_S:-0.10}"

python3 -m py_compile \
  "${REPO}/src/egocentric_arm_planner/scripts/probe_single_flight_gate_node.py"

echo "[CASE ID] ${CASE_ID}"
echo "[RUN ID] ${RUN_ID}"
echo "[OUTPUT] ${ROOT_OUT}"
echo "[UPLOAD ZIP] ${ZIP_PATH}"
echo "[C5.5] Existing VBC/regime architecture + hard GCDF safety"
echo "[C5.5] REPAIR: q_vis objective + 5-step executable-horizon hard GCDF + hold initialization"
echo "[C5.5] PROBE_NORMAL: task objective + 5-step executable-horizon hard GCDF + safe-prefix commit"
echo "[C5.5] NORMAL: task objective + full-horizon hard GCDF"
echo "[C5.40] PROBE ROLLING: certified prefix executes while one raw next-probe candidate may preplan behind the gate"
echo "[C5.9] COMMIT GATE: exact executable prefix+brake+hold -> final GCDF -> exact VBC -> single commit"
echo "[C5.9] GCDF transport: one GPU client, independent local/final channels, final priority"
echo "[C5.9] EXECUTION: committed trajectory published once; tracker owns execution to completion"
echo "[C5.9] EXECUTION VBC: elapsed suffixes use a separate audit-only topic; tracker is never reset"
echo "[C5.40] PROBE success: matching execution token reaches certified prefix duration; brake+hold remains fallback while replacement is certified"
echo "[C5.5] REPAIR exit: actual visibility/confidence acquisition gate"
echo "[C5.5] REPAIR exact-VBC view: prefix=${REPAIR_PREFIX_S}s + brake=${REPAIR_BRAKE_DT_S}s + hold=${REPAIR_HOLD_S}s"
echo "[C5.5] PROBE base prefix: ${PROBE_PREFIX_S}s (then existing probe time scaling)"
echo "[SMOOTH] certified look-ahead handoff=${SMOOTH_HANDOFF_ENABLED}; brake+hold retained as fail-safe tail"
echo "[C5.5] NORMAL exact-VBC view: full horizon"
echo "[C5.5] config: ${CONFIG_FILE}"
echo "[C5.5] startup gate wait is fail-fast (~25 s worst case)"

CASE_ID="${CASE_ID}" RUN_SECONDS="${RUN_SECONDS}" \
  bash "${REPO}/scripts/run_and_pack_phase_c5_4_local_sparse_scp.sh"
