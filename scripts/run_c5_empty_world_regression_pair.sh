#!/usr/bin/env bash
set -euo pipefail

# Pairwise C5 regression test:
#   known-good baseline 72e53a82 vs current checkout
# under the same empty-world / ideal-FOV case_014 setup.
#
# The baseline is executed from a detached git worktree so the user's main
# checkout is never switched or modified.

MAIN_REPO="${MAIN_REPO:-/home/zhicheng/Project/CAREPlanner}"
BASELINE_REF="${BASELINE_REF:-72e53a822e98ae43e5560810363b7b82ae68d007}"
CASE_ID="${CASE_ID:-case_014}"
RUN_SECONDS="${RUN_SECONDS:-20}"
GAZEBO_GUI="${GAZEBO_GUI:-true}"
USE_RVIZ="${USE_RVIZ:-false}"
EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL:-true}"

cd "${MAIN_REPO}"
CURRENT_REF="$(git rev-parse HEAD)"
CURRENT_SHORT="$(git rev-parse --short=8 HEAD)"
BASELINE_FULL="$(git rev-parse "${BASELINE_REF}^{commit}")"
BASELINE_SHORT="$(git rev-parse --short=8 "${BASELINE_FULL}")"
STAMP="$(date +%Y%m%d-%H%M%S)"

ROOT="${MAIN_REPO}/outputs/c5_empty_world_regression/${STAMP}_${BASELINE_SHORT}_vs_${CURRENT_SHORT}"
BASE_WT="/tmp/CAREPlanner_regression_${BASELINE_SHORT}_${STAMP}"
mkdir -p "${ROOT}"

# ROS/Gazebo must use the system ROS Python. The research envs are activated
# explicitly by the child runners only for their GPU/NCDF subprocesses.
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
  echo "[REGRESSION ENV] deactivated inherited conda env: ${_before}"
fi

cleanup_ros() {
  local n
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  sleep 1
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${n}" 2>/dev/null || true
  done
  rm -f /tmp/care_collision_cdf_gpu_c5_5.sock \
        /tmp/care_collision_cdf_gpu_c5_4.sock 2>/dev/null || true
}

cleanup_all() {
  cleanup_ros
  if [[ -d "${BASE_WT}" ]]; then
    git -C "${MAIN_REPO}" worktree remove --force "${BASE_WT}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_all EXIT INT TERM

if timeout 2 rosnode list >/dev/null 2>&1; then
  echo "[ERROR] A ROS master is already running. Stop the old CAREPlanner/Gazebo run first."
  exit 5
fi

echo "================================================================"
echo "C5 EMPTY-WORLD REGRESSION"
echo "baseline : ${BASELINE_FULL}"
echo "current  : ${CURRENT_REF}"
echo "case     : ${CASE_ID}"
echo "duration : ${RUN_SECONDS}s"
echo "output   : ${ROOT}"
echo "================================================================"

git worktree add --detach "${BASE_WT}" "${BASELINE_FULL}"

# Some learned-model checkpoints may intentionally be kept outside historical
# git revisions. Reuse the exact current checkpoint file if the old worktree
# does not contain it; code/config still comes from the historical revision.
link_if_missing() {
  local rel="$1"
  local src="${MAIN_REPO}/${rel}"
  local dst="${BASE_WT}/${rel}"
  if [[ ! -e "${dst}" && -e "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    ln -s "${src}" "${dst}"
    echo "[BASELINE] linked external artifact: ${rel}"
  fi
}
link_if_missing "src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt"
link_if_missing "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt"

run_one() {
  local label="$1"
  local repo="$2"
  local ref_short="$3"
  local run_id="regression_${label}_${CASE_ID}_${STAMP}_${ref_short}"
  local log="${ROOT}/${label}.log"

  echo ""
  echo "================================================================"
  echo "[RUN ${label}] repo=${repo}"
  echo "[RUN ${label}] head=$(git -C "${repo}" rev-parse HEAD)"
  echo "[RUN ${label}] run_id=${run_id}"
  echo "================================================================"

  cleanup_ros

  set +e
  (
    cd "${repo}"
    REPO="${repo}" \
    CASE_ID="${CASE_ID}" \
    RUN_ID="${run_id}" \
    RUN_SECONDS="${RUN_SECONDS}" \
    GAZEBO_GUI="${GAZEBO_GUI}" \
    USE_RVIZ="${USE_RVIZ}" \
    EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL}" \
    bash scripts/run_and_pack_c5_5_vbc_gcdf_regime.sh
  ) 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e

  local run_dir="${repo}/outputs/c5_5_vbc_gcdf_regime/${run_id}/run"
  local eval_json="${ROOT}/${label}_evaluation.json"

  if [[ -d "${run_dir}" ]]; then
    # Use one evaluator implementation for both revisions so the comparison
    # metric itself does not change across commits.
    set +e
    (
      cd "${MAIN_REPO}"
      python3 scripts/evaluate_phase_d_run.py \
        --run-dir "${run_dir}" \
        --case-id "${CASE_ID}" \
        --method "c5_regression_${label}" \
        --trial-id "${STAMP}" \
        --output-json "${eval_json}"
    ) >> "${log}" 2>&1
    local eval_rc=$?
    set -e
  else
    local eval_rc=2
  fi

  python3 - "${ROOT}/${label}_status.json" "${rc}" "${eval_rc}" "${run_dir}" "${eval_json}" <<'PY'
import json, os, sys
out, run_rc, eval_rc, run_dir, eval_json = sys.argv[1:]
payload = {
    "runner_return_code": int(run_rc),
    "evaluator_return_code": int(eval_rc),
    "run_dir": run_dir,
    "evaluation_json": eval_json if os.path.isfile(eval_json) else None,
}
json.dump(payload, open(out, "w"), indent=2)
PY

  # Copy the compact, high-value artifacts before the temporary worktree is
  # removed.
  local keep="${ROOT}/${label}_artifacts"
  mkdir -p "${keep}"
  for f in \
      regime_summary.csv \
      visibility_acquisition_summary.csv \
      candidate_vbc_summary.csv \
      execution_vbc_summary.csv \
      verification_outcome.csv \
      commit_summary.csv \
      tracker_summary.csv \
      joint_states.csv \
      goal_stop_status.json \
      tracker_execution_breakdown.json; do
    if [[ -f "${run_dir}/${f}" ]]; then
      cp -f "${run_dir}/${f}" "${keep}/${f}"
    fi
  done

  return 0
}

run_one "baseline" "${BASE_WT}" "${BASELINE_SHORT}"
run_one "current" "${MAIN_REPO}" "${CURRENT_SHORT}"

python3 - "${ROOT}" "${BASELINE_FULL}" "${CURRENT_REF}" "${CASE_ID}" <<'PY'
import csv
import json
import math
import os
import re
import sys

root, baseline_ref, current_ref, case_id = sys.argv[1:]
TOK = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

def evaluation(label):
    p = os.path.join(root, f"{label}_evaluation.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)

def status(label):
    p = os.path.join(root, f"{label}_status.json")
    with open(p) as f:
        return json.load(f)

def token_rows(label, name):
    p = os.path.join(root, f"{label}_artifacts", name)
    if not os.path.isfile(p):
        return []
    out = []
    with open(p, newline="", errors="replace") as f:
        rd = csv.reader(f)
        h = next(rd, [])
        if not h:
            return out
        di = h.index("field.data") if "field.data" in h else 1
        for row in rd:
            if len(row) > di:
                d = dict(TOK.findall(",".join(row[di:])))
                if d:
                    out.append(d)
    return out

def as_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def summarize(label):
    ev = evaluation(label) or {}
    st = status(label)
    regime = token_rows(label, "regime_summary.csv")
    acq = token_rows(label, "visibility_acquisition_summary.csv")
    commit = token_rows(label, "commit_summary.csv")
    cand = token_rows(label, "candidate_vbc_summary.csv")
    execv = token_rows(label, "execution_vbc_summary.csv")
    tracker = token_rows(label, "tracker_summary.csv")

    c_last = commit[-1] if commit else {}
    a_last = acq[-1] if acq else {}
    r_last = regime[-1] if regime else {}

    return {
        "runner_return_code": st["runner_return_code"],
        "evaluator_return_code": st["evaluator_return_code"],
        "task_success": ev.get("task_success"),
        "goal_reached": ev.get("goal_reached"),
        "final_position_error_m": ev.get("final_position_error_m"),
        "best_position_error_m": ev.get("best_position_error_m"),
        "final_orientation_error_rad": ev.get("final_orientation_error_rad"),
        "final_regime_state": r_last.get("state"),
        "commit_count": max(
            [as_int(r.get("commit_count"), 0) for r in commit] or [0]),
        "candidate_vbc_records": len(cand),
        "candidate_vbc_unsafe_records": sum(
            1 for r in cand if r.get("has_violation") == "1"),
        "execution_vbc_records": len(execv),
        "execution_vbc_unsafe_records": sum(
            1 for r in execv if r.get("has_violation") == "1"),
        "acquisition_complete_ever": any(
            r.get("complete") == "1" for r in acq),
        "remaining_obligations_final": as_int(
            a_last.get("remaining_obligation_count"), -1) if a_last else None,
        "seen_obligations_final": as_int(
            a_last.get("seen_obligation_count"), -1) if a_last else None,
        "tracker_records": len(tracker),
        "last_commit_source": c_last.get("last_source"),
    }

baseline = summarize("baseline")
current = summarize("current")

if baseline.get("task_success") is True and current.get("task_success") is False:
    verdict = "REGRESSION_CONFIRMED"
elif baseline.get("task_success") is False:
    verdict = "BASELINE_NOT_REPRODUCED"
elif baseline.get("task_success") is True and current.get("task_success") is True:
    verdict = "NO_TASK_SUCCESS_REGRESSION"
else:
    verdict = "INCONCLUSIVE"

report = {
    "test": "c5_empty_world_case014_pair",
    "case_id": case_id,
    "baseline_ref": baseline_ref,
    "current_ref": current_ref,
    "baseline": baseline,
    "current": current,
    "verdict": verdict,
}
out = os.path.join(root, "regression_compare.json")
json.dump(report, open(out, "w"), indent=2)
print("\n================ REGRESSION COMPARISON ================")
print(json.dumps(report, indent=2))
print(f"\n[COMPARE JSON] {out}")
print("=======================================================")
PY

echo ""
echo "[DONE] Regression artifacts: ${ROOT}"
echo "[NEXT] Send regression_compare.json plus baseline.log/current.log if needed."
