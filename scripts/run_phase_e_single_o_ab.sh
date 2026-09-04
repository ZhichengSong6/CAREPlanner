#!/usr/bin/env bash
set -euo pipefail

# C5.41 shared-O life/death A/B test.
#
# Same Phase-E empty-world stack in both runs:
#   - real ToF ray mapping
#   - UNKNOWN/OCCUPIED semantics
#   - blocker-aware recursion
#   - final GCDF + exact VBC
#   - E5 measured-state audit
#
# Only difference:
#   A: progressive_shared_repair_enabled=true
#   B: progressive_shared_repair_enabled=false
#
# This is intentionally a single representative case test. It is not a new
# benchmark and does not modify the frozen default C5.41 policy.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
CASE_ID="${CASE_ID:-phase_e_goal_014}"
RUN_SECONDS="${RUN_SECONDS:-30}"
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
  echo "[A/B ENV] sanitized inherited conda env (was ${_before})"
fi

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SHORT="$(git rev-parse --short=8 HEAD)"
ROOT="${REPO}/outputs/phase_e_single_o_ab/${CASE_ID}_${STAMP}_${SHORT}"
mkdir -p "${ROOT}/logs" "${ROOT}/summaries" "${ROOT}/artifacts"

cleanup_ros() {
  local n
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  sleep 0.7
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${n}" 2>/dev/null || true
  done
  rm -f /tmp/care_collision_cdf_gpu_c5_5.sock \
        /tmp/care_collision_cdf_gpu_c5_4.sock 2>/dev/null || true
}

trap cleanup_ros EXIT INT TERM

run_one() {
  local label="$1"
  local shared="$2"

  cleanup_ros

  local run_id="${CASE_ID}_single_o_ab_${label}_${STAMP}_${SHORT}"
  local run_root="${REPO}/outputs/c5_5_vbc_gcdf_regime/${run_id}"
  local log="${ROOT}/logs/${label}.log"

  echo ""
  echo "================================================================"
  echo "[A/B ${label}] case=${CASE_ID}"
  echo "[A/B ${label}] progressive_shared_repair_enabled=${shared}"
  echo "================================================================"

  set +e
  (
    CASE_FILE="${CASE_FILE}" \
    CASE_ID="${CASE_ID}" \
    RUN_ID="${run_id}" \
    RUN_SECONDS="${RUN_SECONDS}" \
    WORLD_FILE="${WORLD_FILE}" \
    CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE}" \
    TOF_FUSION_ENABLED=true \
    EXECUTION_GCDF_AUDIT_ENABLED=true \
    GCDF_BODY_INFLATION_M=0.015 \
    EARLY_STOP_ON_GOAL="${EARLY_STOP_ON_GOAL}" \
    GAZEBO_GUI="${GAZEBO_GUI}" \
    USE_RVIZ="${USE_RVIZ}" \
    PROGRESSIVE_SHARED_REPAIR_ENABLED="${shared}" \
    bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
  ) > >(tee "${log}") 2>&1
  local rc=$?
  set -e

  local eval_json="${ROOT}/summaries/${label}.json"
  local run_dir="${run_root}/run"

  if [[ -d "${run_dir}" ]]; then
    set +e
    python3 scripts/evaluate_phase_d_run.py \
      --repo "${REPO}" \
      --run-dir "${run_dir}" \
      --cases-json "${CASE_FILE}" \
      --case-id "${CASE_ID}" \
      --method "phase_e_single_o_ab_${label}" \
      --trial-id "${STAMP}" \
      --output-json "${eval_json}" \
      >> "${log}" 2>&1
    local eval_rc=$?
    set -e

    mkdir -p "${ROOT}/artifacts/${label}"
    for f in \
      regime_summary.csv \
      visibility_acquisition_summary.csv \
      blocker_stack_summary.csv \
      candidate_vbc_summary.csv \
      execution_vbc_summary.csv \
      verification_outcome.csv \
      commit_summary.csv \
      tracker_summary.csv \
      joint_states.csv \
      goal_stop_status.json; do
      [[ -f "${run_dir}/${f}" ]] && cp -f "${run_dir}/${f}" "${ROOT}/artifacts/${label}/${f}"
    done
  else
    local eval_rc=2
  fi

  python3 - "${ROOT}/summaries/${label}_status.json" "${rc}" "${eval_rc}" <<'PY'
import json, sys
path, rc, erc = sys.argv[1:]
json.dump({"runner_return_code": int(rc), "evaluator_return_code": int(erc)},
          open(path,"w"), indent=2)
PY
}

run_one "shared_on"  "true"
run_one "shared_off" "false"

python3 - "${ROOT}" "${CASE_ID}" <<'PY'
import csv, json, math, os, re, sys

root, case_id = sys.argv[1:]
TOK = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

def load_json(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)

def rows(label, name):
    p=os.path.join(root,"artifacts",label,name)
    if not os.path.isfile(p):
        return []
    out=[]
    with open(p,newline="",errors="replace") as f:
        rd=csv.reader(f); h=next(rd,[])
        if not h: return out
        di=h.index("field.data") if "field.data" in h else 1
        for r in rd:
            if len(r)<=di: continue
            d=dict(TOK.findall(",".join(r[di:])))
            if d: out.append(d)
    return out

def integer(v,d=0):
    try: return int(float(v))
    except Exception: return d

def summarize(label):
    ev=load_json(os.path.join(root,"summaries",label+".json"))
    reg=rows(label,"regime_summary.csv")
    blk=rows(label,"blocker_stack_summary.csv")
    acq=rows(label,"visibility_acquisition_summary.csv")
    com=rows(label,"commit_summary.csv")
    cand=rows(label,"candidate_vbc_summary.csv")

    b_last=blk[-1] if blk else {}
    a_last=acq[-1] if acq else {}
    r_last=reg[-1] if reg else {}

    return {
      "task_success": ev.get("task_success"),
      "time_to_success_s": ev.get("time_to_success_s"),
      "best_position_error_m": ev.get("best_position_error_m"),
      "final_position_error_m": ev.get("final_position_error_m"),
      "final_regime_state": r_last.get("state"),
      "repair_count": integer(r_last.get("repair_entry_count"),0),
      "probe_count": integer(r_last.get("probe_entry_count"),0),
      "commit_count": max([integer(r.get("commit_count"),0) for r in com] or [0]),
      "candidate_vbc_records": len([r for r in cand if r.get("trajectory_source")=="predicted"]),
      "candidate_vbc_unsafe_records": len([
          r for r in cand
          if r.get("trajectory_source")=="predicted" and r.get("has_violation")=="1"]),
      "remaining_obligations_final": integer(a_last.get("remaining_obligation_count"),-1) if a_last else None,
      "seen_obligations_final": integer(a_last.get("seen_obligation_count"),-1) if a_last else None,
      "blocker_push_count": integer(b_last.get("push_count"),0),
      "blocker_pop_count": integer(b_last.get("pop_count"),0),
      "cycle_block_count": integer(b_last.get("cycle_block_count"),0),
      "progressive_shared_enabled": integer(b_last.get("progressive_shared_enabled"),-1),
      "progressive_shared_success_count": integer(b_last.get("progressive_shared_success_count"),0),
    }

on=summarize("shared_on")
off=summarize("shared_off")

if off["task_success"] and not on["task_success"]:
    verdict="SHARED_O_LIKELY_CAUSAL"
elif (not off["task_success"] and
      off["cycle_block_count"]>0 and
      off["probe_count"]==0):
    verdict="SHARED_O_NOT_ROOT_CAUSE"
elif off["cycle_block_count"] < on["cycle_block_count"]:
    verdict="SHARED_O_AGGRAVATES_BUT_NOT_RESOLVED"
else:
    verdict="INCONCLUSIVE"

report={
  "test":"phase_e_single_o_ab",
  "case_id":case_id,
  "shared_on":on,
  "shared_off":off,
  "verdict":verdict,
}
out=os.path.join(root,"single_o_ab_summary.json")
json.dump(report,open(out,"w"),indent=2)
print("\n================ SINGLE-O A/B =================")
print(json.dumps(report,indent=2))
print(f"\n[SUMMARY] {out}")
print("================================================")
PY

python3 - "${ROOT}" "${REPO}/CAREPlanner_PHASE_E_SINGLE_O_AB_${CASE_ID}_${STAMP}_${SHORT}.zip" <<'PY'
import os, sys, zipfile
root,dst=sys.argv[1:]
with zipfile.ZipFile(dst,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for base,_,files in os.walk(root):
        for name in files:
            p=os.path.join(base,name)
            z.write(p,os.path.relpath(p,root))
print(dst)
PY

echo ""
echo "[DONE] ${ROOT}"
echo "[UPLOAD ZIP] ${REPO}/CAREPlanner_PHASE_E_SINGLE_O_AB_${CASE_ID}_${STAMP}_${SHORT}.zip"
