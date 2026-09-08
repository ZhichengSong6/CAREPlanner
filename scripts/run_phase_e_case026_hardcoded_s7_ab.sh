#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
CASE_ID="${CASE_ID:-phase_e_goal_026}"
RUN_SECONDS="${RUN_SECONDS:-60}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"

# Robust LOS-clear S7 candidate from the Case-026 per-sensor feasibility test.
S7_Q="-0.8144537806510925,0.5605988502502441,1.0171856880187988,-2.180335760116577,0.12358028441667557,-1.7687498331069946,1.1331826448440552"
TARGET="0.1,0.05,0.15"

cd "${REPO}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "[ERROR] case file missing: ${CASE_FILE}"
  exit 2
fi

# Keep ROS/Gazebo in system environment.
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

python3 -m py_compile   src/care_visibility_cdf/scripts/vbc_multi_deadline_obligation_impl.py

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SHORT="$(git rev-parse --short=8 HEAD)"
RUN_ID="${CASE_ID}_hardcoded_s7_${STAMP}_${SHORT}"
RUN_ROOT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
DIAG_ROOT="${REPO}/outputs/phase_e_case026_hardcoded_s7/${RUN_ID}"
LOG_FILE="${DIAG_ROOT}/runner.log"
SUMMARY_JSON="${DIAG_ROOT}/case026_s7_override_summary.json"
UPLOAD_ZIP="${REPO}/CAREPlanner_PHASE_E_CASE026_S7_OVERRIDE_${RUN_ID}.zip"

rm -rf "${DIAG_ROOT}"
mkdir -p "${DIAG_ROOT}/artifacts"

export CARE_DIAG_QVIS_OVERRIDE_ENABLED=true
export CARE_DIAG_QVIS_OVERRIDE_OBLIGATION_ID=7
export CARE_DIAG_QVIS_OVERRIDE_TARGET="${TARGET}"
export CARE_DIAG_QVIS_OVERRIDE_TARGET_TOL_M=0.03
export CARE_DIAG_QVIS_OVERRIDE_Q="${S7_Q}"
export CARE_DIAG_QVIS_OVERRIDE_LABEL="case026_robust_s7_nominal00888_cons00378_los_clear"

echo "================================================================"
echo "CASE 026 HARDCODED S7 Q_VIS A/B -- B RUN"
echo "case       : ${CASE_ID}"
echo "runtime    : ${RUN_SECONDS}s"
echo "world      : ${WORLD_FILE}"
echo "override   : obligation 7 @ target [${TARGET}]"
echo "S7 q       : [${S7_Q}]"
echo "semantics  : planner/GCDF/VBC/confidence unchanged"
echo "================================================================"

set +e
(
  CASE_FILE="${CASE_FILE}"   CASE_ID="${CASE_ID}"   RUN_ID="${RUN_ID}"   RUN_SECONDS="${RUN_SECONDS}"   WORLD_FILE="${WORLD_FILE}"   CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE}"   TOF_FUSION_ENABLED=true   EXECUTION_GCDF_AUDIT_ENABLED=true   GCDF_BODY_INFLATION_M=0.015   FORCE_ZERO_INITIAL_Q=true   APPLY_INITIAL_JOINT_OVERRIDES=auto   ENABLE_ORACLE_DIAGNOSTICS=true   EARLY_STOP_ON_GOAL=false   GAZEBO_GUI="${GAZEBO_GUI}"   USE_RVIZ="${USE_RVIZ}"   bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
) > >(tee "${LOG_FILE}") 2>&1
RUN_RC=$?
set -e

RUN_DIR="${RUN_ROOT}/run"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] run directory missing: ${RUN_DIR}"
  exit 3
fi

# Copy compact artifacts.
for f in   visibility_acquisition_summary.csv   blocker_stack_summary.csv   candidate_vbc_summary.csv   execution_vbc_summary.csv   execution_gcdf_hard_hold.csv   regime_summary.csv   joint_states.csv   tof_fusion_summary.csv   e3_summary.csv   goal_stop_status.json   startup_gazebo_tf_snapshot.json; do
  [[ -f "${RUN_DIR}/${f}" ]] && cp -f "${RUN_DIR}/${f}" "${DIAG_ROOT}/artifacts/${f}"
done

mkdir -p "${DIAG_ROOT}/projector_traces"
while IFS= read -r -d '' p; do
  cp -f "${p}" "${DIAG_ROOT}/projector_traces/$(basename "${p}")"
done < <(find "${RUN_ROOT}" -type f -name 'c46_obligation_*.json' -print0 2>/dev/null)

python3 - "${RUN_DIR}" "${DIAG_ROOT}/projector_traces" "${SUMMARY_JSON}" "${RUN_RC}" <<'PY'
import csv, glob, json, math, os, re, sys

run_dir, trace_dir, out_json, run_rc = sys.argv[1:]
run_rc = int(run_rc)
TOKEN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

def fnum(v, default=math.nan):
    try: return float(v)
    except Exception: return default

def inum(v, default=-1):
    try: return int(float(v))
    except Exception: return default

# Locate obligation-7 trace and verify override.
traces = []
for p in sorted(glob.glob(os.path.join(trace_dir, "c46_obligation_007_*.json"))):
    try:
        d = json.load(open(p))
        traces.append((p,d))
    except Exception:
        pass

override_applied = False
override_trace = None
for p,d in traces:
    if bool(d.get("c4_6_diagnostic_qvis_override_applied", False)):
        override_applied = True
        override_trace = {"path":p, "data":d}
        break

# Parse actual visibility acquisition rows.
acq_path = os.path.join(run_dir, "visibility_acquisition_summary.csv")
rows = []
if os.path.isfile(acq_path):
    with open(acq_path, newline="", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        ti = header.index("%time") if "%time" in header else 0
        di = header.index("field.data") if "field.data" in header else 1
        for row in rd:
            if len(row) <= di: continue
            d = dict(TOKEN.findall(",".join(row[di:])))
            if inum(d.get("active_obligation_id")) != 7:
                continue
            if d.get("active_query_status") != "ok":
                continue
            d["_time_s"] = fnum(row[ti]) / 1e9
            rows.append(d)

qdist = [fnum(r.get("active_q_distance_inf")) for r in rows]
qdist = [x for x in qdist if math.isfinite(x)]
vis = [fnum(r.get("active_max_current_visibility")) for r in rows]
vis = [x for x in vis if math.isfinite(x)]
conf = [fnum(r.get("active_min_confidence")) for r in rows]
conf = [x for x in conf if math.isfinite(x)]
seen_frac = [fnum(r.get("active_seen_fraction")) for r in rows]
seen_frac = [x for x in seen_frac if math.isfinite(x)]
seen_rows = [r for r in rows if inum(r.get("active_seen"),0) == 1]
near_rows = [
    r for r in rows
    if fnum(r.get("active_q_distance_inf")) <= 0.03
]
near_and_visible = [
    r for r in near_rows
    if fnum(r.get("active_max_current_visibility")) > 0.0
       or inum(r.get("active_seen"),0) == 1
]

# HARD_HOLD count.
hard_hold_path = os.path.join(run_dir, "execution_gcdf_hard_hold.csv")
hard_hold_true = 0
hard_hold_rows = 0
if os.path.isfile(hard_hold_path):
    with open(hard_hold_path, newline="", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        for row in rd:
            hard_hold_rows += 1
            txt = ",".join(row)
            if "hard_hold=1" in txt or "hard_hold=true" in txt.lower():
                hard_hold_true += 1

if not override_applied:
    verdict = "OVERRIDE_NOT_APPLIED"
elif seen_rows:
    verdict = "S7_ACTUAL_VISIBILITY_SUCCESS"
elif near_and_visible:
    verdict = "S7_RUNTIME_VISIBILITY_POSITIVE_NOT_SEEN_THRESHOLD"
elif near_rows:
    verdict = "S7_REACHED_BUT_RUNTIME_VISIBILITY_ZERO"
elif qdist:
    verdict = "S7_NOT_REACHED_WITHIN_0P03"
else:
    verdict = "NO_VALID_ACQUISITION_ROWS"

summary = {
    "diagnostic": "phase_e_case026_hardcoded_s7_ab_B",
    "run_return_code": run_rc,
    "override_applied": override_applied,
    "override_trace": override_trace,
    "valid_obligation7_query_rows": len(rows),
    "min_active_q_distance_inf": min(qdist) if qdist else None,
    "final_active_q_distance_inf": qdist[-1] if qdist else None,
    "max_current_visibility": max(vis) if vis else None,
    "max_min_confidence": max(conf) if conf else None,
    "max_seen_fraction": max(seen_frac) if seen_frac else None,
    "seen_row_count": len(seen_rows),
    "first_seen_time_s": seen_rows[0]["_time_s"] if seen_rows else None,
    "near_qvis_row_count_qdist_le_0p03": len(near_rows),
    "near_qvis_visible_row_count": len(near_and_visible),
    "execution_gcdf_hard_hold_rows": hard_hold_rows,
    "execution_gcdf_hard_hold_true": hard_hold_true,
    "verdict": verdict,
}
os.makedirs(os.path.dirname(out_json), exist_ok=True)
json.dump(summary, open(out_json,"w"), indent=2, allow_nan=True)

print("")
print("================ CASE026 S7 OVERRIDE SUMMARY ================")
print("override_applied             :", int(override_applied))
print("valid obligation7 queries    :", len(rows))
print("min q_dist_inf               :", summary["min_active_q_distance_inf"])
print("final q_dist_inf             :", summary["final_active_q_distance_inf"])
print("max current_visibility       :", summary["max_current_visibility"])
print("max min_confidence           :", summary["max_min_confidence"])
print("max seen_fraction            :", summary["max_seen_fraction"])
print("seen rows                    :", len(seen_rows))
print("near q_vis rows (<=0.03 rad) :", len(near_rows))
print("near q_vis + visible rows    :", len(near_and_visible))
print("execution GCDF HARD_HOLD     :", hard_hold_true)
print("VERDICT                      :", verdict)
print("[OUTPUT]", out_json)
print("=============================================================")
PY

cat > "${DIAG_ROOT}/metadata.txt" <<EOF
git_head=$(git rev-parse HEAD)
case_id=${CASE_ID}
case_file=${CASE_FILE}
world_file=${WORLD_FILE}
run_seconds=${RUN_SECONDS}
diagnostic=case026_hardcoded_s7_qvis_ab_B
override_enabled=true
override_obligation_id=7
override_target=${TARGET}
override_q=${S7_Q}
override_candidate_source=per_sensor_q0_feasibility_S7_robust
candidate_nominal_margin=0.08878699037077029
candidate_conservative_g=0.037837289648161056
candidate_primitive_self_occluded=false
runner_return_code=${RUN_RC}
EOF

rm -f "${UPLOAD_ZIP}"
python3 - "${DIAG_ROOT}" "${UPLOAD_ZIP}" <<'PY'
import os,sys,zipfile
root,dst=sys.argv[1:]
with zipfile.ZipFile(dst,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for base,_,files in os.walk(root):
        for name in files:
            p=os.path.join(base,name)
            z.write(p,os.path.relpath(p,root))
print(dst)
PY

echo ""
echo "================ CASE026 S7 TEST COMPLETE ==================="
echo "[SUMMARY] ${SUMMARY_JSON}"
echo "[UPLOAD]  ${UPLOAD_ZIP}"
ls -lh "${UPLOAD_ZIP}"
echo "============================================================="
