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

FIXED_TARGET="0.1,0.05,0.15"
FIXED_S7_Q="-0.8144537806510925,0.5605988502502441,1.0171856880187988,-2.180335760116577,0.12358028441667557,-1.7687498331069946,1.1331826448440552"
FIXED_LABEL="case026_fixed_target_robust_s7_nominal00888_cons00378_los_clear"

cd "${REPO}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "[ERROR] case file missing: ${CASE_FILE}"
  exit 2
fi

# ROS/Gazebo must stay outside research conda environments.
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

python3 -m py_compile   src/care_visibility_cdf/scripts/vbc_fixed_visibility_diagnostic_impl.py   src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py   src/care_visibility_cdf/scripts/vbc_multi_deadline_obligation_impl.py

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SHORT="$(git rev-parse --short=8 HEAD)"
RUN_ID="${CASE_ID}_fixed_target_s7_${STAMP}_${SHORT}"
RUN_ROOT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
RUN_DIR="${RUN_ROOT}/run"
DIAG_ROOT="${REPO}/outputs/phase_e_case026_fixed_target_s7/${RUN_ID}"
SUMMARY_JSON="${DIAG_ROOT}/fixed_target_s7_summary.json"
RUNNER_LOG="${DIAG_ROOT}/runner.log"
UPLOAD_ZIP="${REPO}/CAREPlanner_PHASE_E_CASE026_FIXED_TARGET_S7_${RUN_ID}.zip"

rm -rf "${DIAG_ROOT}"
mkdir -p "${DIAG_ROOT}/artifacts"

# Disable the previous match-based diagnostic so there is exactly one mechanism.
unset CARE_DIAG_QVIS_OVERRIDE_ENABLED || true
unset CARE_DIAG_QVIS_OVERRIDE_OBLIGATION_ID || true
unset CARE_DIAG_QVIS_OVERRIDE_TARGET || true
unset CARE_DIAG_QVIS_OVERRIDE_Q || true

export CARE_FIXED_VIS_TARGET="${FIXED_TARGET}"
export CARE_FIXED_VIS_Q="${FIXED_S7_Q}"
export CARE_FIXED_VIS_Q_ZERO="${FIXED_S7_Q}"
export CARE_FIXED_VIS_OBLIGATION_ID=1
export CARE_FIXED_VIS_LABEL="${FIXED_LABEL}"

echo "================================================================"
echo "CASE 026 FIXED TARGET / FIXED S7 VISIBILITY TEST"
echo "case       : ${CASE_ID}"
echo "world      : ${WORLD_FILE}"
echo "runtime    : ${RUN_SECONDS}s"
echo "mode       : fixed_visibility_diagnostic"
echo "target     : [${FIXED_TARGET}]"
echo "q_vis(S7)  : [${FIXED_S7_Q}]"
echo "natural O  : IGNORED by fixed diagnostic node"
echo "unchanged  : Sparse-SCP / final GCDF / exact VBC / tracker / ToF"
echo "================================================================"

set +e
(
  CASE_FILE="${CASE_FILE}"   CASE_ID="${CASE_ID}"   RUN_ID="${RUN_ID}"   RUN_SECONDS="${RUN_SECONDS}"   WORLD_FILE="${WORLD_FILE}"   CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE}"   REGION_SCHEDULE_MODE="fixed_visibility_diagnostic"   PROGRESSIVE_SHARED_REPAIR_ENABLED=false   FRONTIER_STEERING_ENABLED=false   VBC_GATED_FRONTIER_STEP_ENABLED=false   ADAPTIVE_REFINEMENT_ENABLED=false   TOF_FUSION_ENABLED=true   EXECUTION_GCDF_AUDIT_ENABLED=true   GCDF_BODY_INFLATION_M=0.015   FORCE_ZERO_INITIAL_Q=true   APPLY_INITIAL_JOINT_OVERRIDES=auto   ENABLE_ORACLE_DIAGNOSTICS=true   EARLY_STOP_ON_GOAL=false   GAZEBO_GUI="${GAZEBO_GUI}"   USE_RVIZ="${USE_RVIZ}"   bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
) > >(tee "${RUNNER_LOG}") 2>&1
RUN_RC=$?
set -e

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[ERROR] expected run directory missing: ${RUN_DIR}"
  exit 3
fi

for f in   visibility_acquisition_summary.csv   regime_summary.csv   local_planner_summary.csv   candidate_vbc_summary.csv   execution_vbc_summary.csv   execution_gcdf_hard_hold.csv   tracker_summary.csv   joint_states.csv   tof_fusion_summary.csv   goal_stop_status.json   runtime_semantics.txt   startup_gazebo_tf_snapshot.json; do
  [[ -f "${RUN_DIR}/${f}" ]] && cp -f "${RUN_DIR}/${f}" "${DIAG_ROOT}/artifacts/${f}"
done

mkdir -p "${DIAG_ROOT}/fixed_traces"
while IFS= read -r -d '' p; do
  cp -f "${p}" "${DIAG_ROOT}/fixed_traces/$(basename "${p}")"
done < <(find "${RUN_ROOT}" -type f -name 'fixed_visibility_obligation_*.json' -print0 2>/dev/null)

# Preserve the waypoint generator log because it is the authoritative evidence
# that the fixed obligation was injected and natural active sets were ignored.
GEN_LOG="${REPO}/logs/c5_5_vbc_gcdf_regime/${RUN_ID}/run/waypoint_generator.log"
[[ -f "${GEN_LOG}" ]] && cp -f "${GEN_LOG}" "${DIAG_ROOT}/artifacts/waypoint_generator.log"

python3 - "${RUN_DIR}" "${DIAG_ROOT}/fixed_traces" "${SUMMARY_JSON}" "${RUN_RC}" <<'PY'
import csv, glob, json, math, os, re, sys

run_dir, trace_dir, out_json, run_rc = sys.argv[1:]
run_rc = int(run_rc)
TOK = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

def fnum(x, default=math.nan):
    try: return float(x)
    except Exception: return default

def inum(x, default=-1):
    try: return int(float(x))
    except Exception: return default

traces = sorted(glob.glob(os.path.join(trace_dir, "fixed_visibility_obligation_*.json")))
injected = len(traces) > 0
trace = None
if injected:
    try: trace = json.load(open(traces[-1]))
    except Exception: trace = None

rows = []
acq = os.path.join(run_dir, "visibility_acquisition_summary.csv")
if os.path.isfile(acq):
    with open(acq, newline="", errors="replace") as f:
        rd = csv.reader(f)
        h = next(rd, [])
        ti = h.index("%time") if "%time" in h else 0
        di = h.index("field.data") if "field.data" in h else 1
        for row in rd:
            if len(row) <= di:
                continue
            d = dict(TOK.findall(",".join(row[di:])))
            # While the fixed obligation exists it is always active id=1.
            if inum(d.get("active_obligation_id")) != 1:
                continue
            if d.get("active_query_status") != "ok":
                continue
            d["_t"] = fnum(row[ti]) / 1e9
            rows.append(d)

qdist = [fnum(r.get("active_q_distance_inf")) for r in rows]
qdist = [x for x in qdist if math.isfinite(x)]
vis = [fnum(r.get("active_max_current_visibility")) for r in rows]
vis = [x for x in vis if math.isfinite(x)]
conf = [fnum(r.get("active_min_confidence")) for r in rows]
conf = [x for x in conf if math.isfinite(x)]
frac = [fnum(r.get("active_seen_fraction")) for r in rows]
frac = [x for x in frac if math.isfinite(x)]

near = [r for r in rows if fnum(r.get("active_q_distance_inf")) <= 0.03]
near_vis = [
    r for r in near
    if fnum(r.get("active_max_current_visibility")) > 0.0
]
seen = [r for r in rows if inum(r.get("active_seen"), 0) == 1]

# Acquisition completion may remove the obligation before the next summary row,
# so also inspect regime/acquisition text for the explicit removal event.
seen_event = False
for p in [
    os.path.join(run_dir, "visibility_acquisition_summary.csv"),
]:
    if os.path.isfile(p):
        txt = open(p, errors="replace").read()
        if "seen_obligation_count=1" in txt or "seen=1" in txt:
            seen_event = True

# Fixed mode injection evidence from generator log.
gen_candidates = [
    os.path.join(os.path.dirname(os.path.dirname(run_dir)), "logs", "dummy")
]
# Wrapper copied the log outside run_dir, so infer only from trace here.

# GCDF execution hard hold.
hard_true = 0
hard_path = os.path.join(run_dir, "execution_gcdf_hard_hold.csv")
if os.path.isfile(hard_path):
    with open(hard_path, errors="replace") as f:
        for line in f:
            low = line.lower()
            if "hard_hold=1" in low or "hard_hold=true" in low:
                hard_true += 1

# Exact execution VBC unsafe.
execution_vbc_unsafe = 0
exe_path = os.path.join(run_dir, "execution_vbc_summary.csv")
if os.path.isfile(exe_path):
    with open(exe_path, errors="replace") as f:
        for line in f:
            if "has_violation=1" in line:
                execution_vbc_unsafe += 1

if not injected:
    verdict = "FIXED_OBLIGATION_NOT_INJECTED"
elif seen or seen_event:
    verdict = "FIXED_S7_ACTUAL_VISIBILITY_SUCCESS"
elif near_vis:
    verdict = "FIXED_S7_RUNTIME_VISIBILITY_POSITIVE_NOT_SEEN"
elif near:
    verdict = "FIXED_S7_REACHED_BUT_RUNTIME_VISIBILITY_ZERO"
elif qdist:
    verdict = "FIXED_S7_NOT_REACHED_WITHIN_0P03"
else:
    verdict = "NO_VALID_FIXED_TARGET_QUERY_ROWS"

summary = {
    "diagnostic": "phase_e_case026_fixed_target_fixed_s7",
    "run_return_code": run_rc,
    "fixed_obligation_injected": injected,
    "fixed_trace": trace,
    "valid_fixed_target_query_rows": len(rows),
    "min_q_distance_inf": min(qdist) if qdist else None,
    "final_q_distance_inf": qdist[-1] if qdist else None,
    "max_current_visibility": max(vis) if vis else None,
    "max_min_confidence": max(conf) if conf else None,
    "max_seen_fraction": max(frac) if frac else None,
    "near_qvis_rows_le_0p03": len(near),
    "near_qvis_visible_rows": len(near_vis),
    "seen_rows": len(seen),
    "seen_event_detected": bool(seen_event),
    "execution_gcdf_hard_hold_true": hard_true,
    "execution_vbc_unsafe": execution_vbc_unsafe,
    "verdict": verdict,
}
os.makedirs(os.path.dirname(out_json), exist_ok=True)
json.dump(summary, open(out_json, "w"), indent=2, allow_nan=True)

print("")
print("================ FIXED TARGET S7 SUMMARY =================")
print("fixed obligation injected    :", int(injected))
print("valid target query rows      :", len(rows))
print("min q_dist_inf               :", summary["min_q_distance_inf"])
print("final q_dist_inf             :", summary["final_q_distance_inf"])
print("max current_visibility       :", summary["max_current_visibility"])
print("max min_confidence           :", summary["max_min_confidence"])
print("max seen_fraction            :", summary["max_seen_fraction"])
print("near q_vis rows <=0.03       :", len(near))
print("near q_vis visible rows      :", len(near_vis))
print("seen rows / event            :", len(seen), int(seen_event))
print("execution GCDF HARD_HOLD     :", hard_true)
print("execution VBC unsafe         :", execution_vbc_unsafe)
print("VERDICT                      :", verdict)
print("[OUTPUT]", out_json)
print("==========================================================")
PY

cat > "${DIAG_ROOT}/metadata.txt" <<EOF
git_head=$(git rev-parse HEAD)
case_id=${CASE_ID}
run_seconds=${RUN_SECONDS}
world_file=${WORLD_FILE}
region_schedule_mode=fixed_visibility_diagnostic
fixed_target=${FIXED_TARGET}
fixed_q_vis=${FIXED_S7_Q}
fixed_label=${FIXED_LABEL}
natural_active_sets=ignored
candidate_nominal_margin=0.08878699037077029
candidate_conservative_g=0.037837289648161056
candidate_primitive_self_occluded=false
planner=unchanged_local_sparse_scp
final_gcdf=enabled
exact_vbc=enabled
tracker=unchanged
tof_confidence=real_runtime_pipeline
runner_return_code=${RUN_RC}
EOF

rm -f "${UPLOAD_ZIP}"
python3 - "${DIAG_ROOT}" "${UPLOAD_ZIP}" <<'PY'
import os, sys, zipfile
root, dst = sys.argv[1:]
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for base, dirs, files in os.walk(root):
        dirs.sort(); files.sort()
        for name in files:
            p = os.path.join(base, name)
            z.write(p, os.path.relpath(p, root))
print(dst)
PY

echo ""
echo "================ FIXED TARGET TEST COMPLETE ==============="
echo "[SUMMARY] ${SUMMARY_JSON}"
echo "[UPLOAD]  ${UPLOAD_ZIP}"
ls -lh "${UPLOAD_ZIP}"
echo "==========================================================="

exit "${RUN_RC}"
