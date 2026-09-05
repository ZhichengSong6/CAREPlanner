#!/usr/bin/env bash
set -euo pipefail

# One-shot Phase-E blocker A/B/C diagnostic.
#
# A: raw body-sample sphere vs +swept-margin shell provenance.
# B: real 8-sensor Pinocchio FOV oracle at measured seed, learned q_vis, and a
#    deterministic local q search.
# C: repeat on two different tasks and compare first/top blocker identity.
#
# This wrapper is diagnostic-only. Planner/safety decisions are unchanged.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
RUN_SECONDS="${RUN_SECONDS:-20}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
NCDF_ENV="${NCDF_ENV:-ncdf_l4c}"
RUN_ORACLE_DIAGNOSTICS="${RUN_ORACLE_DIAGNOSTICS:-false}"

OLD_CASE_FILE="${OLD_CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
OLD_CASE_ID="${OLD_CASE_ID:-case_014}"
NEW_CASE_FILE="${NEW_CASE_FILE:-${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
NEW_CASE_ID="${NEW_CASE_ID:-phase_e_goal_014}"

cd "${REPO}"
source devel/setup.bash

if [[ ! -f "${OLD_CASE_FILE}" || ! -f "${NEW_CASE_FILE}" ]]; then
  echo "[ERROR] case file missing"
  echo "  old: ${OLD_CASE_FILE}"
  echo "  new: ${NEW_CASE_FILE}"
  exit 2
fi

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found"
  exit 3
fi

BASESTAMP="${BASESTAMP:-$(date +%Y%m%d-%H%M%S)}"
SHORT="$(git rev-parse --short=8 HEAD)"
ROOT="${REPO}/outputs/phase_e_blocker_abc/${BASESTAMP}_${SHORT}"
mkdir -p "${ROOT}"

echo "================================================================"
echo "PHASE-E BLOCKER A/B/C ONE-SHOT DIAGNOSTIC"
echo "git head     : $(git rev-parse HEAD)"
echo "run seconds  : ${RUN_SECONDS} per case"
echo "runtime oracle: ${RUN_ORACLE_DIAGNOSTICS} (offline A/B analyzer still runs oracle)"
echo "case A       : ${OLD_CASE_ID}"
echo "case B       : ${NEW_CASE_ID}"
echo "result root  : ${ROOT}"
echo "================================================================"

run_case() {
  local label="$1"
  local case_file="$2"
  local case_id="$3"
  local stamp="${BASESTAMP}_${label}"
  local diag_root="${REPO}/outputs/phase_e_obligation_geometry_diagnostic/${case_id}_obligation_geometry_diag_${stamp}_${SHORT}"
  local case_json="${ROOT}/${label}_blocker_abc.json"
  local console_log="${ROOT}/${label}_console.log"

  echo ""
  echo "==================== RUN ${label}: ${case_id} ===================="

  ENABLE_ORACLE_DIAGNOSTICS="${RUN_ORACLE_DIAGNOSTICS}" \
  CASE_FILE="${case_file}" \
  CASE_ID="${case_id}" \
  RUN_SECONDS="${RUN_SECONDS}" \
  EARLY_STOP_ON_GOAL=true \
  GAZEBO_GUI="${GAZEBO_GUI}" \
  USE_RVIZ="${USE_RVIZ}" \
  STAMP="${stamp}" \
  bash scripts/run_phase_e_obligation_geometry_diagnostic.sh \
    2>&1 | tee "${console_log}"

  if [[ ! -d "${diag_root}" ]]; then
    echo "[ERROR] expected diagnostic root missing: ${diag_root}"
    exit 4
  fi

  # Re-run only the offline analysis in the same environment used by the
  # learned/Pinocchio visibility code. No ROS actuation is performed here.
  bash -lc "source '${CONDA_SH}'; conda activate '${NCDF_ENV}'; cd '${REPO}'; source devel/setup.bash; python scripts/analyze_phase_e_blocker_abc.py --repo '${REPO}' --diag-root '${diag_root}' --case-id '${case_id}' --output '${case_json}'"

  cp -f "${case_json}" "${diag_root}/blocker_abc_case_report.json"
  echo "[CASE REPORT] ${case_json}"
}

run_case "old" "${OLD_CASE_FILE}" "${OLD_CASE_ID}"
run_case "new" "${NEW_CASE_FILE}" "${NEW_CASE_ID}"

python3 - "${ROOT}/old_blocker_abc.json" "${ROOT}/new_blocker_abc.json" "${ROOT}/combined_blocker_abc.json" <<'PY'
import json, math, sys

old_path, new_path, out_path = sys.argv[1:]
old = json.load(open(old_path))
new = json.load(open(new_path))

def ident(rec):
    if not rec:
        return None
    xyz = rec.get("point_xyz")
    if not isinstance(xyz, list) or len(xyz) != 3:
        return None
    return {
        "link": rec.get("link"),
        "sample_index": rec.get("sample_index"),
        "point_xyz": [round(float(v), 4) for v in xyz],
    }

def same(a, b):
    return a is not None and b is not None and a == b

old_first = ident(old.get("first_blocker"))
new_first = ident(new.get("first_blocker"))
old_top = ident((old.get("top_blockers") or [None])[0])
new_top = ident((new.get("top_blockers") or [None])[0])

old_b = old.get("B_oracle_visibility") or {}
new_b = new.get("B_oracle_visibility") or {}

report = {
    "test": "phase_e_blocker_abc_combined",
    "cases": [old.get("case_id"), new.get("case_id")],
    "C_task_independence": {
        "old_first_blocker": old_first,
        "new_first_blocker": new_first,
        "same_first_blocker": same(old_first, new_first),
        "old_top_blocker": old_top,
        "new_top_blocker": new_top,
        "same_top_blocker": same(old_top, new_top),
        "old_top_fraction_of_unsafe": old.get("top_blocker_fraction_of_unsafe"),
        "new_top_fraction_of_unsafe": new.get("top_blocker_fraction_of_unsafe"),
        "structural_blocker_evidence": bool(
            same(old_first, new_first) or same(old_top, new_top)
        ),
    },
    "A_old": old.get("A_raw_body_vs_margin"),
    "A_new": new.get("A_raw_body_vs_margin"),
    "B_old": {
        "status": old_b.get("status"),
        "visible_at_measured_seed": old_b.get("blocker_nominal_visible_at_measured_seed"),
        "visible_at_q_vis": old_b.get("blocker_nominal_visible_at_q_vis"),
        "local_search_any_visible": old_b.get("local_search_any_nominal_visible"),
        "local_search_best_g": old_b.get("local_search_best_nominal_g"),
        "local_search_best_source": old_b.get("local_search_best_source"),
    },
    "B_new": {
        "status": new_b.get("status"),
        "visible_at_measured_seed": new_b.get("blocker_nominal_visible_at_measured_seed"),
        "visible_at_q_vis": new_b.get("blocker_nominal_visible_at_q_vis"),
        "local_search_any_visible": new_b.get("local_search_any_nominal_visible"),
        "local_search_best_g": new_b.get("local_search_best_nominal_g"),
        "local_search_best_source": new_b.get("local_search_best_source"),
    },
}
with open(out_path, "w") as f:
    json.dump(report, f, indent=2, allow_nan=True)

print("\n================ COMBINED BLOCKER A/B/C ================")
print(json.dumps(report, indent=2, allow_nan=True))
print("=========================================================")
PY

cat > "${ROOT}/metadata.txt" <<EOF
git_head=$(git rev-parse HEAD)
old_case_file=${OLD_CASE_FILE}
old_case_id=${OLD_CASE_ID}
new_case_file=${NEW_CASE_FILE}
new_case_id=${NEW_CASE_ID}
run_seconds_per_case=${RUN_SECONDS}
enable_oracle_diagnostics=${RUN_ORACLE_DIAGNOSTICS}
moving_body_prior_refresh=false
EOF

UPLOAD_ZIP="${REPO}/CAREPlanner_PHASE_E_BLOCKER_ABC_${BASESTAMP}_${SHORT}.zip"
rm -f "${UPLOAD_ZIP}"

python3 - "${ROOT}" "${REPO}/outputs/phase_e_obligation_geometry_diagnostic" "${BASESTAMP}" "${SHORT}" "${OLD_CASE_ID}" "${NEW_CASE_ID}" "${UPLOAD_ZIP}" <<'PY'
import os, sys, zipfile

root, diag_base, stamp, short, old_id, new_id, dst = sys.argv[1:]
wanted = [
    os.path.join(diag_base, f"{old_id}_obligation_geometry_diag_{stamp}_old_{short}"),
    os.path.join(diag_base, f"{new_id}_obligation_geometry_diag_{stamp}_new_{short}"),
]

with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for base, _, files in os.walk(root):
        for name in files:
            p = os.path.join(base, name)
            z.write(p, os.path.join("combined", os.path.relpath(p, root)))
    for d in wanted:
        if not os.path.isdir(d):
            continue
        label = os.path.basename(d)
        for base, _, files in os.walk(d):
            for name in files:
                p = os.path.join(base, name)
                z.write(p, os.path.join("cases", label, os.path.relpath(p, d)))
print(dst)
PY

echo ""
echo "================================================================"
echo "[DONE] One-shot blocker A/B/C diagnostic finished"
echo "[COMBINED JSON] ${ROOT}/combined_blocker_abc.json"
echo "[UPLOAD THIS ONE ZIP] ${UPLOAD_ZIP}"
ls -lh "${UPLOAD_ZIP}"
echo "================================================================"
