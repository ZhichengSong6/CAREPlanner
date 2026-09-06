#!/usr/bin/env bash
set -euo pipefail

# Package the minimum historical Phase-E qualification artifacts needed for a
# strict per-case blocker analysis. This is OFFLINE ONLY: it does not launch ROS,
# Gazebo, the planner, or any learned model.
#
# Default target is the original 30-case empty-world qualification at ed3b873f.
#
# Usage:
#   bash scripts/pack_phase_e_historical_blocker_inputs.sh
#
# Optional:
#   BATCH_ROOT=/path/to/phase_e_empty_qualification_xxx \
#     bash scripts/pack_phase_e_historical_blocker_inputs.sh

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
BATCH_ROOT="${BATCH_ROOT:-${REPO}/outputs/phase_e_empty_world_qualification/phase_e_empty_qualification_20260904-183310_ed3b873f}"

cd "${REPO}"

if [[ ! -d "${BATCH_ROOT}" ]]; then
  echo "[ERROR] batch root not found:"
  echo "        ${BATCH_ROOT}"
  exit 2
fi

if [[ ! -d "${BATCH_ROOT}/case_artifacts" ]]; then
  echo "[ERROR] case_artifacts missing under:"
  echo "        ${BATCH_ROOT}"
  exit 3
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BATCH_NAME="$(basename "${BATCH_ROOT}")"
PACK_ROOT="${REPO}/outputs/phase_e_historical_blocker_pack/${BATCH_NAME}_${STAMP}"
ZIP_PATH="${REPO}/CAREPlanner_PHASE_E_HISTORICAL_BLOCKER_INPUTS_${BATCH_NAME}_${STAMP}.zip"

rm -rf "${PACK_ROOT}"
rm -f "${ZIP_PATH}"

mkdir -p "${PACK_ROOT}/case_artifacts"          "${PACK_ROOT}/case_summaries"          "${PACK_ROOT}/diagnostic"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -f "${src}" "${dst}"
  fi
}

# Batch-level metadata/summaries.
for name in   qualification_metadata.txt   qualification_summary.csv   qualification_summary.json; do
  copy_if_exists "${BATCH_ROOT}/${name}" "${PACK_ROOT}/${name}"
done

# Per-case files. Keep the raw VBC streams plus enough state-machine context to
# reconstruct first/dominant/last blocker and verify the historical semantics.
mapfile -t CASE_DIRS < <(
  find "${BATCH_ROOT}/case_artifacts" -mindepth 1 -maxdepth 1 -type d | sort
)

if [[ "${#CASE_DIRS[@]}" -eq 0 ]]; then
  echo "[ERROR] no case directories found"
  exit 4
fi

FILES=(
  candidate_vbc_summary.csv
  execution_vbc_summary.csv
  blocker_stack_summary.csv
  visibility_acquisition_summary.csv
  regime_summary.csv
  commit_summary.csv
  verification_outcome.csv
  waypoint_schedule_summary.csv
  local_planner_summary.csv
  nominal_progress_summary.csv
  goal_stop_status.json
  c4_4_verified_regime_summary.json
)

for case_dir in "${CASE_DIRS[@]}"; do
  cid="$(basename "${case_dir}")"
  dst="${PACK_ROOT}/case_artifacts/${cid}"
  mkdir -p "${dst}"

  for name in "${FILES[@]}"; do
    copy_if_exists "${case_dir}/${name}" "${dst}/${name}"
  done

  copy_if_exists     "${BATCH_ROOT}/case_summaries/${cid}.json"     "${PACK_ROOT}/case_summaries/${cid}.json"
done

# Produce a small machine-readable manifest so the receiving side can detect
# historical CSV schema mismatches immediately without guessing.
python3 - "${BATCH_ROOT}" "${PACK_ROOT}/diagnostic/schema_manifest.json" <<'PY'
import csv
import json
import os
import re
import sys
from pathlib import Path

batch = Path(sys.argv[1])
out = Path(sys.argv[2])

TOKEN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

report = {
    "batch_root": str(batch),
    "case_count": 0,
    "cases": {},
}

case_root = batch / "case_artifacts"
for case_dir in sorted(p for p in case_root.iterdir() if p.is_dir()):
    cid = case_dir.name
    report["case_count"] += 1
    entry = {}
    for name in (
        "candidate_vbc_summary.csv",
        "blocker_stack_summary.csv",
        "regime_summary.csv",
        "visibility_acquisition_summary.csv",
    ):
        path = case_dir / name
        info = {
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "header": [],
            "row_count": 0,
            "first_nonempty_field_data": None,
            "first_nonempty_tokens": {},
            "trajectory_source_values": [],
            "has_violation_values": [],
        }
        if path.is_file():
            sources = set()
            violations = set()
            try:
                with path.open(newline="", errors="replace") as f:
                    rd = csv.reader(f)
                    header = next(rd, [])
                    info["header"] = header
                    di = header.index("field.data") if "field.data" in header else (
                        1 if len(header) > 1 else 0
                    )
                    for row in rd:
                        info["row_count"] += 1
                        if len(row) <= di:
                            continue
                        text = ",".join(row[di:])
                        d = dict(TOKEN.findall(text))
                        if info["first_nonempty_field_data"] is None and text.strip():
                            info["first_nonempty_field_data"] = text[:3000]
                            info["first_nonempty_tokens"] = d
                        if "trajectory_source" in d:
                            sources.add(d["trajectory_source"])
                        if "has_violation" in d:
                            violations.add(d["has_violation"])
                info["trajectory_source_values"] = sorted(sources)
                info["has_violation_values"] = sorted(violations)
            except Exception as exc:
                info["parse_error"] = repr(exc)
        entry[name] = info
    report["cases"][cid] = entry

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print("[MANIFEST]", out)
PY

# Include a concise human-readable inventory.
{
  echo "batch_root=${BATCH_ROOT}"
  echo "batch_name=${BATCH_NAME}"
  echo "pack_time=${STAMP}"
  echo "repo_head=$(git rev-parse HEAD)"
  echo "case_count=${#CASE_DIRS[@]}"
  echo
  echo "candidate_vbc_summary counts:"
  find "${PACK_ROOT}/case_artifacts" -type f -name candidate_vbc_summary.csv | sort
  echo
  echo "file sizes:"
  find "${PACK_ROOT}" -type f -printf '%s %p\n' | sort -n
} > "${PACK_ROOT}/diagnostic/inventory.txt"

# Zip without depending on external zip options/features.
python3 - "${PACK_ROOT}" "${ZIP_PATH}" <<'PY'
import os
import sys
import zipfile

root, dst = sys.argv[1:]
with zipfile.ZipFile(
    dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
) as zf:
    for base, _, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(base, name)
            arc = os.path.relpath(path, root)
            zf.write(path, arc)
print(dst)
PY

echo
echo "================================================================"
echo "PHASE-E HISTORICAL BLOCKER INPUT PACK COMPLETE"
echo "================================================================"
echo "[BATCH]       ${BATCH_ROOT}"
echo "[CASES]       ${#CASE_DIRS[@]}"
echo "[MANIFEST]    ${PACK_ROOT}/diagnostic/schema_manifest.json"
echo "[UPLOAD THIS] ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
