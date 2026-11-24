#!/usr/bin/env bash
# Diagnostic-only replay. No online planner, ROS master, Gazebo or robot motion.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
cd "$REPO"
: "${VIS_PYTHON:?Set VIS_PYTHON to your viscdf Python}"
: "${GRID25_ROOT:?Set GRID25_ROOT to your existing frozen Grid25 data root}"
CHECK="$GRID25_ROOT/derived/grid25_v3_runtime_integration_v1/runtime_check_cpu.json"
BEST="$GRID25_ROOT/derived/grid25_v3_full_training_v1/run_r1_v3_full/best.pt"
# A historical old8 checkpoint is OPTIONAL: some HPC source checkouts
# intentionally do not contain non-Git model artifacts. Without old8 we can
# still run the same new H9 candidate search, but NO comparative claim.
OLD8="${OLD8_CHECKPOINT:-$REPO/src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt}"
SCALAR="${SCALAR_CHECKPOINT:-$REPO/src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt}"
OUT="$GRID25_ROOT/derived/grid25_v3_runtime_integration_v1/case026"
for f in "$CHECK" "$BEST" "$SCALAR"; do
  [[ -f "$f" ]] || { echo "[STOP] Required checkpoint/verification missing: $f" >&2; exit 2; }
done
BASELINE_AVAILABLE=false
if [[ -f "$OLD8" ]]; then
  BASELINE_AVAILABLE=true
else
  echo "[CASE026 BASELINE UNAVAILABLE] Old8 checkpoint not found: $OLD8"
  echo "[CASE026 H9-ONLY] Proceed with new H9 candidate search; comparison NOT performed"
fi
[[ ! -e "$OUT" ]] || { echo "[STOP] Case026 outputs already exist: $OUT" >&2; exit 2; }
# Reverify that this report belongs to the same frozen fullscale model.
"$VIS_PYTHON" - "$CHECK" "$BEST" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
h=hashlib.sha256()
with open(sys.argv[2],"rb") as f:
    for block in iter(lambda:f.read(8<<20),b""):h.update(block)
assert r.get("status")=="PASS" and r.get("checkpoint_sha256")==h.hexdigest(), "Runtime proof stale or different checkpoint"
assert r.get("device")=="cpu", "Expected completed CPU runtime proof"
print("[CASE026 MODEL IDENTITY VERIFIED]",r["best_step"])
PY
mkdir -p "$OUT"
SCRIPT="$REPO/scripts/test_phase_e_case026_targeted_per_sensor_fallback.py"
COMMON=(--device cpu
        --scalar-checkpoint "$SCALAR"
        --reference-urdf "$REPO/src/arm_description/urdf/Arm.urdf"
        --self-filter-urdf "$REPO/src/arm_description/urdf/Arm_with_self_filter_collision.urdf"
        --projection-iters 10
        --projection-damping 0.5
        --projection-epsilon-f 0.03
        --projection-max-step-norm 0.25
        --root-refine-iters 12
        --root-tolerance-f 0.002
        --branch-ascent-steps 1
        --branch-step-size 0.05
        --branch-max-step-norm 0.25
        --max-branch-attempts 8
        --force-first-sensor 4)
if [[ "$BASELINE_AVAILABLE" == true ]]; then
  echo "[CASE026 BASELINE] Original scalar + old8 sensor model"
  "$VIS_PYTHON" -u "$SCRIPT" "${COMMON[@]}" \
    --per-sensor-checkpoint "$OLD8" --output "$OUT/old8.json" 2>&1 | tee "$OUT/old8.log"
fi
echo "[CASE026 FULL V3] Original scalar + full-grid V3 H9 sensor model"
"$VIS_PYTHON" -u "$SCRIPT" "${COMMON[@]}" \
  --per-sensor-checkpoint "$BEST" --output "$OUT/grid25_v3_h9.json" 2>&1 | tee "$OUT/grid25_v3_h9.log"
"$VIS_PYTHON" - "$OUT" "$BASELINE_AVAILABLE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
baseline_available=sys.argv[2]=="true"
modes=("old8","grid25_v3_h9") if baseline_available else ("grid25_v3_h9",)
summary=dict(
    diagnostic="Case026 sensor candidate search",
    status="COMPLETE",
    baseline_available=baseline_available,
    compared_models=list(modes),
    results={},
    diagnostic_only=True,
    no_los_or_trajectory_safety_claim=True,
)
for name in modes:
    d=json.loads((p/f"{name}.json").read_text())
    v=d["per_sensor"]
    summary["results"][name]=dict(
        verdict=d["verdict"],
        selected_sensor_id=v["selected_sensor_id"],
        strict_s4_fallback=v["strict_original_mode_fallback"],
        branch_compute_ms=v["branch_compute_ms"],
        num_attempts=len(v["attempts"]),
    )
    print("[CASE026]",name,"verdict=",d["verdict"],
          "selected_sensor=",v["selected_sensor_id"],
          "strict_s4_fallback=",v["strict_original_mode_fallback"],
          "branch_ms=",v["branch_compute_ms"],
          "num_attempts=",len(v["attempts"]))
if not baseline_available:
    print("[CASE026 NO BASELINE COMPARISON] Only new H9 evaluated; do not claim improvement over old8")
(p/"comparison_status.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
print("[CASE026 DIAGNOSTIC ONLY] No GCDF/VBC, no trajectory certification, no robot control")
PY
