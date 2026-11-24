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
OLD8="$REPO/src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt"
SCALAR="$REPO/src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt"
OUT="$GRID25_ROOT/derived/grid25_v3_runtime_integration_v1/case026"
for f in "$CHECK" "$BEST" "$OLD8" "$SCALAR"; do
  [[ -f "$f" ]] || { echo "[STOP] Required checkpoint/verification missing: $f" >&2; exit 2; }
done
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
echo "[CASE026 BASELINE] Original scalar + old8 sensor model"
"$VIS_PYTHON" -u "$SCRIPT" "${COMMON[@]}" \
  --per-sensor-checkpoint "$OLD8" --output "$OUT/old8.json" 2>&1 | tee "$OUT/old8.log"
echo "[CASE026 FULL V3] SAME scalar + full-grid V3 H9 sensor model"
"$VIS_PYTHON" -u "$SCRIPT" "${COMMON[@]}" \
  --per-sensor-checkpoint "$BEST" --output "$OUT/grid25_v3_h9.json" 2>&1 | tee "$OUT/grid25_v3_h9.log"
"$VIS_PYTHON" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
for name in ("old8","grid25_v3_h9"):
    d=json.loads((p/f"{name}.json").read_text())
    v=d["per_sensor"]
    print("[CASE026]",name,"verdict=",d["verdict"],
          "selected_sensor=",v["selected_sensor_id"],
          "strict_s4_fallback=",v["strict_original_mode_fallback"],
          "branch_ms=",v["branch_compute_ms"],
          "num_attempts=",len(v["attempts"]))
print("[CASE026 DIAGNOSTIC ONLY] No GCDF/VBC, no trajectory certification, no robot control")
PY
