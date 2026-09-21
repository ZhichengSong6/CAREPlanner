#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)}"
CASE_FILE="${CASE_FILE:-$REPO/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
CASE_ID="${CASE_ID:-phase_e_goal_026}"
EVAL_CASE_ID="${EVAL_CASE_ID:-phase_e_case_009}"
EVAL_CASES_JSON="${EVAL_CASES_JSON:-src/egocentric_arm_planner/config/phase_e_obstacle_core12_v1.json}"
V1_CHECKPOINT="${V1_CHECKPOINT:-$REPO/src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt}"
R1_CHECKPOINT="${R1_CHECKPOINT:-$REPO/src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt}"
RUN_SECONDS="${RUN_SECONDS:-60}"
RUN_TARGETED="${RUN_TARGETED:-true}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
ROOT="${ROOT:-$REPO/outputs/phase_e_r1_runtime_qualification/$STAMP}"

if [[ -z "${VIS_PYTHON:-}" ]]; then
  for p in "$HOME/miniforge3/envs/viscdf/bin/python" "$HOME/anaconda3/envs/viscdf/bin/python" "$HOME/miniconda3/envs/viscdf/bin/python"; do
    if [[ -x "$p" ]]; then VIS_PYTHON="$p"; break; fi
  done
fi
VIS_PYTHON="${VIS_PYTHON:-python3}"

mkdir -p "$ROOT"
cd "$REPO"

for p in "$CASE_FILE" "$V1_CHECKPOINT" "$R1_CHECKPOINT"; do
  [[ -f "$p" ]] || { echo "[ERROR] missing $p" >&2; exit 2; }
done

python3 - "$REPO/$EVAL_CASES_JSON" "$EVAL_CASE_ID" "$CASE_ID" <<'PY'
import json,sys
p,eid,source=sys.argv[1:]
db=json.load(open(p)); c=next((x for x in db['cases'] if x['case_id']==eid),None)
assert c is not None,(p,eid)
assert c.get('source_goal_id')==source,(c.get('source_goal_id'),source)
print('[preflight] evaluation case maps to source goal:',eid,'->',source)
PY

"$VIS_PYTHON" - "$V1_CHECKPOINT" "$R1_CHECKPOINT" <<'PY'
import hashlib,sys
expected=['979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199','4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002']
for p,e in zip(sys.argv[1:],expected):
    h=hashlib.sha256(open(p,'rb').read()).hexdigest()
    if h!=e: raise SystemExit(f'SHA mismatch {p}: {h} != {e}')
    print('[checkpoint-ok]',p,h)
PY

cleanup_ros() {
  local n
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do pkill -TERM -x "$n" 2>/dev/null || true; done
  sleep 1
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do pkill -KILL -x "$n" 2>/dev/null || true; done
  rm -f /tmp/care_collision_cdf_gpu_c5_5.sock /tmp/care_collision_cdf_gpu_c5_4.sock 2>/dev/null || true
}
trap cleanup_ros EXIT INT TERM

if [[ "$RUN_TARGETED" == true ]]; then
  OUT="$ROOT/targeted" \
  REPO="$REPO" \
  V1_CHECKPOINT="$V1_CHECKPOINT" \
  R1_CHECKPOINT="$R1_CHECKPOINT" \
  VIS_PYTHON="$VIS_PYTHON" \
  DEVICE=cuda \
  bash scripts/run_phase_e_v1_r1_targeted_qualification.sh
fi

run_one() {
  local label="$1"
  local ckpt="$2"
  local run_id="phase_e_r1qual_${label}_${CASE_ID}_${STAMP}"
  local runner_log="$ROOT/${label}_runner.log"
  local run_dir="$REPO/outputs/c5_5_vbc_gcdf_regime/$run_id/run"
  local eval_json="$ROOT/${label}_phase_d.json"
  local gen_log="$REPO/logs/c5_5_vbc_gcdf_regime/$run_id/run/waypoint_generator.log"

  cleanup_ros
  echo "[RUN] label=$label checkpoint=$ckpt run_id=$run_id"
  set +e
  (
    REPO="$REPO" \
    CASE_FILE="$CASE_FILE" \
    CASE_ID="$CASE_ID" \
    RUN_ID="$run_id" \
    RUN_SECONDS="$RUN_SECONDS" \
    GAZEBO_GUI="$GAZEBO_GUI" \
    USE_RVIZ="$USE_RVIZ" \
    EARLY_STOP_ON_GOAL=false \
    PER_SENSOR_CHECKPOINT="$ckpt" \
    PER_SENSOR_BRANCH_ASCENT_STEPS=1 \
    PER_SENSOR_MAX_BRANCH_ATTEMPTS=4 \
    bash scripts/run_phase_e_case026_per_sensor_hybrid.sh
  ) 2>&1 | tee "$runner_log"
  local run_rc=${PIPESTATUS[0]}
  set -e

  local eval_rc=2
  if [[ -d "$run_dir" ]]; then
    set +e
    python3 scripts/evaluate_phase_d_run.py \
      --repo "$REPO" \
      --run-dir "$run_dir" \
      --cases-json "$EVAL_CASES_JSON" \
      --case-id "$EVAL_CASE_ID" \
      --method "r1_runtime_qualification_${label}" \
      --trial-id "$STAMP" \
      --output-json "$eval_json" \
      >> "$runner_log" 2>&1
    eval_rc=$?
    set -e
  fi

  mkdir -p "$ROOT/${label}_artifacts"
  for name in regime_summary.csv visibility_acquisition_summary.csv candidate_vbc_summary.csv execution_vbc_summary.csv execution_gcdf_hard_hold.csv execution_gcdf_safety_summary.csv commit_summary.csv tracker_summary.csv local_planner_summary.csv goal_stop_status.json; do
    [[ -f "$run_dir/$name" ]] && cp -f "$run_dir/$name" "$ROOT/${label}_artifacts/$name"
  done
  [[ -f "$gen_log" ]] && cp -f "$gen_log" "$ROOT/${label}_generator.log"

  python3 - "$ROOT/${label}_status.json" "$run_rc" "$eval_rc" "$run_dir" "$eval_json" "$gen_log" <<'PY'
import json,os,sys
out,rr,er,run,ev,gen=sys.argv[1:]
json.dump({'runner_return_code':int(rr),'evaluator_return_code':int(er),'run_dir':run,
           'evaluation_json':ev if os.path.isfile(ev) else None,
           'generator_log':gen if os.path.isfile(gen) else None},open(out,'w'),indent=2)
PY
}

run_one v1 "$V1_CHECKPOINT"
run_one r1 "$R1_CHECKPOINT"
cleanup_ros

TARGETED_ARG=()
[[ -f "$ROOT/targeted/targeted_compare.json" ]] && TARGETED_ARG=(--targeted-compare "$ROOT/targeted/targeted_compare.json")
python3 scripts/analyze_phase_e_v1_r1_runtime_qualification.py \
  --root "$ROOT" \
  --v1-run-dir "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["run_dir"])' "$ROOT/v1_status.json")" \
  --r1-run-dir "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["run_dir"])' "$ROOT/r1_status.json")" \
  --v1-eval "$ROOT/v1_phase_d.json" \
  --r1-eval "$ROOT/r1_phase_d.json" \
  --v1-generator-log "$ROOT/v1_generator.log" \
  --r1-generator-log "$ROOT/r1_generator.log" \
  "${TARGETED_ARG[@]}"

cat > "$ROOT/metadata.txt" <<EOF
git_head=$(git rev-parse HEAD)
case_id=$CASE_ID
eval_case_id=$EVAL_CASE_ID
run_seconds=$RUN_SECONDS
v1_checkpoint=$V1_CHECKPOINT
r1_checkpoint=$R1_CHECKPOINT
online_branch_attempts=4
online_branch_ascent_steps=1
scalar_projector=unchanged
analytic_fov=unchanged
primitive_self_occlusion=unchanged
sparse_scp=unchanged
vbc=unchanged
gcdf=unchanged
tracker=unchanged
EOF

ZIP="$REPO/CAREPlanner_PHASE_E_V1_R1_RUNTIME_QUAL_${STAMP}.zip"
rm -f "$ZIP"
python3 - "$ROOT" "$ZIP" <<'PY'
import os,sys,zipfile
root,dst=sys.argv[1:]
with zipfile.ZipFile(dst,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for base,dirs,files in os.walk(root):
        dirs.sort();files.sort()
        for name in files:
            p=os.path.join(base,name); z.write(p,os.path.relpath(p,root))
print('[zip]',dst)
PY

echo "[SUMMARY] $ROOT/runtime_summary.md"
echo "[REPORT]  $ROOT/runtime_compare.json"
echo "[UPLOAD]  $ZIP"
cat "$ROOT/runtime_summary.md"
