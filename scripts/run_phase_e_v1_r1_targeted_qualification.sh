#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)}"
V1_CHECKPOINT="${V1_CHECKPOINT:-$REPO/src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt}"
R1_CHECKPOINT="${R1_CHECKPOINT:-$REPO/outputs/mainline_b/h9_scratch50k_r012_v1/formal/R1/final.pt}"
SCALAR_CHECKPOINT="${SCALAR_CHECKPOINT:-$REPO/src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt}"
DEVICE="${DEVICE:-cuda}"
VIS_PYTHON="${VIS_PYTHON:-python3}"
TARGETED_PARALLEL_4GPU="${TARGETED_PARALLEL_4GPU:-false}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
OUT="${OUT:-$REPO/outputs/phase_e_r1_runtime_qualification/$STAMP/targeted}"
mkdir -p "$OUT"
cd "$REPO"

"$VIS_PYTHON" -m py_compile \
  src/care_visibility_cdf/scripts/hierarchical_visibility_cdf_model.py \
  src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py \
  scripts/test_phase_e_v1_r1_runtime_adapter.py \
  scripts/test_phase_e_case026_targeted_per_sensor_fallback.py

"$VIS_PYTHON" scripts/test_phase_e_v1_r1_runtime_adapter.py \
  --v1-checkpoint "$V1_CHECKPOINT" \
  --r1-checkpoint "$R1_CHECKPOINT" \
  --device "$DEVICE" \
  --output "$OUT/runtime_adapter_test.json"

run_case026() {
  local gpu="$1"
  local label="$2"
  local ckpt="$3"
  local forced="$4"
  local suffix="$5"
  local out_json="$OUT/case026_${label}${suffix}.json"
  local out_log="$OUT/case026_${label}${suffix}.log"

  CUDA_VISIBLE_DEVICES="$gpu" "$VIS_PYTHON" scripts/test_phase_e_case026_targeted_per_sensor_fallback.py \
    --scalar-checkpoint "$SCALAR_CHECKPOINT" \
    --per-sensor-checkpoint "$ckpt" \
    --device "$DEVICE" \
    --projection-iters 10 \
    --projection-damping 0.5 \
    --projection-epsilon-f 0.03 \
    --projection-max-step-norm 0.25 \
    --root-refine-iters 12 \
    --root-tolerance-f 0.002 \
    --branch-ascent-steps 1 \
    --branch-step-size 0.05 \
    --branch-max-step-norm 0.25 \
    --max-branch-attempts 8 \
    --force-first-sensor "$forced" \
    --output "$out_json" \
    > >(tee "$out_log") 2>&1
}

if [[ "$TARGETED_PARALLEL_4GPU" == true ]]; then
  "$VIS_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() >= 4, torch.cuda.device_count()
print("[targeted] 4-GPU parallel mode:", [torch.cuda.get_device_name(i) for i in range(4)], flush=True)
PY

  # Frozen primary protocol on GPU0/1: force historical S4 first.
  run_case026 0 v1 "$V1_CHECKPOINT" 4 "" &
  P0=$!
  run_case026 1 r1 "$R1_CHECKPOINT" 4 "" &
  P1=$!

  # Supplemental online-order diagnostic on GPU2/3: pure learned ranking.
  run_case026 2 v1 "$V1_CHECKPOINT" -1 "_pure" &
  P2=$!
  run_case026 3 r1 "$R1_CHECKPOINT" -1 "_pure" &
  P3=$!

  rc=0
  wait "$P0" || rc=1
  wait "$P1" || rc=1
  wait "$P2" || rc=1
  wait "$P3" || rc=1
  [[ "$rc" -eq 0 ]] || { echo "[ERROR] one or more 4-GPU targeted subprocesses failed" >&2; exit 5; }
else
  run_case026 0 v1 "$V1_CHECKPOINT" 4 ""
  run_case026 0 r1 "$R1_CHECKPOINT" 4 ""
fi

"$VIS_PYTHON" - "$OUT" <<'PY'
import json, os, sys
root=sys.argv[1]
v=json.load(open(os.path.join(root,'case026_v1.json')))
r=json.load(open(os.path.join(root,'case026_r1.json')))
a=json.load(open(os.path.join(root,'runtime_adapter_test.json')))

def summary(x):
    ps=x['per_sensor']
    return {
        'verdict':x['verdict'],
        'learned_ranking':ps['learned_ranking'],
        'tested_order':ps['tested_order'],
        'rejected_sensor_ids':ps['rejected_sensor_ids'],
        'selected_sensor_id':ps['selected_sensor_id'],
        'strict_original_mode_fallback':ps['strict_original_mode_fallback'],
        'branch_compute_ms':ps['branch_compute_ms'],
        'attempts':[{
            'rank':z['rank'],'sensor_id':z['sensor_id'],
            'root_source':z['root_source'],'initial_score':z['initial_score'],
            'final_score':z['final_score'],
            'conservative_g':z['geometry']['min_conservative_g'],
            'self_occluded':z['geometry']['any_primitive_self_occluded'],
            'accepted':z['geometry']['accepted'],
            'reject_reason':z['geometry']['reject_reason'],
        } for z in ps['attempts']],
    }

def sanity(x):
    g=x['known_blocked_s4_geometry']
    return bool(g.get('any_primitive_self_occluded')) and float(g.get('min_conservative_g',-1)) > 0.0

pure={}
for name in ('v1','r1'):
    p=os.path.join(root,f'case026_{name}_pure.json')
    if os.path.isfile(p):
        pure[name.upper()]=summary(json.load(open(p)))

report={
    'qualification':'case026_targeted_v1_vs_r1',
    'adapter_status':a.get('status'),
    'known_blocked_s4_sanity':{'V1':sanity(v),'R1':sanity(r)},
    'V1':summary(v),'R1':summary(r),
    'pure_ranking_diagnostic':pure,
}
report['valid_diagnostic']=bool(report['adapter_status']=='PASS' and all(report['known_blocked_s4_sanity'].values()))
report['verdict']='TARGETED_VALID' if report['valid_diagnostic'] else 'TARGETED_INVALID'
json.dump(report,open(os.path.join(root,'targeted_compare.json'),'w'),indent=2)

lines=[
    '# Case026 targeted runtime qualification: V1 vs R1','',
    f"Verdict: **{report['verdict']}**",'',
    '| Model | branch verdict | selected sensor | rejected | strict S4→other | branch ms |',
    '|---|---|---:|---|---:|---:|',
]
for name in ('V1','R1'):
    x=report[name]
    lines.append(f"| {name} | {x['verdict']} | {x['selected_sensor_id']} | {x['rejected_sensor_ids']} | {int(x['strict_original_mode_fallback'])} | {x['branch_compute_ms']:.3f} |")
if pure:
    lines += ['', '## Pure learned ranking diagnostic','',
              '| Model | branch verdict | selected sensor | learned ranking | branch ms |',
              '|---|---|---:|---|---:|']
    for name in ('V1','R1'):
        if name in pure:
            x=pure[name]
            lines.append(f"| {name} | {x['verdict']} | {x['selected_sensor_id']} | {x['learned_ranking']} | {x['branch_compute_ms']:.3f} |")

lines += ['', 'Historical blocked S4 sanity:', f"- V1: {report['known_blocked_s4_sanity']['V1']}", f"- R1: {report['known_blocked_s4_sanity']['R1']}", '',
          'NO_CLEAR_SENSOR_BRANCH is a valid targeted outcome. This stage does not certify trajectory/collision/execution safety.']
open(os.path.join(root,'targeted_summary.md'),'w').write('\n'.join(lines)+'\n')
print(json.dumps(report,indent=2))
if not report['valid_diagnostic']: raise SystemExit(4)
print('[done] targeted runtime qualification valid')
PY

echo "[TARGETED OUT] $OUT"
