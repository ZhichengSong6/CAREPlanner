#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-15.0}"

GPU_ENV="${GPU_ENV:-viscdf}"
GPU_DEVICE="${GPU_DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt}"
GPU_SOCKET="${GPU_SOCKET:-/tmp/care_collision_cdf_gpu_c5_3a.sock}"

CDF_SAFETY_MARGIN="${CDF_SAFETY_MARGIN:-0.0}"
CDF_TRUST_REGION_Q_INF="${CDF_TRUST_REGION_Q_INF:-0.20}"
CDF_HORIZON_STEPS="${CDF_HORIZON_STEPS:-20}"
CDF_SNAPSHOT_TIMEOUT="${CDF_SNAPSHOT_TIMEOUT:-0.75}"

SHADOW_RATE="${SHADOW_RATE:-20.0}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.50}"
MAP_RESOLUTION="${MAP_RESOLUTION:-0.05}"
PROXIMITY_MARGIN="${PROXIMITY_MARGIN:-0.075}"
MAX_PAIRS_PER_STEP="${MAX_PAIRS_PER_STEP:-250}"
SIGNED_ZERO_BAND="${SIGNED_ZERO_BAND:-0.05}"

ANCHOR_TOPIC="/care_planner/trajectory_risk/body_sweep_anchors"
MAP_TOPIC="/care_planner/confidence_map/points"
CONSTRAINT_BATCH_TOPIC="/care_planner/collision_cdf/constraint_batch"
CDF_SHADOW_PREDICTION_TOPIC="/care_planner/mpc/cdf_shadow_predicted_trajectory"
CDF_SHADOW_SUMMARY_TOPIC="/care_planner/mpc/cdf_shadow_summary"
CDF_SHADOW_VBC_SUMMARY_TOPIC="/care_planner/cdf_shadow_vbc/summary"

ROOT_OUT="${ROOT_OUT:-${REPO}/outputs/phase_c5_3a_cdf_shadow/${CASE_ID}}"
ROOT_LOG="${ROOT_LOG:-${REPO}/logs/phase_c5_3a_cdf_shadow/${CASE_ID}}"
C4_OUT="${ROOT_OUT}/c4_9"
C4_LOG="${ROOT_LOG}/c4_9"

SELECTOR_JSONL="${ROOT_OUT}/cpp_selector_gpu.jsonl"
SELECTOR_SUMMARY_CSV="${ROOT_OUT}/cpp_selector_gpu_summary.csv"
CDF_SHADOW_SUMMARY_CSV="${ROOT_OUT}/cdf_shadow_qp_summary.csv"
CDF_SHADOW_VBC_SUMMARY_CSV="${ROOT_OUT}/cdf_shadow_vbc_summary.csv"
SUMMARY_JSON="${ROOT_OUT}/c5_3a_cdf_shadow_summary.json"
ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C5_3a_cdf_shadow_${CASE_ID}.zip}"

cd "${REPO}"

echo "[C5.3a] shell syntax preflight..."
bash -n scripts/run_and_pack_phase_c5_3a_cdf_shadow_piqp.sh
bash -n scripts/run_and_pack_phase_c4_9_blocker_aware_repair.sh
bash -n scripts/run_phase_c4_4_verified_regime_smoke.sh

echo "[C5.3a] branch: $(git branch --show-current)"
echo "[C5.3a] head:   $(git rev-parse HEAD)"
echo "[C5.3a] case:   ${CASE_ID}"
echo "[C5.3a] mode: SHADOW CONSTRAINED PIQP ONLY"
echo "[C5.3a] d_safe: ${CDF_SAFETY_MARGIN}"
echo "[C5.3a] trust region q_inf: ${CDF_TRUST_REGION_Q_INF}"
echo "[C5.3a] CDF horizon steps: ${CDF_HORIZON_STEPS}"
echo "[C5.3a] GPU: ${GPU_ENV} / ${GPU_DEVICE}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] signed CDF checkpoint not found: ${CHECKPOINT}"
  exit 2
fi

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found"
  exit 3
fi

catkin build care_confidence_map care_collision_cdf egocentric_arm_planner
source devel/setup.bash

rm -rf "${ROOT_OUT}" "${ROOT_LOG}"
rm -f "${ZIP_PATH}" "${GPU_SOCKET}"
mkdir -p "${ROOT_OUT}" "${ROOT_LOG}"

echo "[C5.3a] CUDA preflight..."
bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${GPU_ENV}'
  cd '${REPO}'
  python - <<'PY'
import torch
print('[preflight] torch:', torch.__version__)
print('[preflight] cuda runtime:', torch.version.cuda)
print('[preflight] cuda available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('[ERROR] GPU_ENV has no usable CUDA PyTorch')
print('[preflight] GPU:', torch.cuda.get_device_name(0))
PY
"

PIDS=()

kill_group() {
  local pid="${1:-}"
  [[ -z "${pid}" ]] && return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || true
    sleep 0.20
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 0.20
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill_group "${pid}"
  done
  PIDS=()
  rm -f "${GPU_SOCKET}"
}
trap cleanup EXIT INT TERM

# Persistent CUDA signed-CDF worker.
setsid bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${GPU_ENV}'
  cd '${REPO}'
  exec python -u src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py \
    --checkpoint '${CHECKPOINT}' \
    --activation gelu \
    --device '${GPU_DEVICE}' \
    --socket '${GPU_SOCKET}' \
    --max-pairs 8000 \
    --warmup-pairs 2048
" >"${ROOT_LOG}/gpu_worker.log" 2>&1 &
PIDS+=("$!")

READY=0
for _ in $(seq 1 200); do
  if [[ -S "${GPU_SOCKET}" ]]; then
    READY=1
    break
  fi
  if ! kill -0 "${PIDS[0]}" 2>/dev/null; then
    break
  fi
  sleep 0.05
done
if [[ "${READY}" != "1" ]]; then
  echo "[ERROR] GPU worker socket did not become ready"
  tail -n 160 "${ROOT_LOG}/gpu_worker.log" || true
  exit 4
fi

# Transport topology watchdog. This is diagnostic only and does not subscribe
# to the large batch payload itself.
setsid bash -lc "
  set +e
  cd '${REPO}'
  source devel/setup.bash
  while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done
  for i in \$(seq 1 80); do
    echo "===== sample=\$i ====="
    date '+wall=%s.%N'
    rostopic info '${CONSTRAINT_BATCH_TOPIC}' 2>&1
    rosnode info /velocity_qp_mpc_waypoint_node 2>&1 | \
      grep -A40 -B5 -E 'Subscriptions:|Publications:' || true
    sleep 0.25
  done
" >"${ROOT_LOG}/constraint_batch_topology.log" 2>&1 &
PIDS+=("$!")

record_topic() {
  local topic="$1"
  local path="$2"
  setsid bash -lc "
    set -e
    cd '${REPO}'
    source devel/setup.bash
    while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done
    while ! rostopic list 2>/dev/null | grep -q '^${topic}$'; do sleep 0.05; done
    exec rostopic echo -p '${topic}'
  " >"${path}" 2>"${path}.err" &
  PIDS+=("$!")
}

record_topic /care_planner/collision_cdf/cpp_gpu_online_summary   "${SELECTOR_SUMMARY_CSV}"
record_topic "${CDF_SHADOW_SUMMARY_TOPIC}"   "${CDF_SHADOW_SUMMARY_CSV}"
record_topic "${CDF_SHADOW_VBC_SUMMARY_TOPIC}"   "${CDF_SHADOW_VBC_SUMMARY_CSV}"

# Frozen C4.9 remains the actual planner/execution path. C5.3a only enables:
#   raw-MPC body anchors + paired CDF batch + asynchronous shadow PIQP.
TRAJECTORY_RISK_INPUT_TOPIC="/care_planner/mpc/predicted_trajectory" \
FORBIDDEN_SPACE_PAIR_PUBLISH_ENABLED="true" \
FORBIDDEN_SPACE_PAIR_TOPIC="${ANCHOR_TOPIC}" \
FORBIDDEN_SPACE_CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD}" \
CDF_SHADOW_ENABLED="true" \
CDF_SHADOW_SAFETY_MARGIN="${CDF_SAFETY_MARGIN}" \
CDF_SHADOW_TRUST_REGION_Q_INF="${CDF_TRUST_REGION_Q_INF}" \
CDF_SHADOW_HORIZON_STEPS="${CDF_HORIZON_STEPS}" \
CDF_SHADOW_SNAPSHOT_TIMEOUT="${CDF_SNAPSHOT_TIMEOUT}" \
CDF_SHADOW_CONSTRAINT_BATCH_TOPIC="${CONSTRAINT_BATCH_TOPIC}" \
CDF_SHADOW_PREDICTION_TOPIC="${CDF_SHADOW_PREDICTION_TOPIC}" \
CDF_SHADOW_SUMMARY_TOPIC="${CDF_SHADOW_SUMMARY_TOPIC}" \
CDF_SELECTOR_ENABLED="true" \
CDF_SELECTOR_GPU_SOCKET="${GPU_SOCKET}" \
CDF_SELECTOR_OUTPUT_JSONL="${SELECTOR_JSONL}" \
CDF_SELECTOR_RATE="${SHADOW_RATE}" \
CDF_SELECTOR_MAP_RESOLUTION="${MAP_RESOLUTION}" \
CDF_SELECTOR_PROXIMITY_MARGIN="${PROXIMITY_MARGIN}" \
CDF_SELECTOR_MAX_PAIRS_PER_STEP="${MAX_PAIRS_PER_STEP}" \
CDF_SELECTOR_SIGNED_ZERO_BAND="${SIGNED_ZERO_BAND}" \
CDF_SHADOW_VBC_AUDIT_ENABLED="true" \
CDF_SHADOW_VBC_SUMMARY_TOPIC="${CDF_SHADOW_VBC_SUMMARY_TOPIC}" \
CASE_ID="${CASE_ID}" \
RUN_SECONDS="${RUN_SECONDS}" \
OUT="${C4_OUT}" \
LOG="${C4_LOG}" \
ZIP_PATH="${ROOT_OUT}/C4_9_baseline_${CASE_ID}.zip" \
bash scripts/run_and_pack_phase_c4_9_blocker_aware_repair.sh

sleep 0.50
cleanup
trap - EXIT INT TERM

pack_debug_bundle() {
  cd "${REPO}"
  rm -f "${ZIP_PATH}"
  zip -r "${ZIP_PATH}" "${ROOT_OUT#${REPO}/}" "${ROOT_LOG#${REPO}/}" >/dev/null
  echo "[C5.3a DEBUG ZIP] ${ZIP_PATH}"
}

# Preserve the complete outer bundle even if post-processing fails.
trap 'rc=$?; if [[ -d "${ROOT_OUT}" || -d "${ROOT_LOG}" ]]; then pack_debug_bundle || true; fi; exit $rc' EXIT

python3 -   "${CDF_SHADOW_SUMMARY_CSV}"   "${CDF_SHADOW_VBC_SUMMARY_CSV}"   "${C4_OUT}/c4_9_blocker_aware_summary.json"   "${SUMMARY_JSON}"   "${CDF_SAFETY_MARGIN}"   "${CDF_TRUST_REGION_Q_INF}"   "${CDF_HORIZON_STEPS}" <<'PY'
import csv
import json
import math
import os
import re
import statistics
import sys

shadow_csv, vbc_csv, raw_json, dst, dsafe, trust, horizon = sys.argv[1:8]
dsafe=float(dsafe); trust=float(trust); horizon=int(horizon)
TOK=re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')

def rows(path):
    out=[]
    if not os.path.isfile(path):
        return out
    with open(path,newline='',errors='replace') as f:
        rd=csv.reader(f)
        header=next(rd,[])
        if not header:
            return out
        ti=header.index('%time') if '%time' in header else 0
        di=header.index('field.data') if 'field.data' in header else 1
        for row in rd:
            if len(row)<=di:
                continue
            data=dict(TOK.findall(','.join(row[di:])))
            try:
                data['_t']=float(row[ti])/1e9
            except Exception:
                data['_t']=math.nan
            if data:
                out.append(data)
    return out

def fnum(v):
    try:
        return float(str(v).replace('ms',''))
    except Exception:
        return math.nan

def stats(vals):
    a=[float(v) for v in vals if isinstance(v,(int,float)) and math.isfinite(float(v))]
    if not a:
        return None
    a.sort()
    def q(frac):
        if len(a)==1:return a[0]
        p=frac*(len(a)-1); lo=int(p); hi=min(lo+1,len(a)-1); w=p-lo
        return a[lo]*(1-w)+a[hi]*w
    return {
        'min':min(a),
        'median':statistics.median(a),
        'mean':statistics.fmean(a),
        'p95':q(0.95),
        'max':max(a),
    }

shadow=rows(shadow_csv)
vbc=rows(vbc_csv)

if not shadow:
    raise SystemExit("[ERROR] C5.3a produced no shadow-QP records")
if not vbc:
    raise SystemExit("[ERROR] C5.3a produced no shadow exact-VBC records")

def col(name, source=shadow):
    return [fnum(r.get(name)) for r in source]

solved=[r for r in shadow if r.get('solved')=='1']
combined=[]
for r in shadow:
    a=fnum(r.get('upstream_pipeline_ms'))
    b=fnum(r.get('shadow_solve_ms'))
    if math.isfinite(a) and math.isfinite(b):
        combined.append(a+b)

raw={}
if os.path.isfile(raw_json):
    raw=json.load(open(raw_json))

vbc_pred=[r for r in vbc if r.get('trajectory_source','predicted')=='predicted']
vbc_safe=sum(r.get('has_violation')=='0' for r in vbc_pred)
vbc_unsafe=sum(r.get('has_violation')=='1' for r in vbc_pred)

last=shadow[-1] if shadow else {}
summary={
    'phase':'C5.3a',
    'name':'Timestamp-Matched Signed-CDF Constrained Shadow PIQP',
    'shadow_only':True,
    'constraint_enforced_on_execution':False,
    'cdf_safety_margin':dsafe,
    'cdf_trust_region_q_inf':trust,
    'cdf_constraint_horizon_steps':horizon,
    'shadow_qp_records':len(shadow),
    'shadow_qp_solved_records':len(solved),
    'shadow_qp_solve_rate':len(solved)/len(shadow) if shadow else 0.0,
    'active_constraint_rows':stats(col('active_rows')),
    'qlin_error_inf':stats(col('qlin_error_inf')),
    'raw_min_cdf_distance':stats(col('raw_min_d')),
    'shadow_linearized_min_cdf_distance':stats(col('shadow_linearized_min_d')),
    'upstream_cdf_pipeline_ms':stats(col('upstream_pipeline_ms')),
    'shadow_piqp_solve_ms':stats(col('shadow_solve_ms')),
    'cdf_pipeline_plus_shadow_solve_ms':stats(combined),
    'raw_to_shadow_end_to_end_age_ms':stats(col('end_to_end_age_ms')),
    'last_jobs_received':int(float(last.get('jobs_received',0))) if last else 0,
    'last_jobs_processed':int(float(last.get('jobs_processed',0))) if last else 0,
    'last_jobs_dropped':int(float(last.get('jobs_dropped',0))) if last else 0,
    'last_stamp_miss':int(float(last.get('stamp_miss',0))) if last else 0,
    'shadow_vbc_records':len(vbc_pred),
    'shadow_vbc_safe_records':vbc_safe,
    'shadow_vbc_unsafe_records':vbc_unsafe,
    'shadow_vbc_safe_rate':vbc_safe/len(vbc_pred) if vbc_pred else None,
    'raw_candidate_safe_records':raw.get('candidate_safe'),
    'raw_candidate_unsafe_records':raw.get('candidate_unsafe'),
    'raw_execution_unsafe_records':raw.get('execution_unsafe'),
    'raw_final_regime_state':raw.get('final_regime_state'),
    'raw_blocker_push_count':raw.get('blocker_push_count'),
    'raw_blocker_pop_count':raw.get('blocker_pop_count'),
}

combined_stats=summary['cdf_pipeline_plus_shadow_solve_ms']
summary['combined_compute_p95_under_50ms']=bool(
    combined_stats and combined_stats['p95'] < 50.0
)
qlin=summary['qlin_error_inf']
summary['timestamp_and_linearization_alignment_pass']=bool(
    qlin and qlin['p95'] < 1e-4 and summary['last_stamp_miss']==0
)

with open(dst,'w') as f:
    json.dump(summary,f,indent=2)

print("")
print("========== C5.3a SHADOW PIQP SUMMARY ==========")
print(json.dumps(summary,indent=2))
print("================================================")
PY

cat > "${ROOT_OUT}/c5_3a_run_metadata.txt" <<EOF
branch=$(git branch --show-current)
head=$(git rev-parse HEAD)
case_id=${CASE_ID}
run_seconds=${RUN_SECONDS}
shadow_only=true
constraint_enforced_on_execution=false
cdf_variant=signed
cdf_activation=gelu
cdf_safety_margin=${CDF_SAFETY_MARGIN}
cdf_trust_region_q_inf=${CDF_TRUST_REGION_Q_INF}
cdf_constraint_horizon_steps=${CDF_HORIZON_STEPS}
selector_launch_mode=integrated_same_roslaunch
shadow_vbc_launch_mode=integrated_same_roslaunch
constraint_batch_topic=${CONSTRAINT_BATCH_TOPIC}
shadow_prediction_topic=${CDF_SHADOW_PREDICTION_TOPIC}
shadow_vbc_summary_topic=${CDF_SHADOW_VBC_SUMMARY_TOPIC}
EOF

trap - EXIT
pack_debug_bundle

echo ""
echo "[C5.3a COMPLETE]"
echo "[SUMMARY] ${SUMMARY_JSON}"
echo "[SHADOW QP] ${CDF_SHADOW_SUMMARY_CSV}"
echo "[SHADOW VBC] ${CDF_SHADOW_VBC_SUMMARY_CSV}"
echo "[ZIP] ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
