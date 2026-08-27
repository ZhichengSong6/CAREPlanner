#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-15.0}"
CDF_ENV="${CDF_ENV:-ncdf_l4c}"
CDF_DEVICE="${CDF_DEVICE:-cpu}"
CHECKPOINT="${CHECKPOINT:-${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict.pt}"
SHADOW_RATE="${SHADOW_RATE:-5.0}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.50}"
DEDUP_RESOLUTION="${DEDUP_RESOLUTION:-0.05}"
SIGNED_ZERO_BAND="${SIGNED_ZERO_BAND:-0.05}"

ROOT_OUT="${ROOT_OUT:-${REPO}/outputs/phase_c5_2_multi_step_forbidden_space_cdf_shadow/${CASE_ID}}"
ROOT_LOG="${ROOT_LOG:-${REPO}/logs/phase_c5_2_multi_step_forbidden_space_cdf_shadow/${CASE_ID}}"
C4_OUT="${ROOT_OUT}/c4_9"
C4_LOG="${ROOT_LOG}/c4_9"
SHADOW_JSONL="${ROOT_OUT}/multi_step_forbidden_space_cdf.jsonl"
SHADOW_SUMMARY_CSV="${ROOT_OUT}/multi_step_forbidden_space_summary.csv"
SUMMARY_JSON="${ROOT_OUT}/c5_2_multi_step_forbidden_space_cdf_summary.json"
ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C5_2_multi_step_forbidden_space_cdf_shadow_${CASE_ID}.zip}"

cd "${REPO}"
echo "[C5.2] branch: $(git branch --show-current)"
echo "[C5.2] head:   $(git rev-parse HEAD)"
echo "[C5.2] case:   ${CASE_ID}"
echo "[C5.2] checkpoint: ${CHECKPOINT}"
echo "[C5.2] runtime: ${CDF_ENV} / ${CDF_DEVICE}"
echo "[C5.2] mode: SHADOW ONLY (no MPC constraint enforcement)"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] collision CDF checkpoint not found: ${CHECKPOINT}"
  exit 2
fi

catkin build care_confidence_map care_collision_cdf
source devel/setup.bash

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found"
  exit 3
fi

rm -rf "${ROOT_OUT}" "${ROOT_LOG}"
rm -f "${ZIP_PATH}"
mkdir -p "${ROOT_OUT}" "${ROOT_LOG}"

EXTRA_PIDS=()

kill_group() {
  local pid="${1:-}"
  [[ -z "${pid}" ]] && return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || true
    sleep 0.15
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 0.15
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup_extra() {
  local pid
  for pid in "${EXTRA_PIDS[@]:-}"; do
    kill_group "${pid}"
  done
  EXTRA_PIDS=()
}
trap cleanup_extra EXIT INT TERM

# Start a watcher before C4.9 launches ROS. Once the master appears it loads
# parameters and starts the learned CDF in the same isolated conda-process
# pattern already used by the visibility CDF.
setsid bash -lc "
  set -e
  cd '${REPO}'
  source devel/setup.bash
  while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done

  rosparam load src/care_collision_cdf/config/collision_cdf.yaml /collision_cdf_server
  rosparam set /collision_cdf_server/checkpoint '${CHECKPOINT}'
  rosparam set /collision_cdf_server/device '${CDF_DEVICE}'
  rosparam set /collision_cdf_server/checkpoint_key latest

  source '${CONDA_SH}'
  conda activate '${CDF_ENV}'
  cd '${REPO}'
  source devel/setup.bash
  exec python -u src/care_collision_cdf/scripts/collision_cdf_server_node.py
" >"${ROOT_LOG}/collision_cdf_server.log" 2>&1 &
EXTRA_PIDS+=("$!")

# Shadow evaluator starts only after the generated pair service exists.
setsid bash -lc "
  set -e
  cd '${REPO}'
  source devel/setup.bash
  while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done
  while ! rosservice list 2>/dev/null | grep -q '^/care_planner/collision_cdf/query_pairs$'; do
    sleep 0.05
  done
  exec rosrun care_collision_cdf multi_step_forbidden_space_cdf_shadow_node.py \
    _input_topic:=/care_planner/trajectory_risk/low_confidence_sweep_pairs \
    _pair_service:=/care_planner/collision_cdf/query_pairs \
    _summary_topic:=/care_planner/collision_cdf/multi_step_forbidden_space_summary \
    _output_jsonl:='${SHADOW_JSONL}' \
    _rate:='${SHADOW_RATE}' \
    _dedup_resolution:='${DEDUP_RESOLUTION}' \
    _signed_zero_band:='${SIGNED_ZERO_BAND}' \
    _max_pairs:=8000
" >"${ROOT_LOG}/multi_step_forbidden_space_shadow.log" 2>&1 &
EXTRA_PIDS+=("$!")

# Record the compact human-readable shadow stream.
setsid bash -lc "
  set -e
  cd '${REPO}'
  source devel/setup.bash
  while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done
  while ! rostopic list 2>/dev/null | grep -q '^/care_planner/collision_cdf/multi_step_forbidden_space_summary$'; do
    sleep 0.05
  done
  exec rostopic echo -p /care_planner/collision_cdf/multi_step_forbidden_space_summary
" >"${SHADOW_SUMMARY_CSV}" 2>"${ROOT_LOG}/shadow_summary_recorder.err" &
EXTRA_PIDS+=("$!")

# Reuse the frozen C4.9 experiment. The only optional launch overrides are:
#   * trajectory_risk evaluates the RAW MPC q_1...q_K instead of task trajectory;
#   * it exports low-confidence sweep pairs for the shadow observer.
# All planner / VBC / commit semantics remain unchanged.
TRAJECTORY_RISK_INPUT_TOPIC="/care_planner/mpc/predicted_trajectory" \
FORBIDDEN_SPACE_PAIR_PUBLISH_ENABLED="true" \
FORBIDDEN_SPACE_PAIR_TOPIC="/care_planner/trajectory_risk/low_confidence_sweep_pairs" \
FORBIDDEN_SPACE_CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD}" \
CASE_ID="${CASE_ID}" \
RUN_SECONDS="${RUN_SECONDS}" \
OUT="${C4_OUT}" \
LOG="${C4_LOG}" \
ZIP_PATH="${ROOT_OUT}/C4_9_baseline_${CASE_ID}.zip" \
bash scripts/run_and_pack_phase_c4_9_blocker_aware_repair.sh

# Let the last published pair cloud be consumed before tearing down observers.
sleep 0.25
cleanup_extra
trap - EXIT INT TERM

python3 - "${SHADOW_JSONL}" "${SUMMARY_JSON}" "${CONFIDENCE_THRESHOLD}" "${SIGNED_ZERO_BAND}" <<'PY'
import json, math, os, statistics, sys
src, dst, confidence_threshold, signed_zero_band = sys.argv[1:5]
confidence_threshold = float(confidence_threshold)
signed_zero_band = float(signed_zero_band)
records=[]
if os.path.isfile(src):
    with open(src, errors="replace") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass

nonempty=[r for r in records if int(r.get("deduplicated_pair_count",0))>0]
def vals(key, rows=nonempty):
    out=[]
    for r in rows:
        v=r.get(key)
        if isinstance(v,(int,float)) and math.isfinite(float(v)):
            out.append(float(v))
    return out
def stats(a):
    if not a:
        return None
    a=sorted(a)
    def q(frac):
        if len(a)==1:return a[0]
        p=frac*(len(a)-1); lo=int(p); hi=min(lo+1,len(a)-1); w=p-lo
        return a[lo]*(1-w)+a[hi]*w
    return {
        "min":min(a),
        "median":statistics.median(a),
        "mean":statistics.fmean(a),
        "p95":q(0.95),
        "max":max(a),
    }

pairs=[float(r["deduplicated_pair_count"]) for r in nonempty]
raw=[float(r["raw_pair_count"]) for r in nonempty]
steps=[float(r["active_step_count"]) for r in nonempty]
infer=vals("inference_ms")
rtt=vals("service_roundtrip_ms")
dmins=[
    float(r["distance"]["min"])
    for r in nonempty
    if isinstance(r.get("distance"),dict)
    and isinstance(r["distance"].get("min"),(int,float))
]
dp05=[
    float(r["distance"]["p05"])
    for r in nonempty
    if isinstance(r.get("distance"),dict)
    and isinstance(r["distance"].get("p05"),(int,float))
]

signed_negative_rates=[
    float(r["signed_counts"]["negative_rate"])
    for r in nonempty
    if isinstance(r.get("signed_counts"),dict)
    and isinstance(r["signed_counts"].get("negative_rate"),(int,float))
]
signed_near_zero_rates=[
    float(r["signed_counts"]["near_zero_rate"])
    for r in nonempty
    if isinstance(r.get("signed_counts"),dict)
    and isinstance(r["signed_counts"].get("near_zero_rate"),(int,float))
]
signed_positive_rates=[
    float(r["signed_counts"]["positive_rate"])
    for r in nonempty
    if isinstance(r.get("signed_counts"),dict)
    and isinstance(r["signed_counts"].get("positive_rate"),(int,float))
]

global_min_record=None
if nonempty:
    candidates=[
        r for r in nonempty
        if isinstance(r.get("distance"),dict)
        and isinstance(r["distance"].get("min"),(int,float))
    ]
    if candidates:
        global_min_record=min(candidates,key=lambda r:r["distance"]["min"])

summary={
    "phase":"C5.2",
    "name":"Multi-Step Forbidden-Space CDF Shadow Diagnostic",
    "shadow_only":True,
    "constraint_enforced":False,
    "source":"low-confidence predicted body-sweep samples from raw MPC trajectory",
    "confidence_threshold":confidence_threshold,
    "records_total":len(records),
    "records_nonempty":len(nonempty),
    "pair_count":stats(pairs),
    "raw_pair_count":stats(raw),
    "active_step_count":stats(steps),
    "cdf_inference_ms":stats(infer),
    "service_roundtrip_ms":stats(rtt),
    "record_min_distance":stats(dmins),
    "record_p05_distance":stats(dp05),
    "signed_zero_band":signed_zero_band,
    "record_negative_rate":stats(signed_negative_rates),
    "record_near_zero_rate":stats(signed_near_zero_rates),
    "record_positive_rate":stats(signed_positive_rates),
    "global_min_pair":(
        global_min_record.get("global_min_pair")
        if global_min_record else None
    ),
}
with open(dst,"w") as f:
    json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))

if not records:
    raise SystemExit("[ERROR] C5.2 produced no shadow records")
if not nonempty:
    raise SystemExit("[ERROR] C5.2 never observed a non-empty forbidden-space pair batch")
PY

cat > "${ROOT_OUT}/c5_2_run_metadata.txt" <<EOF
branch=$(git branch --show-current)
head=$(git rev-parse HEAD)
case_id=${CASE_ID}
run_seconds=${RUN_SECONDS}
name=Multi-Step Forbidden-Space CDF Shadow Diagnostic
shadow_only=true
constraint_enforced=false
trajectory_source=/care_planner/mpc/predicted_trajectory
confidence_threshold=${CONFIDENCE_THRESHOLD}
dedup_resolution=${DEDUP_RESOLUTION}
signed_zero_band=${SIGNED_ZERO_BAND}
cdf_env=${CDF_ENV}
cdf_device=${CDF_DEVICE}
checkpoint=${CHECKPOINT}
EOF

cd "${REPO}"
zip -r "${ZIP_PATH}" "${ROOT_OUT#${REPO}/}" "${ROOT_LOG#${REPO}/}"

echo ""
echo "[C5.2 COMPLETE]"
echo "[SUMMARY] ${SUMMARY_JSON}"
echo "[JSONL]   ${SHADOW_JSONL}"
echo "[ZIP]     ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
