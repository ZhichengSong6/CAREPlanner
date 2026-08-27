#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-15.0}"

GPU_ENV="${GPU_ENV:-viscdf}"
GPU_DEVICE="${GPU_DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt}"
GPU_SOCKET="${GPU_SOCKET:-/tmp/care_collision_cdf_gpu_c5_2g.sock}"

SHADOW_RATE="${SHADOW_RATE:-20.0}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.50}"
MAP_RESOLUTION="${MAP_RESOLUTION:-0.05}"
PROXIMITY_MARGIN="${PROXIMITY_MARGIN:-0.075}"
MAX_PAIRS_PER_STEP="${MAX_PAIRS_PER_STEP:-250}"
SIGNED_ZERO_BAND="${SIGNED_ZERO_BAND:-0.05}"

ANCHOR_TOPIC="/care_planner/trajectory_risk/body_sweep_anchors"
MAP_TOPIC="/care_planner/confidence_map/points"
SUMMARY_TOPIC="/care_planner/collision_cdf/gpu_online_summary"

ROOT_OUT="${ROOT_OUT:-${REPO}/outputs/phase_c5_2g_gpu_online/${CASE_ID}}"
ROOT_LOG="${ROOT_LOG:-${REPO}/logs/phase_c5_2g_gpu_online/${CASE_ID}}"
C4_OUT="${ROOT_OUT}/c4_9"
C4_LOG="${ROOT_LOG}/c4_9"
JSONL="${ROOT_OUT}/gpu_online.jsonl"
SUMMARY_CSV="${ROOT_OUT}/gpu_online_summary.csv"
SUMMARY_JSON="${ROOT_OUT}/c5_2g_gpu_online_summary.json"
ZIP_PATH="${ZIP_PATH:-${REPO}/CAREPlanner_C5_2g_gpu_online_${CASE_ID}.zip}"

cd "${REPO}"

echo "[C5.2g] branch: $(git branch --show-current)"
echo "[C5.2g] head:   $(git rev-parse HEAD)"
echo "[C5.2g] case:   ${CASE_ID}"
echo "[C5.2g] env:    ${GPU_ENV}"
echo "[C5.2g] device: ${GPU_DEVICE}"
echo "[C5.2g] checkpoint: ${CHECKPOINT}"
echo "[C5.2g] rate: ${SHADOW_RATE} Hz"
echo "[C5.2g] mode: SHADOW ONLY"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] signed checkpoint not found: ${CHECKPOINT}"
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

catkin build care_confidence_map care_collision_cdf
source devel/setup.bash

rm -rf "${ROOT_OUT}" "${ROOT_LOG}"
rm -f "${ZIP_PATH}" "${GPU_SOCKET}"
mkdir -p "${ROOT_OUT}" "${ROOT_LOG}"

echo "[C5.2g] CUDA preflight..."
bash -lc "
  set -e
  source '${CONDA_SH}'
  conda activate '${GPU_ENV}'
  cd '${REPO}'
  python - <<'PY'
import sys
import torch
print('[preflight] python:', sys.executable)
print('[preflight] torch:', torch.__version__)
print('[preflight] CUDA runtime:', torch.version.cuda)
print('[preflight] cuda available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('[ERROR] GPU_ENV has no usable CUDA PyTorch')
print('[preflight] GPU:', torch.cuda.get_device_name(0))
PY
  python -m py_compile \
    src/care_collision_cdf/scripts/collision_cdf_model.py \
    src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py
"

python3 -m py_compile   src/care_collision_cdf/scripts/multi_step_forbidden_space_cdf_gpu_shadow_node.py

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

echo "[C5.2g] starting persistent CUDA worker..."
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

echo "[C5.2g] GPU worker ready"

# Start the ROS shadow only after the C4.9 runner brings up the ROS master.
setsid bash -lc "
  set -e
  cd '${REPO}'
  source devel/setup.bash
  while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done
  exec rosrun care_collision_cdf multi_step_forbidden_space_cdf_gpu_shadow_node.py \
    _anchor_topic:='${ANCHOR_TOPIC}' \
    _map_topic:='${MAP_TOPIC}' \
    _summary_topic:='${SUMMARY_TOPIC}' \
    _output_jsonl:='${JSONL}' \
    _gpu_socket:='${GPU_SOCKET}' \
    _rate:='${SHADOW_RATE}' \
    _confidence_threshold:='${CONFIDENCE_THRESHOLD}' \
    _map_resolution:='${MAP_RESOLUTION}' \
    _proximity_margin:='${PROXIMITY_MARGIN}' \
    _max_pairs_per_step:='${MAX_PAIRS_PER_STEP}' \
    _signed_zero_band:='${SIGNED_ZERO_BAND}' \
    _max_pairs:=8000
" >"${ROOT_LOG}/gpu_online_shadow.log" 2>&1 &
PIDS+=("$!")

setsid bash -lc "
  set -e
  cd '${REPO}'
  source devel/setup.bash
  while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done
  while ! rostopic list 2>/dev/null | grep -q '^${SUMMARY_TOPIC}$'; do
    sleep 0.05
  done
  exec rostopic echo -p '${SUMMARY_TOPIC}'
" >"${SUMMARY_CSV}" 2>"${ROOT_LOG}/summary_recorder.err" &
PIDS+=("$!")

# Frozen C4.9 behavior. Only raw-MPC body-anchor shadow export is enabled.
TRAJECTORY_RISK_INPUT_TOPIC="/care_planner/mpc/predicted_trajectory" \
FORBIDDEN_SPACE_PAIR_PUBLISH_ENABLED="true" \
FORBIDDEN_SPACE_PAIR_TOPIC="${ANCHOR_TOPIC}" \
FORBIDDEN_SPACE_CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD}" \
CASE_ID="${CASE_ID}" \
RUN_SECONDS="${RUN_SECONDS}" \
OUT="${C4_OUT}" \
LOG="${C4_LOG}" \
ZIP_PATH="${ROOT_OUT}/C4_9_baseline_${CASE_ID}.zip" \
bash scripts/run_and_pack_phase_c4_9_blocker_aware_repair.sh

sleep 0.30
cleanup
trap - EXIT INT TERM

python3 - "${JSONL}" "${SUMMARY_JSON}" "${SHADOW_RATE}" <<'PY'
import json
import math
import os
import statistics
import sys

src, dst, target_rate = sys.argv[1:4]
target_rate = float(target_rate)

rows = []
if os.path.isfile(src):
    with open(src, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

if not rows:
    raise SystemExit("[ERROR] C5.2g produced no online GPU records")

def values(path):
    out = []
    for row in rows:
        value = row
        ok = True
        for key in path:
            if not isinstance(value, dict) or key not in value:
                ok = False
                break
            value = value[key]
        if ok and isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                out.append(value)
    return out

def stats(vals):
    if not vals:
        return None
    vals = sorted(vals)
    def q(frac):
        if len(vals) == 1:
            return vals[0]
        p = frac * (len(vals) - 1)
        lo = int(p)
        hi = min(lo + 1, len(vals) - 1)
        w = p - lo
        return vals[lo] * (1.0 - w) + vals[hi] * w
    return {
        "min": min(vals),
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "p95": q(0.95),
        "max": max(vals),
    }

wall = values(("wall_time",))
duration = max(wall) - min(wall) if len(wall) >= 2 else 0.0
effective_rate = (
    (len(wall) - 1) / duration
    if duration > 1e-9 and len(wall) >= 2 else 0.0
)

pipeline = stats(values(("timing_ms", "online_pipeline_ms")))
selection = stats(values(("timing_ms", "pair_selection_ms")))
anchor_decode = stats(values(("timing_ms", "anchor_decode_ms")))
pack = stats(values(("timing_ms", "request_pack_ms")))
ipc = stats(values(("timing_ms", "ipc_roundtrip_ms")))
h2d = stats(values(("timing_ms", "worker_h2d_ms")))
infer = stats(values(("timing_ms", "worker_inference_ms")))
d2h = stats(values(("timing_ms", "worker_d2h_ms")))
worker_total = stats(values(("timing_ms", "worker_total_ms")))
map_index = stats(values(("timing_ms", "map_index_build_ms")))
pairs = stats(values(("retained_pair_count",)))
raw_pairs = stats(values(("raw_local_pair_count",)))

budget_ms = 1000.0 / target_rate

summary = {
    "phase": "C5.2g",
    "name": "GPU Online End-to-End Forbidden-Space Shadow Diagnostic",
    "shadow_only": True,
    "constraint_enforced": False,
    "target_rate_hz": target_rate,
    "cycle_budget_ms": budget_ms,
    "records": len(rows),
    "observed_duration_s": duration,
    "effective_processed_rate_hz": effective_rate,
    "retained_pair_count": pairs,
    "raw_local_pair_count": raw_pairs,
    "anchor_decode_ms": anchor_decode,
    "pair_selection_ms": selection,
    "request_pack_ms": pack,
    "ipc_roundtrip_ms": ipc,
    "gpu_h2d_ms": h2d,
    "gpu_inference_ms": infer,
    "gpu_d2h_ms": d2h,
    "gpu_worker_total_ms": worker_total,
    "async_map_index_build_ms": map_index,
    "online_pipeline_ms": pipeline,
    "records_with_negative_cdf": sum(
        int(r.get("signed_counts", {}).get("negative", 0)) > 0
        for r in rows
    ),
    "meets_20hz_pipeline_budget_p95": (
        bool(pipeline and pipeline["p95"] < budget_ms)
    ),
    "effective_rate_fraction_of_target": (
        effective_rate / target_rate if target_rate > 0 else None
    ),
}

with open(dst, "w") as f:
    json.dump(summary, f, indent=2)

print("")
print("========== C5.2g GPU ONLINE SUMMARY ==========")
print(json.dumps(summary, indent=2))
print("==============================================")
PY

cat > "${ROOT_OUT}/c5_2g_run_metadata.txt" <<EOF
branch=$(git branch --show-current)
head=$(git rev-parse HEAD)
case_id=${CASE_ID}
run_seconds=${RUN_SECONDS}
shadow_only=true
constraint_enforced=false
gpu_env=${GPU_ENV}
gpu_device=${GPU_DEVICE}
checkpoint=${CHECKPOINT}
gpu_socket=${GPU_SOCKET}
target_rate_hz=${SHADOW_RATE}
confidence_threshold=${CONFIDENCE_THRESHOLD}
map_resolution=${MAP_RESOLUTION}
proximity_margin=${PROXIMITY_MARGIN}
max_pairs_per_step=${MAX_PAIRS_PER_STEP}
EOF

cd "${REPO}"
zip -r "${ZIP_PATH}" "${ROOT_OUT#${REPO}/}" "${ROOT_LOG#${REPO}/}"

echo ""
echo "[C5.2g COMPLETE]"
echo "[SUMMARY] ${SUMMARY_JSON}"
echo "[JSONL]   ${JSONL}"
echo "[ZIP]     ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"
