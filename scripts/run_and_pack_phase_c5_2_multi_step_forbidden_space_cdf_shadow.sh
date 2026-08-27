#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_003}"
RUN_SECONDS="${RUN_SECONDS:-15.0}"

CDF_ENV="${CDF_ENV:-ncdf_l4c}"
CDF_DEVICE="${CDF_DEVICE:-cpu}"
CHECKPOINT="${CHECKPOINT:-${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt}"

SHADOW_RATE="${SHADOW_RATE:-5.0}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.50}"
MAP_RESOLUTION="${MAP_RESOLUTION:-0.05}"
PROXIMITY_MARGIN="${PROXIMITY_MARGIN:-0.075}"
MAX_PAIRS_PER_STEP="${MAX_PAIRS_PER_STEP:-250}"
SIGNED_ZERO_BAND="${SIGNED_ZERO_BAND:-0.05}"
PROXY_SIGN_MARGIN_M="${PROXY_SIGN_MARGIN_M:-0.01}"

ANCHOR_TOPIC="/care_planner/trajectory_risk/body_sweep_anchors"
MAP_TOPIC="/care_planner/confidence_map/points"
SUMMARY_TOPIC="/care_planner/collision_cdf/multi_step_forbidden_space_summary"

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
echo "[C5.2] signed checkpoint: ${CHECKPOINT}"
echo "[C5.2] runtime: ${CDF_ENV} / ${CDF_DEVICE}"
echo "[C5.2] confidence threshold: ${CONFIDENCE_THRESHOLD}"
echo "[C5.2] proximity margin: ${PROXIMITY_MARGIN} m"
echo "[C5.2] max pairs per horizon step: ${MAX_PAIRS_PER_STEP}"
echo "[C5.2] mode: SHADOW ONLY (no MPC constraint enforcement)"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] signed collision CDF checkpoint not found:"
  echo "  ${CHECKPOINT}"
  echo "Expected current file:"
  echo "  src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt"
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

# Start the signed learned CDF once ROS exists.  The learned model stays in the
# same isolated conda-process pattern already used by the visibility CDF.
setsid bash -lc "
  set -e
  cd '${REPO}'
  source devel/setup.bash
  while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done

  rosparam load src/care_collision_cdf/config/collision_cdf.yaml /collision_cdf_server
  rosparam set /collision_cdf_server/checkpoint '${CHECKPOINT}'
  rosparam set /collision_cdf_server/device '${CDF_DEVICE}'
  rosparam set /collision_cdf_server/checkpoint_key latest
  rosparam set /collision_cdf_server/collision_cdf/activation gelu
  rosparam set /collision_cdf_server/collision_cdf/signed true

  source '${CONDA_SH}'
  conda activate '${CDF_ENV}'
  cd '${REPO}'
  source devel/setup.bash
  exec python -u src/care_collision_cdf/scripts/collision_cdf_server_node.py
" >"${ROOT_LOG}/collision_cdf_server.log" 2>&1 &
EXTRA_PIDS+=("$!")

# The shadow node uses body samples only as spatial anchors.  Actual CDF points
# are low-confidence voxel centers read from the complete confidence-map cloud.
setsid bash -lc "
  set -e
  cd '${REPO}'
  source devel/setup.bash
  while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done
  while ! rosservice list 2>/dev/null | grep -q '^/care_planner/collision_cdf/query_pairs$'; do
    sleep 0.05
  done
  exec rosrun care_collision_cdf multi_step_forbidden_space_cdf_shadow_node.py \
    _anchor_topic:='${ANCHOR_TOPIC}' \
    _map_topic:='${MAP_TOPIC}' \
    _pair_service:=/care_planner/collision_cdf/query_pairs \
    _summary_topic:='${SUMMARY_TOPIC}' \
    _output_jsonl:='${SHADOW_JSONL}' \
    _rate:='${SHADOW_RATE}' \
    _confidence_threshold:='${CONFIDENCE_THRESHOLD}' \
    _map_resolution:='${MAP_RESOLUTION}' \
    _proximity_margin:='${PROXIMITY_MARGIN}' \
    _max_pairs_per_step:='${MAX_PAIRS_PER_STEP}' \
    _signed_zero_band:='${SIGNED_ZERO_BAND}' \
    _proxy_sign_margin_m:='${PROXY_SIGN_MARGIN_M}' \
    _max_pairs:=8000
" >"${ROOT_LOG}/multi_step_forbidden_space_shadow.log" 2>&1 &
EXTRA_PIDS+=("$!")

setsid bash -lc "
  set -e
  cd '${REPO}'
  source devel/setup.bash
  while ! timeout 1 rostopic list >/dev/null 2>&1; do sleep 0.05; done
  while ! rostopic list 2>/dev/null | grep -q '^${SUMMARY_TOPIC}$'; do
    sleep 0.05
  done
  exec rostopic echo -p '${SUMMARY_TOPIC}'
" >"${SHADOW_SUMMARY_CSV}" 2>"${ROOT_LOG}/shadow_summary_recorder.err" &
EXTRA_PIDS+=("$!")

# Reuse the frozen C4.9 experiment.  Only trajectory-risk shadow export changes:
# it evaluates the RAW MPC prediction and publishes all body-sweep anchors.
# Planner objectives, safe-prefix generation, exact VBC, commit/reject, and
# execution semantics remain unchanged.
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
cleanup_extra
trap - EXIT INT TERM

if ! grep -q "activation=gelu" "${ROOT_LOG}/collision_cdf_server.log"; then
  echo "[ERROR] signed CDF server did not confirm activation=gelu"
  tail -n 120 "${ROOT_LOG}/collision_cdf_server.log" || true
  exit 4
fi

python3 - "${SHADOW_JSONL}" "${SUMMARY_JSON}"   "${CONFIDENCE_THRESHOLD}" "${SIGNED_ZERO_BAND}"   "${PROXIMITY_MARGIN}" "${MAX_PAIRS_PER_STEP}" <<'PY'
import json
import math
import os
import statistics
import sys

(
    src,
    dst,
    confidence_threshold,
    signed_zero_band,
    proximity_margin,
    max_pairs_per_step,
) = sys.argv[1:7]

confidence_threshold = float(confidence_threshold)
signed_zero_band = float(signed_zero_band)
proximity_margin = float(proximity_margin)
max_pairs_per_step = int(max_pairs_per_step)

records = []
if os.path.isfile(src):
    with open(src, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass

nonempty = [
    r for r in records
    if int(r.get("retained_pair_count", r.get("deduplicated_pair_count", 0))) > 0
]

def stats(values):
    a = [
        float(v) for v in values
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    if not a:
        return None
    a.sort()
    def quantile(frac):
        if len(a) == 1:
            return a[0]
        p = frac * (len(a) - 1)
        lo = int(p)
        hi = min(lo + 1, len(a) - 1)
        w = p - lo
        return a[lo] * (1.0 - w) + a[hi] * w
    return {
        "min": min(a),
        "median": statistics.median(a),
        "mean": statistics.fmean(a),
        "p95": quantile(0.95),
        "max": max(a),
    }

def nested(rows, *keys):
    out = []
    for row in rows:
        value = row
        ok = True
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                ok = False
                break
            value = value[key]
        if ok and isinstance(value, (int, float)):
            out.append(float(value))
    return out

global_min_record = None
if nonempty:
    candidates = [
        r for r in nonempty
        if isinstance(r.get("distance"), dict)
        and isinstance(r["distance"].get("min"), (int, float))
    ]
    if candidates:
        global_min_record = min(
            candidates, key=lambda r: float(r["distance"]["min"])
        )

deepest_proxy_record = None
if nonempty:
    candidates = [
        r for r in nonempty
        if isinstance(r.get("approx_body_clearance_m"), dict)
        and isinstance(
            r["approx_body_clearance_m"].get("min"), (int, float)
        )
    ]
    if candidates:
        deepest_proxy_record = min(
            candidates,
            key=lambda r: float(r["approx_body_clearance_m"]["min"]),
        )

summary = {
    "phase": "C5.2",
    "name": "Multi-Step Forbidden-Space CDF Shadow Diagnostic",
    "shadow_only": True,
    "constraint_enforced": False,
    "cdf_variant": "signed",
    "cdf_activation": "gelu",
    "trajectory_source": "/care_planner/mpc/predicted_trajectory",
    "forbidden_point_source": (
        "confidence-map voxel centers with confidence below threshold, "
        "locally selected around predicted robot-body anchors"
    ),
    "confidence_threshold": confidence_threshold,
    "proximity_margin_m": proximity_margin,
    "max_pairs_per_step": max_pairs_per_step,
    "signed_zero_band": signed_zero_band,
    "records_total": len(records),
    "records_nonempty": len(nonempty),
    "anchor_count": stats([r.get("anchor_count") for r in nonempty]),
    "low_confidence_voxel_count": stats(
        [r.get("low_confidence_voxel_count") for r in nonempty]
    ),
    "raw_local_pair_count": stats(
        [r.get("raw_local_pair_count") for r in nonempty]
    ),
    "retained_pair_count": stats(
        [r.get("retained_pair_count") for r in nonempty]
    ),
    "active_step_count": stats(
        [r.get("active_step_count") for r in nonempty]
    ),
    "cdf_inference_ms": stats(
        [r.get("inference_ms") for r in nonempty]
    ),
    "service_roundtrip_ms": stats(
        [r.get("service_roundtrip_ms") for r in nonempty]
    ),
    "record_min_distance": stats(nested(nonempty, "distance", "min")),
    "record_p05_distance": stats(nested(nonempty, "distance", "p05")),
    "record_negative_rate": stats(
        nested(nonempty, "signed_counts", "negative_rate")
    ),
    "record_near_zero_rate": stats(
        nested(nonempty, "signed_counts", "near_zero_rate")
    ),
    "record_positive_rate": stats(
        nested(nonempty, "signed_counts", "positive_rate")
    ),
    "record_min_approx_body_clearance_m": stats(
        nested(nonempty, "approx_body_clearance_m", "min")
    ),
    "negative_given_proxy_inside_rate": stats(
        nested(
            nonempty,
            "proxy_sign",
            "negative_given_proxy_inside_rate",
        )
    ),
    "positive_given_proxy_outside_rate": stats(
        nested(
            nonempty,
            "proxy_sign",
            "positive_given_proxy_outside_rate",
        )
    ),
    "records_with_negative_cdf": sum(
        1
        for r in nonempty
        if isinstance(r.get("signed_counts"), dict)
        and int(r["signed_counts"].get("negative", 0)) > 0
    ),
    "records_with_proxy_overlap": sum(
        1
        for r in nonempty
        if isinstance(r.get("proxy_sign"), dict)
        and int(r["proxy_sign"].get("proxy_inside_count", 0)) > 0
    ),
    "global_min_pair": (
        global_min_record.get("global_min_pair")
        if global_min_record else None
    ),
    "deepest_proxy_overlap_pair": (
        deepest_proxy_record.get("deepest_proxy_overlap_pair")
        if deepest_proxy_record else None
    ),
}

with open(dst, "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))

if not records:
    raise SystemExit("[ERROR] C5.2 produced no shadow records")
if not nonempty:
    raise SystemExit(
        "[ERROR] C5.2 never observed a non-empty forbidden-space voxel batch"
    )
PY

cat > "${ROOT_OUT}/c5_2_run_metadata.txt" <<EOF
branch=$(git branch --show-current)
head=$(git rev-parse HEAD)
case_id=${CASE_ID}
run_seconds=${RUN_SECONDS}
name=Multi-Step Forbidden-Space CDF Shadow Diagnostic
shadow_only=true
constraint_enforced=false
cdf_variant=signed
cdf_activation=gelu
trajectory_source=/care_planner/mpc/predicted_trajectory
anchor_topic=${ANCHOR_TOPIC}
map_topic=${MAP_TOPIC}
confidence_threshold=${CONFIDENCE_THRESHOLD}
map_resolution=${MAP_RESOLUTION}
proximity_margin=${PROXIMITY_MARGIN}
max_pairs_per_step=${MAX_PAIRS_PER_STEP}
signed_zero_band=${SIGNED_ZERO_BAND}
proxy_sign_margin_m=${PROXY_SIGN_MARGIN_M}
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
