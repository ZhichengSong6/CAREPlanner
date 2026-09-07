#!/usr/bin/env bash
set -euo pipefail

# Clean Phase-E self-filter validation:
#   1) hotspot provenance logging OFF;
#   2) exact dedicated-URDF self filter, no runtime padding;
#   3) lean recorder profile to avoid measurement perturbation;
#   4) repeat the three historical ghost-HARD_HOLD cases;
#   5) package only the core perception/safety summaries.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_FILE="${CASE_FILE:-${REPO}/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/confidence_map_phase_e_ray.yaml}"
TOF_FUSION_CONFIG_FILE="${TOF_FUSION_CONFIG_FILE:-${REPO}/src/care_confidence_map/config/tof_fusion_self_filter.yaml}"

RUN_SECONDS="${RUN_SECONDS:-20}"
TRIALS_PER_CASE="${TRIALS_PER_CASE:-3}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
KEEP_RAW_LOGS="${KEEP_RAW_LOGS:-0}"

CASES=(phase_e_goal_004 phase_e_goal_015 phase_e_goal_022)

cd "${REPO}"

STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
GIT_SHORT="$(git rev-parse --short=8 HEAD)"
DIAG_ID="phase_e_self_filter_clean_${STAMP}_${GIT_SHORT}"
ROOT="${REPO}/outputs/phase_e_self_filter_clean/${DIAG_ID}"
ART="${ROOT}/artifacts"
ZIP="${REPO}/CAREPlanner_PHASE_E_SELF_FILTER_CLEAN_${DIAG_ID}.zip"

rm -rf "${ROOT}"
rm -f "${ZIP}"
mkdir -p "${ART}"

cleanup_ros() {
  local n
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  sleep 0.5
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${n}" 2>/dev/null || true
  done
  rm -f /tmp/care_collision_cdf_gpu_c5_5.sock         /tmp/care_collision_cdf_gpu_c5_4.sock 2>/dev/null || true
}

cat > "${ROOT}/metadata.txt" <<EOF
diagnostic=phase_e_self_filter_clean_smoke
git_head=$(git rev-parse HEAD)
world_file=${WORLD_FILE}
confidence_map_config=${CONFIDENCE_MAP_CONFIG_FILE}
tof_fusion_config=${TOF_FUSION_CONFIG_FILE}
recording_profile=self_filter_perf
hotspot_logging=false
self_filter_model=dedicated_urdf_primitives
runtime_padding_m=0
run_seconds=${RUN_SECONDS}
trials_per_case=${TRIALS_PER_CASE}
cases=${CASES[*]}
EOF

echo "================================================================"
echo "PHASE-E SELF-FILTER CLEAN SMOKE"
echo "hotspot logging      : OFF"
echo "recording profile    : self_filter_perf"
echo "trials per case      : ${TRIALS_PER_CASE}"
echo "run seconds / trial  : ${RUN_SECONDS}"
echo "================================================================"

for CASE_ID in "${CASES[@]}"; do
  for TRIAL in $(seq 1 "${TRIALS_PER_CASE}"); do
    cleanup_ros

    TRIAL_TAG=$(printf "trial_%02d" "${TRIAL}")
    RUN_ID="${CASE_ID}_self_filter_clean_${TRIAL_TAG}_${STAMP}_${GIT_SHORT}"
    RUN_ROOT="${REPO}/outputs/c5_5_vbc_gcdf_regime/${RUN_ID}"
    RUN_LOG_ROOT="${REPO}/logs/c5_5_vbc_gcdf_regime/${RUN_ID}"
    CASE_ZIP="${REPO}/CAREPlanner_C5_RESULT_${RUN_ID}.zip"
    DST="${ART}/${CASE_ID}/${TRIAL_TAG}"
    mkdir -p "${DST}"

    echo ""
    echo "[RUN] case=${CASE_ID} trial=${TRIAL}/${TRIALS_PER_CASE}"

    set +e
    CASE_FILE="${CASE_FILE}"     CASE_ID="${CASE_ID}"     RUN_ID="${RUN_ID}"     RUN_SECONDS="${RUN_SECONDS}"     WORLD_FILE="${WORLD_FILE}"     CONFIDENCE_MAP_CONFIG_FILE="${CONFIDENCE_MAP_CONFIG_FILE}"     TOF_FUSION_CONFIG_FILE="${TOF_FUSION_CONFIG_FILE}"     TOF_FUSION_ENABLED=true     EXECUTION_GCDF_AUDIT_ENABLED=true     GCDF_BODY_INFLATION_M=0.015     FORCE_ZERO_INITIAL_Q=true     EARLY_STOP_ON_GOAL=false     GAZEBO_GUI="${GAZEBO_GUI}"     USE_RVIZ="${USE_RVIZ}"     RECORDING_PROFILE=self_filter_perf       bash scripts/run_and_pack_phase_e5_execution_gcdf.sh
    RC=$?
    set -e

    echo "${RC}" > "${DST}/runner_rc.txt"

    if [[ -d "${RUN_ROOT}/run" ]]; then
      for name in         tof_fusion_summary.csv         e3_summary.csv         execution_gcdf_safety_summary.csv         execution_gcdf_hard_hold.csv         joint_states.csv; do
        [[ -f "${RUN_ROOT}/run/${name}" ]] &&           cp -f "${RUN_ROOT}/run/${name}" "${DST}/${name}"
      done
    fi

    # Do not retain the full per-case packed result; this wrapper produces one
    # compact diagnostic archive.  Raw process logs are removed after a clean
    # runner exit unless explicitly requested.
    rm -f "${CASE_ZIP}"
    if [[ "${KEEP_RAW_LOGS}" != "1" && "${RC}" -eq 0 ]]; then
      rm -rf "${RUN_LOG_ROOT}"
    fi
  done
done

cleanup_ros

python3 - "${ART}" "${ROOT}/self_filter_clean_summary.csv" "${ROOT}/self_filter_clean_summary.txt" <<'PY'
import csv
import glob
import math
import os
import re
import statistics
import sys

art, csv_out, txt_out = sys.argv[1:]
TOK = re.compile(r'([A-Za-z0-9_]+)=([^\s]+)')

def fnum(x):
    try:
        return float(x)
    except Exception:
        return math.nan

def parse_summary(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, errors="replace") as f:
        f.readline()
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw or "," not in raw:
                continue
            ts_s, data = raw.split(",", 1)
            try:
                # rostopic echo -p emits ROS-time nanoseconds.
                ts_ros_s = float(ts_s) / 1e9
            except Exception:
                ts_ros_s = math.nan
            rows.append((ts_ros_s, dict(TOK.findall(data))))
    return rows

def vals(rows, key):
    out = []
    steady = rows[10:] if len(rows) > 20 else rows
    for _, r in steady:
        v = fnum(r.get(key))
        if math.isfinite(v):
            out.append(v)
    return out

def sim_topic_hz(rows):
    steady = rows[10:] if len(rows) > 20 else rows
    ts = [t for t, _ in steady if math.isfinite(t)]
    if len(ts) < 2 or ts[-1] <= ts[0]:
        return math.nan
    return (len(ts) - 1) / (ts[-1] - ts[0])

records = []
for trial_dir in sorted(glob.glob(os.path.join(art, "*", "trial_*"))):
    case_id = os.path.basename(os.path.dirname(trial_dir))
    trial = os.path.basename(trial_dir)

    tof = parse_summary(os.path.join(trial_dir, "tof_fusion_summary.csv"))
    safety = parse_summary(
        os.path.join(trial_dir, "execution_gcdf_safety_summary.csv"))

    hz = vals(tof, "publish_hz")
    pub_ms = vals(tof, "publish_callback_ms")
    cb_ms = vals(tof, "cloud_callback_ms_mean")
    sf_ms = vals(tof, "self_filter_ms_mean")
    tf_ms = vals(tof, "body_tf_lookup_ms_mean")

    hard_events = 0
    occupied_hard = 0
    for _, r in safety:
        try:
            hard_events = max(hard_events, int(float(r.get("hard_event_count", 0))))
        except Exception:
            pass
        if r.get("state") == "HARD_HOLD" and r.get("min_source") == "occupied":
            occupied_hard = 1

    last = tof[-1][1] if tof else {}
    rec = {
        "case_id": case_id,
        "trial": trial,
        "tof_samples": len(tof),
        "sim_topic_hz": sim_topic_hz(tof),
        "median_wall_publish_hz": statistics.median(hz) if hz else math.nan,
        "p10_wall_publish_hz": (
            sorted(hz)[max(0, int(0.10*(len(hz)-1)))] if hz else math.nan),
        "median_publish_callback_ms": (
            statistics.median(pub_ms) if pub_ms else math.nan),
        "median_cloud_callback_ms": (
            statistics.median(cb_ms) if cb_ms else math.nan),
        "median_self_filter_ms": (
            statistics.median(sf_ms) if sf_ms else math.nan),
        "median_body_tf_ms": (
            statistics.median(tf_ms) if tf_ms else math.nan),
        "body_tf_failures": int(fnum(last.get("body_tf_failures", 0)) or 0),
        "dropped_for_body_tf": int(
            fnum(last.get("dropped_for_body_tf", 0)) or 0),
        "hard_event_count": hard_events,
        "occupied_hard_hold": occupied_hard,
    }
    rec["wall_15hz_pass"] = int(
        math.isfinite(rec["median_wall_publish_hz"]) and
        rec["median_wall_publish_hz"] >= 14.5)
    records.append(rec)

fields = [
    "case_id", "trial", "tof_samples",
    "sim_topic_hz", "median_wall_publish_hz", "p10_wall_publish_hz",
    "median_publish_callback_ms", "median_cloud_callback_ms",
    "median_self_filter_ms", "median_body_tf_ms",
    "body_tf_failures", "dropped_for_body_tf",
    "hard_event_count", "occupied_hard_hold", "wall_15hz_pass",
]
with open(csv_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(records)

with open(txt_out, "w") as f:
    f.write("Phase-E exact-URDF self-filter clean smoke\n")
    f.write("Hotspot logging OFF; lean recorder profile.\n")
    f.write("Wall-clock 15 Hz pass threshold: median >= 14.5 Hz.\n\n")
    for r in records:
        f.write(
            "{case_id} {trial}: sim_hz={sim_topic_hz:.3f}, "
            "wall_hz={median_wall_publish_hz:.3f}, "
            "p10_hz={p10_wall_publish_hz:.3f}, "
            "self_filter={median_self_filter_ms:.3f} ms, "
            "cloud_cb={median_cloud_callback_ms:.3f} ms, "
            "body_tf={median_body_tf_ms:.3f} ms, "
            "drops={dropped_for_body_tf}, "
            "occupied_HARD_HOLD={occupied_hard_hold}, "
            "15Hz_pass={wall_15hz_pass}\n".format(**r)
        )

print(txt_out)
PY

python3 - "${ROOT}" "${ZIP}" <<'PY'
import os
import sys
import zipfile

root, dst = sys.argv[1:]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for base, _, files in os.walk(root):
        for name in files:
            p = os.path.join(base, name)
            z.write(p, os.path.relpath(p, root))
print(dst)
PY

echo ""
echo "================ CLEAN SELF-FILTER SMOKE COMPLETE ================"
cat "${ROOT}/self_filter_clean_summary.txt"
echo "[UPLOAD ZIP] ${ZIP}"
ls -lh "${ZIP}"
