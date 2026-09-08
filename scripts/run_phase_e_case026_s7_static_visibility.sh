#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"

cd "${REPO}"
source /opt/ros/noetic/setup.bash
source "${REPO}/devel/setup.bash"

STAMP="$(date +%Y%m%d-%H%M%S)"
SHORT="$(git rev-parse --short=8 HEAD)"
ROOT="${REPO}/outputs/phase_e_case026_s7_static_visibility/${STAMP}_${SHORT}"
mkdir -p "${ROOT}"
RESULT_JSON="${ROOT}/s7_static_visibility.json"
POSE_JSON="${ROOT}/s7_gazebo_tf_snapshot.json"
ZIP="${REPO}/CAREPlanner_PHASE_E_CASE026_S7_STATIC_VISIBILITY_${STAMP}_${SHORT}.zip"

cleanup() {
  set +e
  timeout 1 rostopic pub -1     /care_arm/arm_group_velocity_controller/command     std_msgs/Float64MultiArray     "data: [0,0,0,0,0,0,0]" >/dev/null 2>&1 || true
  [[ -n "${LAUNCH_PID:-}" ]] && kill -TERM "${LAUNCH_PID}" 2>/dev/null || true
  sleep 0.5
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

cleanup
trap cleanup EXIT INT TERM

echo "=============================================================="
echo "CASE 026 S7 STATIC REAL-VISIBILITY TEST"
echo "planner/GCDF/VBC : OFF"
echo "world            : ${WORLD_FILE}"
echo "target           : [0.10, 0.05, 0.15]"
echo "pose             : fixed robust S7 candidate"
echo "movement         : slow joint velocity servo from zero"
echo "=============================================================="

roslaunch care_confidence_map phase_e_mapping_smoke.launch   world_file:="${WORLD_FILE}"   gazebo_gui:="${GAZEBO_GUI}"   use_rviz:="${USE_RVIZ}"   paused:=false   > "${ROOT}/roslaunch.log" 2>&1 &
LAUNCH_PID=$!

echo "[WAIT] joint states"
timeout 30 rostopic echo -n 1 /care_arm/joint_states >/dev/null

echo "[WAIT] confidence-map service"
timeout 30 bash -c '
  until rosservice info /care_planner/confidence_map/query >/dev/null 2>&1; do
    sleep 0.2
  done
'

echo "[WAIT] fused ToF"
timeout 30 rostopic echo -n 1   /care_planner/perception/tof_fused_filtered >/dev/null

echo "[MOVE + QUERY] moving to S7, then testing fixed target"
python3 scripts/test_phase_e_case026_s7_static_visibility.py   --output-json "${RESULT_JSON}"   2>&1 | tee "${ROOT}/s7_static_visibility.log"

echo "[POSE AUDIT] final Gazebo-vs-TF consistency at S7"
python3 src/care_confidence_map/scripts/capture_static_gazebo_tf_snapshot.py   --output "${POSE_JSON}"   --base-frame base_link   --model-name care_arm   --raw-topic /link4_sensor2/tof/cloud   --samples 3   --period 0.10   --max-clouds 1   > "${ROOT}/s7_pose_audit.log" 2>&1

python3 - "${POSE_JSON}" <<'PY'
import json, statistics, sys
d=json.load(open(sys.argv[1]))
watch=['link2','link4','link4_sensor2_tof_gz_link']
bad=[]
for name in watch:
    vals=[]
    for s in d.get('pose_samples',[]):
        v=s.get('links',{}).get(name,{})
        if 'delta_norm_m' in v:
            vals.append(1000.0*float(v['delta_norm_m']))
    med=statistics.median(vals) if vals else float('inf')
    print('[S7 POSE] {} median_delta_mm={:.3f}'.format(name,med))
    if med > 5.0:
        bad.append((name,med))
if bad:
    print('[WARN] Gazebo/TF mismatch at S7: {}'.format(bad))
PY

rm -f "${ZIP}"
python3 - "${ROOT}" "${ZIP}" <<'PY'
import os,sys,zipfile
root,dst=sys.argv[1:]
with zipfile.ZipFile(dst,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for base,dirs,files in os.walk(root):
        dirs.sort(); files.sort()
        for name in files:
            p=os.path.join(base,name)
            z.write(p,os.path.relpath(p,root))
print(dst)
PY

echo ""
cat "${RESULT_JSON}"
echo ""
echo "[UPLOAD] ${ZIP}"
ls -lh "${ZIP}"
