#!/usr/bin/env bash
set -euo pipefail

# Static replay of the exact measured configuration from the historical
# phase_e_goal_015 false-self-hit HARD_HOLD run.  No planner is launched.
# Purpose: distinguish dynamic timestamp skew from a static render/frame issue.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
TOF_CONFIG="${TOF_CONFIG:-${REPO}/src/care_confidence_map/config/tof_fusion_self_filter_hotspot_diag.yaml}"
DURATION="${DURATION:-25}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-true}"

# Measured stationary q from the old failing phase_e_goal_015 trial after
# HARD_HOLD (joint_states around t=9.685 s and later).
Q1="${Q1:--0.010447779055979822}"
Q2="${Q2:--0.284581330813765}"
Q3="${Q3:-0.0781421766217969}"
Q4="${Q4:-1.43162293039126}"
Q5="${Q5:-0.30118971223300584}"
Q6="${Q6:--0.9285468923388054}"
Q7="${Q7:--0.3227665346687294}"

cd "${REPO}"
source /opt/ros/noetic/setup.bash
[[ -f devel/setup.bash ]] && source devel/setup.bash

STAMP="$(date +%Y%m%d-%H%M%S)"
SHORT="$(git rev-parse --short=8 HEAD)"
ROOT="${REPO}/outputs/phase_e_static_self_hit_repro/${STAMP}_${SHORT}"
ZIP="${REPO}/CAREPlanner_PHASE_E_STATIC_SELF_HIT_REPRO_${STAMP}_${SHORT}.zip"
mkdir -p "${ROOT}"

GAZEBO_PID=""
FILTER_PID=""
DIAG_PID=""
REC_PIDS=()

kill_group() {
  local pid="${1:-}"
  [[ -z "${pid}" ]] && return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || true
    sleep 0.25
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 0.25
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  for p in "${REC_PIDS[@]:-}"; do kill_group "${p}"; done
  kill_group "${DIAG_PID}"
  kill_group "${FILTER_PID}"
  kill_group "${GAZEBO_PID}"
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  return "${rc}"
}
trap cleanup EXIT INT TERM
cleanup
trap cleanup EXIT INT TERM

cat > "${ROOT}/metadata.txt" <<EOF
diagnostic=phase_e_static_self_hit_reproduction
git_head=$(git rev-parse HEAD)
world_file=${WORLD_FILE}
tof_config=${TOF_CONFIG}
duration_s=${DURATION}
planner_enabled=false
initial_q=[${Q1},${Q2},${Q3},${Q4},${Q5},${Q6},${Q7}]
source=historical_phase_e_goal_015_failed_trial_stationary_post_hard_hold
EOF

echo "================================================================"
echo "PHASE-E STATIC SELF-HIT REPRODUCTION"
echo "planner : OFF"
echo "world   : ${WORLD_FILE}"
echo "sensor  : link4_sensor2"
echo "q       : [${Q1},${Q2},${Q3},${Q4},${Q5},${Q6},${Q7}]"
echo "purpose : remove robot motion from the timestamp hypothesis"
echo "================================================================"

setsid roslaunch arm_description gazebo_velocity_control.launch \
  world_file:="${WORLD_FILE}" \
  gazebo_gui:="${GAZEBO_GUI}" \
  use_rviz:="${USE_RVIZ}" \
  > "${ROOT}/gazebo.log" 2>&1 &
GAZEBO_PID=$!

echo "[WAIT] ROS master + joint states"
for _ in $(seq 1 300); do
  if rostopic echo -n 1 /care_arm/joint_states >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done
timeout 10 rostopic echo -n 1 /care_arm/joint_states >/dev/null

echo "[STATIC SETUP] stop the velocity controller before setting q"
timeout 10 rosservice call /care_arm/controller_manager/switch_controller \
  "{start_controllers: [], stop_controllers: ['arm_group_velocity_controller'], strictness: 2, start_asap: false, timeout: 0.0}" \
  > "${ROOT}/stop_velocity_controller.txt"

echo "[STATIC SETUP] set exact Gazebo joint configuration after controller startup"
timeout 10 rosservice call /gazebo/set_model_configuration \
  "{model_name: 'care_arm', urdf_param_name: 'robot_description', joint_names: ['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3'], joint_positions: [${Q1},${Q2},${Q3},${Q4},${Q5},${Q6},${Q7}]}" \
  > "${ROOT}/set_model_configuration.txt"

sleep 0.5

echo "[VERIFY] static replay q"
python3 - "${Q1}" "${Q2}" "${Q3}" "${Q4}" "${Q5}" "${Q6}" "${Q7}" <<'PY'
import sys, rospy
from sensor_msgs.msg import JointState
target=[float(x) for x in sys.argv[1:]]
names=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
rospy.init_node('static_self_hit_q_check', anonymous=True, disable_signals=True)
msg=rospy.wait_for_message('/care_arm/joint_states', JointState, timeout=10.0)
idx={n:i for i,n in enumerate(msg.name)}
q=[float(msg.position[idx[n]]) for n in names]
v=[float(msg.velocity[idx[n]]) if idx[n] < len(msg.velocity) else float('nan') for n in names]
err=max(abs(a-b) for a,b in zip(q,target))
print('[STATIC Q] measured=',q)
print('[STATIC Q] target  =',target)
print('[STATIC Q] err_inf = %.6f rad' % err)
print('[STATIC Q] velocity=',v)
if err > 0.02:
    raise SystemExit('static replay q mismatch: refusing to continue invalid test')
PY

setsid roslaunch care_confidence_map tof_fusion_self_filter.launch \
  config_file:="${TOF_CONFIG}" \
  > "${ROOT}/tof_filter.log" 2>&1 &
FILTER_PID=$!

echo "[WAIT] filtered ToF node"
for _ in $(seq 1 120); do
  if rosnode list 2>/dev/null | grep -q '^/tof_fusion_self_filter$'; then break; fi
  sleep 0.1
done
timeout 20 rostopic echo -n 1 /link4_sensor2/tof/cloud >/dev/null

echo "[SNAPSHOT] Gazebo actual link/render poses vs ROS TF"
python3 src/care_confidence_map/scripts/capture_static_gazebo_tf_snapshot.py \
  --output "${ROOT}/static_gazebo_tf_snapshot.json" \
  --base-frame base_link \
  --model-name care_arm \
  --raw-topic /link4_sensor2/tof/cloud \
  --samples 8 \
  --period 0.20 \
  --max-clouds 8 \
  > "${ROOT}/static_gazebo_tf_snapshot.log" 2>&1
cat "${ROOT}/static_gazebo_tf_snapshot.log"

setsid rosrun care_confidence_map runtime_self_hit_rviz_diagnostic.py \
  _base_frame:=base_link \
  _raw_topic:=/link4_sensor2/tof/cloud \
  _sensor_frame:=link4_sensor2_tof_link \
  _sensor_id:=5 \
  _sensor_name:=link4_sensor2 \
  _self_filter_urdf:="${REPO}/src/arm_description/urdf/Arm_with_self_filter_collision.urdf" \
  _reference_urdf:="${REPO}/src/arm_description/urdf/Arm.urdf" \
  _gazebo_link_states_topic:=/gazebo/link_states \
  _gazebo_model_name:=care_arm \
  _marker_topic:=/care_planner/debug/markers \
  _pixel_stride:=8 \
  _near_clip:=0.15 \
  _far_clip:=0.75 \
  _horizontal_fov_deg:=55.0 \
  _vertical_fov_deg:=72.0 \
  _hotspot_x_min:=-0.025 \
  _hotspot_x_max:=0.075 \
  _hotspot_y_min:=-0.025 \
  _hotspot_y_max:=0.125 \
  _hotspot_z_min:=0.20 \
  _hotspot_z_max:=0.425 \
  > "${ROOT}/runtime_diag.log" 2>&1 &
DIAG_PID=$!

# Capture the exact URDF->SDF conversion used by Gazebo Classic when possible.
# This lets us inspect whether converted link / visual / sensor frames carry
# compensating offsets that are invisible in the URDF.
if command -v gz >/dev/null 2>&1; then
  gz sdf -p "${REPO}/src/arm_description/urdf/Arm.urdf" \
    > "${ROOT}/Arm.generated.sdf" \
    2> "${ROOT}/gz_sdf.stderr" || true
fi

# Record measured q and ToF performance while the arm is deliberately static.
setsid rostopic echo -p /care_arm/joint_states > "${ROOT}/joint_states.csv" 2>/dev/null &
REC_PIDS+=($!)
setsid rostopic echo -p /care_planner/perception/tof_fusion_summary > "${ROOT}/tof_fusion_summary.csv" 2>/dev/null &
REC_PIDS+=($!)

echo ""
echo "[RUN] static diagnostic for ${DURATION}s."
echo "[RVIZ] cyan=primitives, purple=Gazebo actual visuals, green=self, red=suspect."
echo "[RVIZ] If red points appear while qdot~=0, ordinary motion/TF delay is ruled out."
sleep "${DURATION}"

# Extract the evidence before cleanup.
grep -F "[TOF_HOTSPOT_HIT]" "${ROOT}/tof_filter.log" > "${ROOT}/tof_hotspot_hits.log" || true
grep -F "[RUNTIME_SELF_HIT_DIAG]" "${ROOT}/runtime_diag.log" > "${ROOT}/runtime_self_hit_diag.log" || true

python3 - "${ROOT}" <<'PY'
import os,re,sys,math,csv
root=sys.argv[1]
p=os.path.join(root,'tof_hotspot_hits.log')
n0=n1=0
max_out=None
if os.path.isfile(p):
    for line in open(p,errors='replace'):
        if 'is_self=0' in line:
            n0 += 1
            m=re.search(r'signed_surface_distance=([-+0-9.eE]+)',line)
            if m:
                d=float(m.group(1))
                max_out=d if max_out is None else max(max_out,d)
        elif 'is_self=1' in line:
            n1 += 1

vmax=0.0
jp=os.path.join(root,'joint_states.csv')
if os.path.isfile(jp):
    try:
        rows=list(csv.DictReader(open(jp)))
        for r in rows:
            vs=[]
            for i in range(7):
                try: vs.append(float(r['field.velocity%d'%i]))
                except Exception: pass
            if vs: vmax=max(vmax,math.sqrt(sum(x*x for x in vs)))
    except Exception:
        pass

with open(os.path.join(root,'summary.txt'),'w') as f:
    f.write('hotspot_is_self_0=%d\n' % n0)
    f.write('hotspot_is_self_1=%d\n' % n1)
    f.write('max_outside_mm=%s\n' % (
        'none' if max_out is None else ('%.3f' % (1000.0*max_out))))
    f.write('max_joint_velocity_norm_rad_s=%.9f\n' % vmax)

print('[STATIC RESULT] is_self=0:',n0,'is_self=1:',n1,
      'max_outside_mm:',None if max_out is None else 1000.0*max_out,
      'max_qdot_norm:',vmax)
PY

rm -f "${ZIP}"
python3 - "${ROOT}" "${ZIP}" <<'PY'
import os,sys,zipfile
root,dst=sys.argv[1:]
with zipfile.ZipFile(dst,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for base,_,files in os.walk(root):
        for name in files:
            p=os.path.join(base,name)
            z.write(p,os.path.relpath(p,root))
print(dst)
PY

echo ""
cat "${ROOT}/summary.txt"
echo "[POSE SNAPSHOT]"
grep -E "dxyz_mm|cloud stamp" "${ROOT}/static_gazebo_tf_snapshot.log" | tail -n 20 || true
echo "[UPLOAD] ${ZIP}"
ls -lh "${ZIP}"
