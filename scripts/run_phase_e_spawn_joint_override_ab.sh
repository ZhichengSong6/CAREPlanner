#!/usr/bin/env bash
set -euo pipefail

# Phase-E startup A/B:
#   A = historical spawn_model -J 0 ...  (SetModelConfiguration path)
#   B = natural URDF q=0 spawn, with NO -J arguments
#
# Each trial is planner-free and empty-world.  We verify q≈0, then query
# Gazebo actual link poses via /gazebo/get_link_state and compare them against
# ROS TF.  The raw link4_sensor2 depth signature is captured at the same time.
#
# Primary discriminator:
#   link2 / link4 / link4_sensor2_tof_gz_link Gazebo-vs-TF translation error.
#
# If A repeatedly shows ~37.7 / ~45 mm while B stays ~0, spawn_model's
# -J / SetModelConfiguration startup path is the root cause.

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
WORLD_FILE="${WORLD_FILE:-${REPO}/src/arm_description/worlds/maixsense_empty.world}"
ROUNDS="${ROUNDS:-5}"
GAZEBO_GUI="${GAZEBO_GUI:-false}"
USE_RVIZ="${USE_RVIZ:-false}"
Q_TOL="${Q_TOL:-0.01}"

cd "${REPO}"
source /opt/ros/noetic/setup.bash
[[ -f devel/setup.bash ]] && source devel/setup.bash

STAMP="$(date +%Y%m%d-%H%M%S)"
SHORT="$(git rev-parse --short=8 HEAD)"
ROOT="${REPO}/outputs/phase_e_spawn_joint_override_ab/${STAMP}_${SHORT}"
ZIP="${REPO}/CAREPlanner_PHASE_E_SPAWN_JOINT_OVERRIDE_AB_${STAMP}_${SHORT}.zip"
mkdir -p "${ROOT}"

GAZEBO_PID=""

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

cleanup_trial() {
  set +e
  kill_group "${GAZEBO_PID}"
  GAZEBO_PID=""
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${n}" 2>/dev/null || true
  done
  sleep 0.4
  for n in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -KILL -x "${n}" 2>/dev/null || true
  done
  set -e
}
trap cleanup_trial EXIT INT TERM

cat > "${ROOT}/metadata.txt" <<EOF
diagnostic=phase_e_spawn_joint_override_ab
git_head=$(git rev-parse HEAD)
world_file=${WORLD_FILE}
rounds=${ROUNDS}
variant_A=apply_initial_joint_overrides:true__spawn_model_-J_zero
variant_B=apply_initial_joint_overrides:false__natural_urdf_q0_no_-J
planner_enabled=false
EOF

echo -e "round\tvariant\tapply_initial_joint_overrides\tq_err_inf_rad\tlink1_mm\tlink2_mm\tlink3_mm\tlink4_mm\trender_mm\trender_deg\traw_hash0\tfinite0\tclassification"   > "${ROOT}/summary.tsv"

run_trial() {
  local round="$1"
  local variant="$2"
  local apply="$3"
  local trial="${ROOT}/round_$(printf '%02d' "${round}")_${variant}"
  mkdir -p "${trial}"

  cleanup_trial
  trap cleanup_trial EXIT INT TERM

  echo ""
  echo "================================================================"
  echo "ROUND ${round}/${ROUNDS}  VARIANT ${variant}"
  if [[ "${apply}" == "true" ]]; then
    echo "spawn path: EXPLICIT -J 0 for all seven joints"
  else
    echo "spawn path: NO -J arguments"
  fi
  echo "================================================================"

  setsid roslaunch arm_description gazebo_velocity_control.launch \
    world_file:="${WORLD_FILE}" \
    gazebo_gui:="${GAZEBO_GUI}" \
    use_rviz:="${USE_RVIZ}" \
    apply_initial_joint_overrides:="${apply}" \
    initial_joint1:=0.0 \
    initial_joint2:=0.0 \
    initial_joint3:=0.0 \
    initial_joint4:=0.0 \
    initial_wrist_joint1:=0.0 \
    initial_wrist_joint2:=0.0 \
    initial_wrist_joint3:=0.0 \
    > "${trial}/gazebo.log" 2>&1 &
  GAZEBO_PID=$!

  echo "[WAIT] joint states + raw link4_sensor2 cloud"
  for _ in $(seq 1 300); do
    if rostopic echo -n 1 /care_arm/joint_states >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
  timeout 10 rostopic echo -n 1 /care_arm/joint_states > "${trial}/joint_state_once.txt"
  timeout 20 rostopic echo -n 1 /link4_sensor2/tof/cloud >/dev/null

  # Give fixed-joint preservation / sensors a short deterministic settle.
  sleep 1.0

  echo "[VERIFY] q≈0"
  python3 - "${Q_TOL}" "${trial}/q_verify.json" <<'PY'
import json, sys, rospy
from sensor_msgs.msg import JointState

tol=float(sys.argv[1])
out=sys.argv[2]
names=['joint1','joint2','joint3','joint4',
       'wrist_joint1','wrist_joint2','wrist_joint3']

rospy.init_node('spawn_ab_q_verify', anonymous=True, disable_signals=True)
msg=rospy.wait_for_message('/care_arm/joint_states', JointState, timeout=10.0)
idx={n:i for i,n in enumerate(msg.name)}
q=[float(msg.position[idx[n]]) for n in names]
v=[float(msg.velocity[idx[n]]) if idx[n] < len(msg.velocity) else None
   for n in names]
err=max(abs(x) for x in q)
json.dump({'q':q,'velocity':v,'err_inf_rad':err,'tol_rad':tol},
          open(out,'w'),indent=2)
print('[Q VERIFY] q=',q,'err_inf=',err)
if err > tol:
    raise SystemExit('q=0 verification failed')
PY

  echo "[SNAPSHOT] Gazebo actual pose vs ROS TF"
  python3 src/care_confidence_map/scripts/capture_static_gazebo_tf_snapshot.py \
    --output "${trial}/static_gazebo_tf_snapshot.json" \
    --base-frame base_link \
    --model-name care_arm \
    --raw-topic /link4_sensor2/tof/cloud \
    --samples 8 \
    --period 0.20 \
    --max-clouds 8 \
    > "${trial}/static_gazebo_tf_snapshot.log" 2>&1

  cat "${trial}/static_gazebo_tf_snapshot.log"

  python3 - "${round}" "${variant}" "${apply}" "${trial}" "${ROOT}/summary.tsv" <<'PY'
import json, os, statistics, sys

round_id,variant,apply,trial,summary=sys.argv[1:]
qj=json.load(open(os.path.join(trial,'q_verify.json')))
d=json.load(open(os.path.join(trial,'static_gazebo_tf_snapshot.json')))

links=['link1','link2','link3','link4','link4_sensor2_tof_gz_link']
med={}
ang={}
for name in links:
    vals=[]
    avals=[]
    for s in d.get('pose_samples',[]):
        v=s.get('links',{}).get(name,{})
        if 'delta_norm_m' in v:
            vals.append(1000.0*float(v['delta_norm_m']))
            avals.append(float(v.get('delta_angle_deg',0.0)))
    med[name]=statistics.median(vals) if vals else float('nan')
    ang[name]=statistics.median(avals) if avals else float('nan')

clouds=d.get('raw_cloud_signatures',[])
h=clouds[0].get('sha256_stride8_xyz_um','none') if clouds else 'none'
finite=clouds[0].get('finite_count','none') if clouds else 'none'

# 5 mm is deliberately generous: the historical bad mode is 37.7/45 mm.
bad=max(med['link2'],med['link4'],med['link4_sensor2_tof_gz_link'])
cls='BAD_POSE' if bad > 5.0 else 'ALIGNED'

row=[
    round_id,variant,apply,
    '%.9f'%float(qj['err_inf_rad']),
    '%.3f'%med['link1'],
    '%.3f'%med['link2'],
    '%.3f'%med['link3'],
    '%.3f'%med['link4'],
    '%.3f'%med['link4_sensor2_tof_gz_link'],
    '%.5f'%ang['link4_sensor2_tof_gz_link'],
    h[:16],str(finite),cls,
]
with open(summary,'a') as f:
    f.write('\t'.join(row)+'\n')
print('[TRIAL SUMMARY]', ' '.join(row))
PY
}

for round in $(seq 1 "${ROUNDS}"); do
  # Alternate A then B within each round so machine-time drift cannot masquerade
  # as a variant effect.
  run_trial "${round}" A true
  run_trial "${round}" B false
done

cleanup_trial
trap - EXIT INT TERM

echo ""
echo "===================== A/B SUMMARY ====================="
column -t -s $'\t' "${ROOT}/summary.tsv" || cat "${ROOT}/summary.tsv"

python3 - "${ROOT}/summary.tsv" "${ROOT}/aggregate.txt" <<'PY'
import csv, collections, statistics, sys

src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
lines=[]
for variant in ['A','B']:
    rr=[r for r in rows if r['variant']==variant]
    classes=collections.Counter(r['classification'] for r in rr)
    def med(k):
        vals=[float(r[k]) for r in rr]
        return statistics.median(vals) if vals else float('nan')
    line=(
        'variant=%s n=%d aligned=%d bad_pose=%d '
        'median_link2_mm=%.3f median_link4_mm=%.3f '
        'median_render_mm=%.3f' %
        (variant,len(rr),classes['ALIGNED'],classes['BAD_POSE'],
         med('link2_mm'),med('link4_mm'),med('render_mm')))
    lines.append(line)

# Cross-tab raw depth signatures with pose mode.
sig=collections.defaultdict(collections.Counter)
for r in rows:
    sig[r['raw_hash0']][r['classification']] += 1
lines.append('raw_hash_classification:')
for h,c in sorted(sig.items()):
    lines.append('  %s %s' % (h,dict(c)))

open(dst,'w').write('\n'.join(lines)+'\n')
print('\n'.join(lines))
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
echo "[UPLOAD] ${ZIP}"
ls -lh "${ZIP}"
