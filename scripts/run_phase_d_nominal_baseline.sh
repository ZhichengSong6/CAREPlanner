#!/usr/bin/env bash
set -euo pipefail

# Nominal task-only baseline for CAREPlanner A/B visualization.
#
# Same robot, same case EE goal, same trajectory tracker and acceleration
# limiter as CAREPlanner, but no confidence map, VBC, VisCDF, GCDF,
# Local Sparse-SCP, REPAIR, or PROBE.  The tracker executes the original
# /care_planner/task_trajectory directly.
#
# Usage:
#   CASE_ID=case_014 RUN_SECONDS=25 \
#     bash scripts/run_phase_d_nominal_baseline.sh

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
CASE_ID="${CASE_ID:-case_014}"
RUN_SECONDS="${RUN_SECONDS:-25.0}"
CASE_FILE="${CASE_FILE:-${REPO}/src/egocentric_arm_planner/config/phase_c2_vbc_cases.json}"
CONFIG_FILE="${CONFIG_FILE:-${REPO}/src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml}"

STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
GIT_SHORT="$(git -C "${REPO}" rev-parse --short=8 HEAD)"
RUN_ID="${RUN_ID:-${CASE_ID}_task_only_${STAMP}_${GIT_SHORT}}"
ROOT="${REPO}/outputs/phase_d_nominal_baseline/${RUN_ID}"
OUT="${ROOT}/run"
LOG="${ROOT}/logs"

TRACKER_DESIRED_TOPIC="/care_planner/baseline/tracker_velocity_desired"
ACTUATOR_TOPIC="/care_arm/arm_group_velocity_controller/command"
TASK_TOPIC="/care_planner/task_trajectory"

cd "${REPO}"
source devel/setup.bash

if timeout 2 rosnode list >/dev/null 2>&1; then
  echo "[ERROR] ROS master already running. Close other ROS/Gazebo sessions first." >&2
  exit 1
fi

if ! GOAL_LINE="$(python3 - "${CASE_FILE}" "${CASE_ID}" <<'PY'
import json, sys
path, cid = sys.argv[1:]
db = json.load(open(path))
case = next((c for c in db["cases"] if c["case_id"] == cid), None)
if case is None:
    raise SystemExit("unknown case_id: " + cid)
print(*case["goal_position"], *case["goal_orientation"])
PY
)"; then
  echo "[ERROR] failed to resolve CASE_ID=${CASE_ID}" >&2
  exit 2
fi
read -r GX GY GZ GQX GQY GQZ GQW <<< "${GOAL_LINE}"

rm -rf "${ROOT}"
mkdir -p "${OUT}" "${LOG}"

cat > "${ROOT}/baseline_metadata.txt" <<EOF
mode=task_only_nominal
case_id=${CASE_ID}
run_id=${RUN_ID}
git_head=$(git rev-parse HEAD)
config_file=${CONFIG_FILE}
task_trajectory_topic=${TASK_TOPIC}
tracker_input_topic=${TASK_TOPIC}
careplanner_modules_enabled=0
EOF

GAZEBO_PID=""
PLANNER_PID=""
TRACKER_PID=""
REC_PIDS=()

kill_group() {
  local pid="${1:-}"
  [ -z "${pid}" ] && return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -INT -- "-${pid}" 2>/dev/null || true
    sleep 0.2
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    sleep 0.2
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  for pid in "${REC_PIDS[@]:-}"; do kill_group "${pid}"; done
  REC_PIDS=()
  kill_group "${TRACKER_PID}"; TRACKER_PID=""
  kill_group "${PLANNER_PID}"; PLANNER_PID=""
  kill_group "${GAZEBO_PID}"; GAZEBO_PID=""
  for name in gzclient gzserver rviz rosmaster roscore roslaunch; do
    pkill -TERM -x "${name}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

record_topic() {
  local topic="$1"
  local path="$2"
  setsid bash -lc "source '${REPO}/devel/setup.bash'; exec rostopic echo -p '${topic}'" \
    > "${path}" 2>&1 &
  REC_PIDS+=("$!")
}

echo "[BASELINE] case=${CASE_ID}"
echo "[BASELINE] same EE goal + same tracker, CAREPlanner modules disabled"
echo "[OUTPUT] ${ROOT}"

setsid roslaunch arm_description gazebo_velocity_control.launch \
  gazebo_gui:=false use_rviz:=false > "${LOG}/gazebo.log" 2>&1 &
GAZEBO_PID=$!

READY=0
for _ in $(seq 1 120); do
  if timeout 2 rostopic echo -n 1 /care_arm/joint_states >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 0.25
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] Gazebo joint state timeout" >&2
  exit 1
fi

# Upstream task generator only.  Its execution manager is disabled so the
# baseline and CAREPlanner use the same downstream full-trajectory tracker.
setsid roslaunch egocentric_arm_planner phase1_planner.launch \
  config_file:="${CONFIG_FILE}" \
  use_execution_manager:=false \
  use_interactive_marker:=false \
  use_trajectory_visualizer:=false \
  > "${LOG}/task_planner.log" 2>&1 &
PLANNER_PID=$!

setsid roslaunch egocentric_arm_planner c4_3_low_level_tracker.launch \
  config_file:="${CONFIG_FILE}" \
  input_trajectory:="${TASK_TOPIC}" \
  output_velocity_command:="${TRACKER_DESIRED_TOPIC}" \
  use_acceleration_limiter:=true \
  rate_limiter_output_velocity_command:="${ACTUATOR_TOPIC}" \
  > "${LOG}/tracker.log" 2>&1 &
TRACKER_PID=$!

READY=0
for _ in $(seq 1 120); do
  NODES="$(rosnode list 2>/dev/null || true)"
  if echo "${NODES}" | grep -q '^/receding_horizon_planner_node$' && \
     echo "${NODES}" | grep -q '^/trajectory_execution_manager_node$' && \
     echo "${NODES}" | grep -q '^/joint_velocity_rate_limiter$'; then
    READY=1
    break
  fi
  sleep 0.1
done
if [ "${READY}" != "1" ]; then
  echo "[ERROR] baseline planner/tracker nodes did not start" >&2
  rosnode list 2>/dev/null || true
  exit 1
fi

record_topic /care_arm/joint_states "${OUT}/joint_states.csv"
record_topic "${TASK_TOPIC}" "${OUT}/task_trajectory.csv"
record_topic /care_planner/execution/tracker_summary "${OUT}/tracker_summary.csv"
record_topic /care_planner/execution/reference_state "${OUT}/reference_state.csv"
record_topic "${TRACKER_DESIRED_TOPIC}" "${OUT}/tracker_desired_velocity.csv"
record_topic "${ACTUATOR_TOPIC}" "${OUT}/actuator_command.csv"

sleep 0.3

python3 - "${GX}" "${GY}" "${GZ}" "${GQX}" "${GQY}" "${GQZ}" "${GQW}" <<'PY'
import sys
import rospy
from geometry_msgs.msg import PoseStamped

vals = list(map(float, sys.argv[1:]))
rospy.init_node("careplanner_nominal_baseline_goal", anonymous=True)
pub = rospy.Publisher("/care_planner/ee_target_pose", PoseStamped, queue_size=1, latch=True)

deadline = rospy.Time.now() + rospy.Duration(2.0)
while pub.get_num_connections() == 0 and rospy.Time.now() < deadline and not rospy.is_shutdown():
    rospy.sleep(0.02)

msg = PoseStamped()
msg.header.stamp = rospy.Time.now()
msg.header.frame_id = "base_link"
msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = vals[:3]
msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = vals[3:]
pub.publish(msg)
rospy.sleep(0.2)
print("[BASELINE] published EE goal")
PY

# /care_planner/task_trajectory is intentionally a one-shot, non-latched
# publication.  A new 'rostopic echo' subscriber started after the goal can
# therefore miss the valid trajectory even though the tracker and recorder
# (which were subscribed before the goal) received it.  Verify the downstream
# tracker instead: it publishes at controller rate and carries the accepted
# trajectory's nonzero execution_stamp_ns.
if ! python3 - <<'PY'
import re
import rospy
from std_msgs.msg import String

rospy.init_node(
    "careplanner_nominal_baseline_tracker_wait",
    anonymous=True,
    disable_signals=True,
)
topic = "/care_planner/execution/tracker_summary"
deadline = rospy.Time.now() + rospy.Duration(4.0)
last = ""
stamp_re = re.compile(r"execution_stamp_ns=([0-9]+)")

while not rospy.is_shutdown() and rospy.Time.now() < deadline:
    try:
        msg = rospy.wait_for_message(topic, String, timeout=0.25)
        last = str(msg.data)
    except rospy.ROSException:
        continue

    m = stamp_re.search(last)
    if m is not None and int(m.group(1)) > 0:
        print("[BASELINE] tracker accepted one-shot task trajectory: " + last)
        raise SystemExit(0)

print("[ERROR] tracker never accepted the one-shot task trajectory")
if last:
    print("[LAST TRACKER SUMMARY] " + last)
raise SystemExit(1)
PY
then
  echo "[ERROR] nominal task trajectory was not accepted by the tracker" >&2
  echo "[DEBUG] task topic is transient/non-latched; inspect generation below" >&2
  grep -E 'New EE target|Cached one-shot nominal command|generation failed|intervention failed|Rejecting dynamically invalid|task_trajectory_topic|command_trajectory_topic' \
    "${LOG}/task_planner.log" | tail -n 80 || true
  exit 1
fi

echo "[RUN] task-only baseline for ${RUN_SECONDS}s"
sleep "${RUN_SECONDS}"

# Stop producers first; recorders then flush.
kill_group "${TRACKER_PID}"; TRACKER_PID=""
kill_group "${PLANNER_PID}"; PLANNER_PID=""
sleep 0.2
for pid in "${REC_PIDS[@]:-}"; do kill_group "${pid}"; done
REC_PIDS=()

echo "[BASELINE COMPLETE]"
echo "[RUN DIR] ${OUT}"
echo "${OUT}" > "${REPO}/outputs/phase_d_nominal_baseline/latest_${CASE_ID}.txt"
