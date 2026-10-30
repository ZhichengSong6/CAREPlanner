#!/usr/bin/env bash
# Build only; runtime is created by prepare.py.
set -eo pipefail
care_baseline_run="$(realpath "${1:?prepared run directory required}")"
test -f "${care_baseline_run}/preparation.json"
source /opt/ros/noetic/setup.bash
set -u
care_baseline_runtime="${care_baseline_run}/fixed/runtime"
cd "${care_baseline_runtime}"
# catkin otherwise resolves an already-initialized ancestor workspace.
mkdir -p "${care_baseline_runtime}/.catkin_tools/profiles"
catkin init --workspace "${care_baseline_runtime}"
test -d "${care_baseline_runtime}/.catkin_tools"
catkin config --workspace "${care_baseline_runtime}" --extend /opt/ros/noetic \
  --source-space "${care_baseline_runtime}/src" --build-space "${care_baseline_runtime}/build" \
  --devel-space "${care_baseline_runtime}/devel" --log-space "${care_baseline_runtime}/logs" \
  --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCATKIN_ENABLE_TESTING=OFF \
  -DCARE_INSTALL_CHECKPOINTS=OFF -DCMAKE_NO_SYSTEM_FROM_IMPORTED=ON
catkin build --workspace "${care_baseline_runtime}" egocentric_arm_planner care_visibility_cdf arm_description --no-status -j2
