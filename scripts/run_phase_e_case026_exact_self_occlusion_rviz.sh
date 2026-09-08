#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"

# Keep ROS/RViz on the system environment.  A research conda env may override
# rospy/Qt/libstdc++ even though this diagnostic itself needs no GPU env.
if [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); then
  if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  while [[ "${CONDA_SHLVL:-0}" =~ ^[0-9]+$ ]] && (( CONDA_SHLVL > 0 )); do
    conda deactivate || break
  done
fi

cd "${REPO}"

if timeout 2 rosnode list >/dev/null 2>&1; then
  echo "[ERROR] ROS master already running. Close other ROS/Gazebo sessions first."
  exit 1
fi

echo "================================================================"
echo "CASE 026 EXACT SELF-OCCLUSION RVIZ DIAGNOSTIC"
echo "q_vis  : [-0.26948, 0.75081, -0.26677, -1.87081, 0.19022, -0.17289, -0.37924]"
echo "sensor : link4_sensor1_tof_link (S4)"
echo "target : [0.10, 0.05, 0.15]"
echo "test   : conservative primitive raycast vs exact STL triangle raycast"
echo "================================================================"

catkin build care_confidence_map
source devel/setup.bash

python3 -m py_compile   src/care_confidence_map/scripts/phase_e_case026_exact_self_occlusion_rviz.py

exec roslaunch care_confidence_map phase_e_case026_exact_self_occlusion_rviz.launch
