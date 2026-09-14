#!/usr/bin/env bash
# Build/install only. Does not start ROS nodes, a GPU worker, Gazebo or a robot.
set -euo pipefail
release_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
release_workspace="$(cd -- "${release_script_dir}/.." && pwd)"
cd -- "${release_workspace}"
source /opt/ros/noetic/setup.bash
catkin config --profile primitive_release --extend /opt/ros/noetic \
  --build-space build/primitive_release --devel-space devel/primitive_release \
  --install-space install/primitive_release --log-space logs/primitive_release \
  --install --cmake-args -DCARE_INSTALL_CHECKPOINTS=OFF \
  -DCATKIN_ENABLE_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
catkin build --profile primitive_release egocentric_arm_planner care_visibility_cdf \
  arm_description --no-status -j2
# Checkpoints stay external. Legacy samples remain available for A/B and rollback.
printf '%s\n' 'Install candidate built: install/primitive_release' \
  'This is not a release approval. Run install preflight and private acceptance next.'
