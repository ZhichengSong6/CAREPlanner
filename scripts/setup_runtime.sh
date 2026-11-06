#!/usr/bin/env bash
# Explicitly invoked setup for Ubuntu 20.04 with ROS Noetic already installed.
set -euo pipefail
care_setup_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${care_setup_root}/.runtime_tmp"
care_setup_tmp="$(mktemp -d "${care_setup_root}/.runtime_tmp/setup.XXXXXX")"
export TMPDIR="${care_setup_tmp}" TMP="${care_setup_tmp}" TEMP="${care_setup_tmp}"
trap 'rm -rf -- "${care_setup_tmp}"' EXIT
care_setup_env="${CARE_RUNTIME_ENV:-careplanner}"
care_visibility_env="${CARE_VISIBILITY_ENV:-careplanner_visibility}"
if [[ "${care_setup_env}" == "${care_visibility_env}" ]]; then
  echo 'Use separate GPU and CPU visibility environment names.' >&2
  exit 2
fi
if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo 'Install ROS Noetic on Ubuntu 20.04 first; see README.md.' >&2
  exit 2
fi
if [[ -z "${CONDA_SH:-}" ]]; then
  for care_conda_dir in miniconda3 anaconda3 miniforge3 mambaforge; do
    if [[ -f "${HOME}/${care_conda_dir}/etc/profile.d/conda.sh" ]]; then
      CONDA_SH="${HOME}/${care_conda_dir}/etc/profile.d/conda.sh"
      break
    fi
  done
fi
test -f "${CONDA_SH:?Install Conda or set CONDA_SH to its etc/profile.d/conda.sh}"
source "${CONDA_SH}"
sudo apt-get update
sudo apt-get install -y build-essential cmake git zip xvfb xauth libgl1-mesa-dri libssl-dev libyaml-cpp-dev \
  libeigen3-dev python3-catkin-tools python3-numpy python3-yaml \
  ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control ros-noetic-ros-controllers \
  ros-noetic-robot-state-publisher ros-noetic-xacro ros-noetic-tf2-geometry-msgs \
  ros-noetic-interactive-markers ros-noetic-pcl-ros ros-noetic-urdfdom-py \
  ros-noetic-pinocchio ros-noetic-hpp-fcl
if ! conda run -n "${care_setup_env}" python --version >/dev/null 2>&1; then
  conda create -y -n "${care_setup_env}" python=3.10 pip
fi
conda run -n "${care_setup_env}" python -m pip install \
  'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
conda run -n "${care_setup_env}" python -m pip install \
  'numpy==2.2.6' 'PyYAML==6.0.3' 'rospkg==1.6.1' 'catkin_pkg==1.1.0' 'urdf-parser-py==0.0.4'
# Pin the NumPy BLAS implementation, not just its Python version. The validated
# R1 CPU path uses MKL; a generic pip NumPy wheel is a different runtime.
if ! conda run -n "${care_visibility_env}" python --version >/dev/null 2>&1; then
  conda create -y -n "${care_visibility_env}" --override-channels -c defaults \
    'python=3.10.20' pip 'numpy=2.2.5=py310h08c6c3d_3' \
    'numpy-base=2.2.5=py310h00548fb_3' 'blas=1.0=mkl' 'mkl=2025.0.0'
else
  conda install -y -n "${care_visibility_env}" --override-channels -c defaults \
    'python=3.10.20' 'numpy=2.2.5=py310h08c6c3d_3' \
    'numpy-base=2.2.5=py310h00548fb_3' 'blas=1.0=mkl' 'mkl=2025.0.0'
fi
conda run -n "${care_visibility_env}" python -m pip install \
  'torch==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu
conda run -n "${care_visibility_env}" python -m pip install \
  'PyYAML==6.0.3' 'rospkg==1.6.1' 'catkin_pkg==1.1.0' 'urdf-parser-py==0.0.4'
git -C "${care_setup_root}" submodule update --init --recursive
printf '%s\n' "Runtime environment ready: ${care_setup_env}" \
  "Visibility environment ready: ${care_visibility_env}" \
  "Next: bash scripts/run_demo.sh --env ${care_setup_env} --visibility-env ${care_visibility_env}"
