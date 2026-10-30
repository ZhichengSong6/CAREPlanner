#!/usr/bin/env bash
set -eo pipefail
care_demo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo 'ROS Noetic is required. See README.md before running this entry.' >&2
  exit 2
fi
source /opt/ros/noetic/setup.bash
exec /usr/bin/python3 "${care_demo_root}/scripts/run_demo.py" "$@"
