#!/usr/bin/env bash
# One supported install-only command entry. No command means no ROS launch.
set -euo pipefail
if [[ $# -eq 0 ]]; then
  printf '%s\n' 'Usage: bash scripts/with_phase_e_primitive_release.sh COMMAND [ARG ...]' >&2
  exit 2
fi
release_entry_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
release_entry_prefix="$(cd -- "${release_entry_dir}/.." && pwd)/install/primitive_release"
if [[ ! -f "${release_entry_prefix}/setup.bash" ]]; then
  printf '%s\n' 'Missing install/primitive_release/setup.bash; build the release profile first.' >&2
  exit 1
fi
# Do not inherit another catkin workspace's package/library/module overlays.
# This only changes the child environment, never the user's interactive shell.
# Keep ROS_MASTER_URI and other explicit runtime settings; this is not a private
# master or a permission to run hardware. Use absolute paths for non-system Python.
exec env -u CMAKE_PREFIX_PATH -u ROS_PACKAGE_PATH -u ROSLISP_PACKAGE_DIRECTORIES \
  -u LD_LIBRARY_PATH -u PYTHONPATH -u PKG_CONFIG_PATH \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash --noprofile --norc -c '
    set -e
    source "$1/setup.bash"
    shift
    exec "$@"
  ' primitive-release "${release_entry_prefix}" "$@"
