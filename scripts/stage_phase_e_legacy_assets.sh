#!/usr/bin/env bash
# Stage this already-built package only. Never write deployment setup files:
# catkin has absolute install destinations, so --prefix alone is insufficient.
set -euo pipefail
if [[ $# != 1 || -z "$1" ]]; then
  echo "usage: bash scripts/stage_phase_e_legacy_assets.sh NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
stage_out="$1"
if [[ -e "$stage_out" || -L "$stage_out" ]]; then
  echo "refusing existing staging directory: $stage_out" >&2
  exit 2
fi
mkdir -p "$stage_out"
stage_out="$(cd "$stage_out" && pwd)"
DESTDIR="$stage_out/root" cmake --install build/care_confidence_map --prefix /care-legacy-stage \
  > "$stage_out/install.log" 2>&1
echo "$stage_out/root/care-legacy-stage"
