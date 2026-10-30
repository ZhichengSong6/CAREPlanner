#!/usr/bin/env bash
set -euo pipefail
care_continuation_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 -B \
  "$care_continuation_repo/outputs/phase_e_continuation_20261029/run.py" "$@"
