#!/usr/bin/env bash
set -euo pipefail
care_executable_task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 -B \
  "$care_executable_task_repo/outputs/phase_e_executable_task_20261029/run.py" "$@"
