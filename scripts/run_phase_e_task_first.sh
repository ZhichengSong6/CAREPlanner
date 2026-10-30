#!/usr/bin/env bash
set -euo pipefail
care_task_first_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /usr/bin/python3 -B \
  "$care_task_first_repo/outputs/phase_e_task_first_20261028/run.py" "$@"
