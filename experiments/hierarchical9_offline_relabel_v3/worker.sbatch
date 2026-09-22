#!/usr/bin/env bash
set -euo pipefail
: "${LABEL_SCRIPT:?Use submit.sh}"
bash "$LABEL_SCRIPT"
