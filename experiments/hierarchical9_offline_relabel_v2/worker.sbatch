#!/usr/bin/env bash
set -euo pipefail
: "${LABEL_VERIFY_SCRIPT:?Use submit.sh}"
bash "$LABEL_VERIFY_SCRIPT"
