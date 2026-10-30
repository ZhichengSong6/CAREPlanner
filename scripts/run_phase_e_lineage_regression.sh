#!/usr/bin/env bash
set -euo pipefail
cd /home/zhicheng/Project/CAREPlanner
export LINEAGE_STAMP="${LINEAGE_STAMP:-$(date +%Y%m%d-%H%M%S)-lineage}"
export LINEAGE_ROOT="outputs/phase_e_target_lineage/${LINEAGE_STAMP}"
python3 - <<'PY'
import os,sys,re
from pathlib import Path
sys.path.insert(0,'scripts')
from run_phase_e_repeatability import fingerprint,save,assert_no_ros,CHECKPOINTS
assert re.fullmatch(r'[A-Za-z0-9_-]+',os.environ['LINEAGE_STAMP'])
assert_no_ros()
out=Path(os.environ['LINEAGE_ROOT']);out.mkdir(parents=True,exist_ok=False)
state=fingerprint()
assert all(state[p]==sha for p,sha in CHECKPOINTS.items())
save(out/'frozen_sha256.json',state)
PY
export KEEP_CASE_ZIPS=1
export BATCH_STAMP="${LINEAGE_STAMP}-qp"
export BATCH_ID="phase_e_qp_events_${BATCH_STAMP}"
export DIAGNOSTIC_CASES=004,015,012 PER_SENSOR_HYBRID_ENABLED=true
bash scripts/run_phase_e_qp_events_regression.sh > "${LINEAGE_ROOT}/qp.log" 2>&1
python3 - <<'PY'
import os,sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
from run_phase_e_repeatability import assert_frozen,assert_no_ros
assert_frozen(json.loads((Path(os.environ['LINEAGE_ROOT'])/'frozen_sha256.json').read_text()))
assert_no_ros()
PY
echo 'QP batch complete; starting scalar/H9 007/026'
export AB_STAMP="${LINEAGE_STAMP}-ab"
bash scripts/run_phase_e_observation_ab.sh > "${LINEAGE_ROOT}/ab.log" 2>&1
python3 scripts/summarize_phase_e_qp_events.py \
  --qp-stamp "${LINEAGE_STAMP}-qp" --ab-stamp "${LINEAGE_STAMP}-ab" \
  --output "${LINEAGE_ROOT}/summary.json"
python3 - <<'PY'
import os,sys,json
from pathlib import Path
sys.path.insert(0,'scripts')
from run_phase_e_repeatability import assert_frozen,assert_no_ros
assert_frozen(json.loads((Path(os.environ['LINEAGE_ROOT'])/'frozen_sha256.json').read_text()))
assert_no_ros()
print('PASS frozen inputs and no remaining ROS/Gazebo processes')
PY
