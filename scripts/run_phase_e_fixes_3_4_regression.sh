#!/usr/bin/env bash
set -euo pipefail
# Six original cases only; fixed H9 V1 and existing qualification safety gates.
REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"
cd "${REPO}"
export BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d-%H%M%S)-fix34}"
export BATCH_ID="${BATCH_ID:-phase_e_h9_fix34_${BATCH_STAMP}}"
export CASE_FILE="${REPO}/outputs/phase_e_fix34_inputs/${BATCH_STAMP}.json"
python3 - "${CASE_FILE}" <<'PY'
import hashlib, json, pathlib, sys
source=pathlib.Path('outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json')
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'fbcd2816f594e1970f59e4aae2183f14a9764e623d80b86b2d37cf3a4cad1d45'
db=json.loads(raw)
ids=['phase_e_goal_008','phase_e_goal_004','phase_e_goal_015','phase_e_goal_002','phase_e_goal_026','phase_e_goal_012']
cases=[next(c for c in db['cases'] if c['case_id']==i) for i in ids]
subset=dict(db,cases=cases,diagnostic_source_sha256=hashlib.sha256(raw).hexdigest())
target=pathlib.Path(sys.argv[1]); target.parent.mkdir(parents=True,exist_ok=True)
with target.open('x') as f: json.dump(subset,f,indent=2)
print('Exact case subset:',ids)
PY
export EXPECTED_CASE_COUNT=6 RUN_SECONDS=45 EARLY_STOP_ON_GOAL=true
export PER_SENSOR_HYBRID_ENABLED=true
export PER_SENSOR_CHECKPOINT="${REPO}/src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt"
export NCDF_ENV=viscdf NCDF_DEVICE=cuda GPU_ENV=viscdf GPU_DEVICE=cuda:0
export GAZEBO_GUI=false USE_RVIZ=false
bash scripts/run_phase_e_empty_world_qualification.sh
