#!/usr/bin/env bash
set -euo pipefail
# Bounded, exact original-case diagnostics, never an implicit 30-case run.
cd /home/zhicheng/Project/CAREPlanner
export BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d-%H%M%S)-qp-events}"
export BATCH_ID="${BATCH_ID:-phase_e_qp_events_${BATCH_STAMP}}"
export CASE_FILE="${PWD}/outputs/phase_e_qp_events_inputs/${BATCH_STAMP}.json"
export DIAGNOSTIC_CASES="${DIAGNOSTIC_CASES:-004,015,012}"
python3 - "${CASE_FILE}" <<'PY'
import hashlib, json, os, pathlib, sys
source=pathlib.Path('outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json')
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='fbcd2816f594e1970f59e4aae2183f14a9764e623d80b86b2d37cf3a4cad1d45'
db=json.loads(raw)
ids=['phase_e_goal_'+i for i in os.environ['DIAGNOSTIC_CASES'].split(',')]
assert 1 <= len(ids) <= 6 and len(set(ids)) == len(ids)
cases=[next(c for c in db['cases'] if c['case_id']==i) for i in ids]
out=pathlib.Path(sys.argv[1]); out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:
    json.dump(dict(db,cases=cases,diagnostic_source_sha256=hashlib.sha256(raw).hexdigest()),f,indent=2)
paths=[p for root in ('src/egocentric_arm_planner','src/care_visibility_cdf/scripts','scripts')
       for p in pathlib.Path(root).rglob('*') if p.suffix in ('.cpp','.hpp','.py','.yaml','.launch','.sh')]
with out.with_suffix('.source_sha256.json').open('x') as f:
    json.dump({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},f,indent=2)
print('Exact bounded case subset:',ids)
PY
export EXPECTED_CASE_COUNT="$(python3 -c 'import os; print(len(os.environ["DIAGNOSTIC_CASES"].split(",")))')"
export RUN_SECONDS=45 EARLY_STOP_ON_GOAL=true
export PER_SENSOR_HYBRID_ENABLED="${PER_SENSOR_HYBRID_ENABLED:-true}"
export PER_SENSOR_CHECKPOINT="${PWD}/src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt"
export NCDF_ENV=viscdf NCDF_DEVICE=cuda GPU_ENV=viscdf GPU_DEVICE=cuda:0
export GAZEBO_GUI=false USE_RVIZ=false
bash scripts/run_phase_e_empty_world_qualification.sh
