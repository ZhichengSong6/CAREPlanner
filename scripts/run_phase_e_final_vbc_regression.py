#!/usr/bin/env python3
"""Bounded canonical H9/primitive cases, with tested-binary/frozen guards."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from run_phase_e_repeatability import (REPO, PYTHON, CHECKPOINTS, fingerprint,
    assert_frozen, assert_no_ros, capture, save)
from run_phase_e_vbc_geometry_pair import validate_geometry_report, geometry_audit
from run_phase_e_repair_failure_regression import phase_audit
from summarize_phase_e_qp_events import collect
from analyze_phase_e_observation_events import string_rows
from audit_phase_e_rejection_snapshots import audit_snapshots


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stamp', required=True)
    ap.add_argument('--cases',choices=['007,026','026'],default='007,026',
                    help='Single 026 is reserved for bounded diagnostic-transport revalidation')
    ap.add_argument('--state-test-report', type=Path, required=True)
    ap.add_argument('--geometry-validation-report', type=Path, required=True)
    ap.add_argument('--snapshot-test-report', type=Path, required=True)
    ap.add_argument('--enable-bounded-recovery', action='store_true')
    args=ap.parse_args()
    cases=args.cases.split(',')
    assert re.fullmatch('[A-Za-z0-9_-]+',args.stamp)
    os.chdir(REPO); assert_no_ros()
    rviz=subprocess.run(['ps','-C','rviz','-o','pid='],capture_output=True,text=True)
    assert rviz.returncode in (0,1) and not rviz.stdout.strip(), 'existing RViz'
    short=capture(['git','rev-parse','--short=8','HEAD'])
    out=REPO/'outputs/phase_e_final_vbc_guard'/args.stamp
    batch=REPO/'outputs/phase_e_empty_world_qualification'/('phase_e_'+args.stamp)
    paths=[out,batch,REPO/('CAREPlanner_PHASE_E_EMPTY_QUALIFICATION_phase_e_'+args.stamp+'.zip')]
    for suffix in ('.json','.source_sha256.json'):
        paths.append(REPO/'outputs/phase_e_qp_events_inputs'/(args.stamp+suffix))
    for case in cases:
        run='phase_e_goal_'+case+'_empty_qualification_'+args.stamp+'_'+short
        paths += [REPO/'outputs/c5_5_vbc_gcdf_regime'/run, REPO/'logs/c5_5_vbc_gcdf_regime'/run,
                  REPO/('CAREPlanner_C5_RESULT_'+run+'.zip')]
    assert not any(p.exists() for p in paths), 'Existing run identity; no overwrite'
    frozen=fingerprint()
    assert all(frozen[p]==v for p,v in CHECKPOINTS.items()), 'Fixed checkpoint changed'
    state=json.loads(args.state_test_report.read_text())
    assert state['status']=='PASS' and state['binary_sha256']==frozen['devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node']
    snapshot_test=json.loads(args.snapshot_test_report.read_text())
    assert snapshot_test['status']=='PASS' and Path(snapshot_test['snapshot']).is_file(), 'Snapshot ROS transport test not passed'
    geometry=validate_geometry_report(args.geometry_validation_report,frozen)
    cuda=capture([PYTHON,'-c','import json,torch; assert torch.cuda.is_available(); '
        'q=torch.ones(7,device="cuda",requires_grad=True); g=torch.autograd.grad((q*q).sum(),q)[0]; '
        'print(json.dumps(dict(gpu=torch.cuda.get_device_name(0),gradient=g.cpu().tolist())))'])
    out.mkdir(parents=True,exist_ok=False)
    save(out/'frozen_sha256.json',frozen)
    save(out/'geometry_validation_binding.json',geometry)
    save(out/'preflight.json',dict(cuda=json.loads(cuda),state_test_report=str(args.state_test_report),
        state_test_sha256=capture(['sha256sum',str(args.state_test_report)]).split()[0],
        snapshot_test_report=str(args.snapshot_test_report),
        snapshot_test_sha256=capture(['sha256sum',str(args.snapshot_test_report)]).split()[0],
        dirty_status=capture(['git','status','--short']),planned_runs=len(cases),cases=cases))
    overrides=dict(BATCH_STAMP=args.stamp,BATCH_ID='phase_e_'+args.stamp,
        DIAGNOSTIC_CASES=args.cases,PER_SENSOR_HYBRID_ENABLED='true',
        FINAL_VBC_RECOVERY_ENABLED='true' if args.enable_bounded_recovery else 'false',
        VBC_GEOMETRY_BACKEND='primitive',VBC_SWEPT_VOLUME_MARGIN_M='0.0',KEEP_CASE_ZIPS='1')
    save(out/'environment.json',overrides)
    print('PREFLIGHT PASS; START exactly',cases,flush=True)
    with (out/'runner.log').open('x') as stream:
        proc=subprocess.run(['bash','scripts/run_phase_e_qp_events_regression.sh'],
            env=dict(os.environ,**overrides),stdout=stream,stderr=subprocess.STDOUT)
    save(out/'exit.json',dict(return_code=proc.returncode))
    assert_frozen(frozen); assert_no_ros()
    assert proc.returncode==0, 'runner failed; no automatic retry'
    pool=json.loads((REPO/'outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json').read_text())
    subset=json.loads((REPO/'outputs/phase_e_qp_events_inputs'/(args.stamp+'.json')).read_text())
    assert subset['cases']==[next(c for c in pool['cases'] if c['case_id']=='phase_e_goal_'+i) for i in cases]
    rows=collect(REPO,batch,args.stamp,'h9_primitive')
    for row in rows:
        row['repair_state']=phase_audit(row)
        row['geometry_audit']=geometry_audit(row,'primitive')
        local=string_rows(Path(row['run'])/'local_planner_summary.csv',row['benchmark_window'])
        row['final_vbc_feedback']=[r for r in local if r.get('event','').startswith('repair_final_vbc_')]
        snapshots=sorted((Path(row['run'])/'rejection_snapshots').glob('*.json'))
        row['snapshot_files']=[str(p) for p in snapshots]
        row['snapshot_bytes']=sum(p.stat().st_size for p in snapshots)
        row['snapshot_audit']=audit_snapshots(row['run'],row['benchmark_window'])
    save(out/'summary.json',dict(cases=rows,frozen_inputs_equal=True,
        note='Task failures retained; no claim of trajectory-matched improvement.'))
    assert all(r['overall_safe'] and r['startup_order_pass'] for r in rows), 'Safety/startup invariant failed'
    assert all(r['snapshot_audit']['status']!='FAIL' for r in rows), 'Snapshot integrity/capture failed; see saved summary'
    print('COMPLETE',[(r['case_id'],r['task_success'],r['overall_safe'],len(r['snapshot_files'])) for r in rows],flush=True)


if __name__=='__main__': main()
