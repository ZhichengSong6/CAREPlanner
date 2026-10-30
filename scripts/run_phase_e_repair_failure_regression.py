#!/usr/bin/env python3
"""Four frozen-runtime simulations: scalar/H9 x original failures 007/026.

Reuses the canonical case runner. No automatic repeats or runtime edits.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess

from run_phase_e_repeatability import (REPO, PYTHON, CHECKPOINTS, fingerprint,
    assert_frozen, assert_no_ros, capture, save)
from summarize_phase_e_qp_events import collect
from analyze_phase_e_observation_events import string_rows


def phase_audit(case):
    run, window = Path(case['run']), case['benchmark_window']
    events = string_rows(run/'local_planner_summary.csv', window)
    repair = [e for e in events if e.get('repair') == '1']
    started = [e for e in repair if e.get('event') == 'plan_started']
    solved = [e for e in repair if e.get('event') == 'scp_solved']
    candidates = [e for e in repair if e.get('event') == 'candidate_published']
    retreat_candidates = [e for e in candidates if e.get('repair_phase') == 'retreat']
    failures = [e for e in repair if e.get('event') == 'scp_failed']
    feedback = [e for e in repair if e.get('event') in ('repair_qp_exhausted','repair_plan_exhausted')]
    regime = string_rows(run/'regime_summary.csv', window)
    witnesses = string_rows(run/'local_witness_diagnostics.csv', window)
    upstream = string_rows(run/'local_witness_selector_diagnostics.csv', window)
    # Trace result flags belong to the captured plan. Check within that plan,
    # not against the latest active observation token or a nearby timestamp.
    violations = [e for e in retreat_candidates if not (
        e.get('repair_witness_fresh') == '1' and e.get('solved') == '1' and
        e.get('observation_token') == 'none' and e.get('visibility_objective_step') == '-1')]
    return dict(phase_fields_present=all('repair_phase' in e for e in repair),
        plans_by_phase=dict(Counter(e.get('repair_phase','MISSING') for e in started)),
        solved_by_phase=dict(Counter(e.get('repair_phase','MISSING') for e in solved)),
        candidates_by_phase=dict(Counter(e.get('repair_phase','MISSING') for e in candidates)),
        failure_status_counts=dict(Counter(e.get('status','MISSING') for e in failures)),
        witness_missing_failures=[e for e in failures if e.get('status')=='repair_witness_missing_or_invalid'],
        exhausted_events=feedback,retreat_candidate_gate_violations=violations,
        max_retained_witnesses=max([int(e.get('repair_witness_count',0)) for e in repair] or [0]),
        last_local_event=events[-1] if events else None,
        last_regime=regime[-1] if regime else None,
        repair_plan_starts=started,repair_failures=failures,retreat_candidates=retreat_candidates,
        witness_reason_counts=dict(Counter(e.get('reason','MISSING') for e in witnesses)),
        upstream_witness_status_counts=dict(Counter(e.get('status','MISSING') for e in upstream)),
        witness_diagnostics=witnesses,upstream_witness_diagnostics=upstream)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stamp',default=datetime.now().strftime('%Y%m%d-%H%M%S')+'-repair')
    ap.add_argument('--state-test-report', default='outputs/phase_e_repair_state/production_20260911_03/result.json',
                    help='PASS production planner test report whose binary SHA must match this run')
    args=ap.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',args.stamp): ap.error('invalid stamp')
    os.chdir(REPO)
    assert_no_ros()
    short=capture(['git','rev-parse','--short=8','HEAD'])
    # The old runner recreates its run/log paths. Assert exact targets absent
    # BEFORE invoking it so existing user outputs can never be overwritten.
    for model in ('scalar','h9'):
        stamp=args.stamp+'-'+model
        batch='phase_e_'+stamp
        paths=[REPO/'outputs/phase_e_empty_world_qualification'/batch,
               REPO/('CAREPlanner_PHASE_E_EMPTY_QUALIFICATION_'+batch+'.zip'),
               REPO/'outputs/phase_e_qp_events_inputs'/(stamp+'.json')]
        for case in ('007','026'):
            run='phase_e_goal_'+case+'_empty_qualification_'+stamp+'_'+short
            paths += [REPO/'outputs/c5_5_vbc_gcdf_regime'/run,
                      REPO/'logs/c5_5_vbc_gcdf_regime'/run,
                      REPO/('CAREPlanner_C5_RESULT_'+run+'.zip')]
        if any(p.exists() for p in paths): raise RuntimeError('run identity already exists')
    out=REPO/'outputs/phase_e_repair_failure_regression'/args.stamp
    out.mkdir(parents=True,exist_ok=False)
    frozen=fingerprint()
    assert all(frozen[p]==v for p,v in CHECKPOINTS.items()), 'fixed checkpoint changed'
    passed=json.loads((REPO/args.state_test_report).read_text())
    assert passed['status']=='PASS'
    assert frozen['devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node']==passed['binary_sha256'], 'planner differs from tested binary'
    save(out/'frozen_sha256.json',frozen)
    cuda=capture([PYTHON,'-c','import json,torch; assert torch.cuda.is_available(); '
        'q=torch.ones(7,device="cuda",requires_grad=True); '
        'g=torch.autograd.grad((q*q).sum(),q)[0]; '
        'print(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),gradient=g.cpu().tolist())))'])
    save(out/'preflight.json',dict(cuda=json.loads(cuda),gpu=capture(['nvidia-smi']),
        git_head=capture(['git','rev-parse','HEAD']),dirty_status=capture(['git','status','--short']),
        planned_runs=4,cases=['007','026'],watchdog_s=45,
        baseline='outputs/phase_e_target_lineage/20260911-081348-lineage/summary.json'))
    old=json.loads((REPO/'outputs/phase_e_target_lineage/20260911-081348-lineage/summary.json').read_text())
    rows=[]
    print('OUTPUT',out,flush=True)
    for model in ('scalar','h9'):
        assert_frozen(frozen); assert_no_ros()
        stamp=args.stamp+'-'+model
        env=dict(os.environ,BATCH_STAMP=stamp,BATCH_ID='phase_e_'+stamp,
                 DIAGNOSTIC_CASES='007,026',PER_SENSOR_HYBRID_ENABLED=str(model=='h9').lower(),KEEP_CASE_ZIPS='1')
        print('START',model,'007/026',flush=True)
        with (out/(model+'.log')).open('x') as stream:
            result=subprocess.run(['bash','scripts/run_phase_e_qp_events_regression.sh'],env=env,
                                  stdout=stream,stderr=subprocess.STDOUT)
        save(out/(model+'_exit.json'),dict(return_code=result.returncode))
        assert_frozen(frozen); assert_no_ros()
        if result.returncode: raise RuntimeError('runner failed; no automatic retry')
        batch=REPO/'outputs/phase_e_empty_world_qualification'/('phase_e_'+stamp)
        group=collect(REPO,batch,stamp,model)
        for row in group:
            row['repair_state']=phase_audit(row)
            baseline=next(c for c in old['cases'] if c['model']==model and c['case_id']==row['case_id'])
            row['baseline']={k:baseline[k] for k in ('run','task_success','overall_safe','final_position_error_m','final_orientation_error_rad')}
        save(out/(model+'_summary.json'),dict(cases=group))
        rows+=group
        print('END',model,[(r['case_id'],r['task_success'],r['overall_safe'],r['repair_state']['plans_by_phase']) for r in group],flush=True)
        if not all(r['overall_safe'] and r['startup_order_pass'] and
                   r['repair_state']['phase_fields_present'] and
                   not r['repair_state']['retreat_candidate_gate_violations'] for r in group):
            raise RuntimeError('safety/startup/trace invariant failed; stop before next batch')
    assert_frozen(frozen); assert_no_ros()
    save(out/'summary.json',dict(cases=rows,completed_runs=len(rows),frozen_inputs_equal=True,
        note='Single run per group; task failures retained. Synthetic state-machine PASS does not imply task success.'))
    print('COMPLETE four runs; frozen inputs unchanged',flush=True)


if __name__=='__main__': main()
