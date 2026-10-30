#!/usr/bin/env python3
"""Counterfactual scheduling replay of recorded CASE006 evidence, no commands.

Uses recorded certified trajectory shape (constant rebase, zero crop) and
query-row evidence. It does not replay changed closed-loop motion/map updates.
"""
import argparse
import bisect
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from diagnose_phase_e_case006_o6_recovery import strings
from evaluate_phase_d_run import joint_states,JOINTS

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--policy-binary',type=Path,required=True)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    run=root/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_006_r1_case006_sweep_fix_20261007/run'
    inputs=[run/(n+'.csv') for n in ['verification_outcome','local_planner_summary','local_witness_selector_diagnostics',
        'local_gcdf_query','committed_trajectory','tracker_summary','joint_states']]
    def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    hashes={str(p.relative_to(root)):digest(p) for p in inputs}
    ver=strings(inputs[0]);pub=[r for r in strings(inputs[1]) if r.get('event')=='candidate_published']
    witness=strings(inputs[2])
    def numeric(p):
        with p.open() as f:return {r['field.header.stamp']:r for r in csv.DictReader(f)}
    queries=numeric(inputs[3]);commits=numeric(inputs[4]);tracker=strings(inputs[5])
    samples=joint_states(str(inputs[6]),JOINTS,True);ts=[t for t,q in samples]
    def q(row):return np.array([[float(row[f'field.points{k}.positions{j}']) for j in range(7)] for k in range(21)])
    events=[];evidence=[]
    for v in ver:
        if not (19.393e9<=v['receipt_ns']<=27.402e9 and v['result']=='safe'):continue
        assert v['verification_view']=='full_horizon'
        for k in ['dispatch_suffix_phase_s','precommit_suffix_phase_s']:
            assert float(v[k])==0,(v['seq'],k)
        raw=int(v['raw_candidate_stamp_ns']);execution=int(v['execution_stamp_ns'])
        # Recorder callback receipt may differ by 1 ms; query identity comes
        # from this unique publication, and the offset is retained explicitly.
        p=min(pub,key=lambda r:abs(r['receipt_ns']-raw))
        assert abs(p['receipt_ns']-raw)<=1000000
        query=p['query_stamp_ns'];ww=[r for r in witness if r['query_stamp_ns']==query]
        assert len(ww)==60 and p['obligation_cdf_rows']=='60'
        hold=q(queries[query]);cq=q(commits[str(execution)])
        shift=cq[0]-hold[0]
        assert abs(np.abs(shift).max()-float(v['handoff_rebase_shift_inf']))<1e-6
        # Production rebase adds one constant joint vector to every knot.
        rawq=cq-shift
        # CASE006 late task seed direction has positive joint2 displacement;
        # exact q_ref is not separately recorded. Check the task-seed endpoint
        # on this plan as the logged directional proxy (not a KKT proof).
        start=next(r for r in strings(inputs[1]) if r.get('plan_seq')==p['plan_seq'] and r.get('event')=='plan_started')
        seed=q(queries[start['query_stamp_ns']]);direction=seed[-1]-hold[0]
        qualifying=[]
        for w in ww:
            grad=np.array(json.loads(w['gradient']));qlin=np.array(json.loads(w['q_linearization']))
            clearance=float(w['distance'])+float(grad@(rawq[int(w['timestep'])]-qlin))-.05
            along=float(grad@direction)
            if w['status']=='evaluated_unknown' and abs(clearance)<=.001 and along<-1e-6:
                qualifying.append(dict(point=json.loads(w['point']),clearance=clearance,along=along))
        span=float(np.abs(cq-cq[0]).max());assert qualifying
        evidence.append(dict(seq=v['seq'],query=query,raw=raw,execution=execution,span=span,
            publication_receipt_offset_ns=p['receipt_ns']-raw,qualifying_rows=len(qualifying)))
        events.append((raw,0,f'R {raw} {query} {span:.17g} 1'))
        events.append((v['receipt_ns'],1,f'C {raw} {execution}'))
    for r in tracker:
        t=r['receipt_ns']
        if not 19.393e9<=t<=27.402e9:continue
        execution=int(r.get('execution_stamp_ns',0));phase=float(r.get('phase_s',-1))
        if r.get('active')!='1' or r.get('complete')=='1' or r.get('execution_aborted')!='0':continue
        if phase<.05 or (t-execution)/1e9<.05:continue
        i=bisect.bisect_right(ts,t)-1;stamp,mq=samples[i]
        eligible=int(0<=t-stamp<=200000000 and .05<=(t-execution)/1e9<=2)
        events.append((t,2,f'O {t/1e9:.9f} {execution} {eligible} '+' '.join(f'{x:.17g}' for x in mq)))
    text='\n'.join(e[2] for e in sorted(events))+'\n'
    (a.output_dir/'events.txt').write_text(text)
    p=subprocess.run([str(a.policy_binary.resolve()),'--replay'],input=text,text=True,capture_output=True,check=True)
    lines=p.stdout.splitlines();assert len(lines)==1,lines
    t,count=lines[0].split();t=float(t)
    assert 20.<t<25. and int(count)>=3
    assert all(digest(root/path)==h for path,h in hashes.items())
    result=dict(status='PASS',early_event_ros_s=t,execution_stamps=int(count),original_event_ros_s=27.402,
        earlier_ros_s=27.402-t,certified_candidates=evidence,input_sha256=hashes,
        scope='Fixed-record counterfactual; certified shape and task-seed directional proxy. Not a new rollout or exact callback-order reconstruction.')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
