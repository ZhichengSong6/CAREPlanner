#!/usr/bin/env python3
"""Fixed trace replay of CASE024 tight-row classification and tangent steering.

No ROS publication, model calls, production changes, or full-QP replay claim.
"""
import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from diagnose_phase_e_case006_o6_recovery import strings, digest
from analyze_phase_e_observation_events import tokens

ROOT = Path(__file__).resolve().parents[1]
BATCH = 'r1_case024_replacement_handoff_20261009'
RUN = ROOT / 'outputs/c5_5_vbc_gcdf_regime' / ('phase_e_goal_024_' + BATCH) / 'run'
LOG = ROOT / 'logs/c5_5_vbc_gcdf_regime' / ('phase_e_goal_024_' + BATCH) / 'run/controlled.log'

def key(point):
    return tuple(round(float(v), 5) for v in point)

def tangent(q0, desired, gradients, attempt):
    basis = []
    for g in gradients:
        v = g / np.linalg.norm(g)
        for b in basis:
            v -= np.dot(b, v) * b
        if np.linalg.norm(v) > 1e-7:
            basis.append(v / np.linalg.norm(v))
    projector = np.eye(7) - sum(np.outer(b, b) for b in basis)
    directions, axes = [], []
    for j in range(7):
        v = projector[:, j]
        if np.linalg.norm(v) > .1:
            directions.extend([v, -v]); axes.extend([j+1, -(j+1)])
    v = directions[(attempt - 1) % len(directions)].copy()
    forward = projector @ (desired - q0)
    if np.linalg.norm(forward) > 1e-6:
        v += .2 * forward / np.linalg.norm(forward)
    v *= .03 / np.max(np.abs(v))
    return q0 + v, axes[(attempt - 1) % len(directions)], float(np.linalg.norm(forward))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--binary', type=Path, required=True)
    args = ap.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    inputs = [LOG, Path(__file__).resolve(), Path(__file__).resolve().with_suffix('.cpp'),
              ROOT/'scripts/diagnose_phase_e_case006_o6_recovery.py',
              ROOT/'scripts/analyze_phase_e_observation_events.py',
              ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/safe_frontier_recovery.hpp',
              ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp']
    def read(name, plain=False):
        p = RUN / (name + '.csv'); inputs.append(p)
        if plain:
            with p.open() as f: return list(csv.DictReader(f))
        return strings(p)
    planner = read('local_planner_summary')
    solved = [r for r in planner if r['event']=='scp_solved' and 19107000000<=r['receipt_ns']<=22518000000]
    assert len(solved)==52
    witnesses=defaultdict(list)
    for r in read('local_witness_selector_diagnostics'): witnesses[r['query_stamp_ns']].append(r)
    log=LOG.read_text()
    recovered_rows=[]
    wanted={r['query_stamp_ns'] for r in solved}
    for payload in re.findall(r'C5_WITNESS event=witness_reevaluation [^\n]*?gradient=\[[^\]\n]+\]',log):
        row=tokens(payload)
        if row.get('query_stamp_ns') not in wanted: continue
        try:
            assert len(json.loads(row['gradient']))==7
            assert len(json.loads(row['q_linearization']))==7
            assert len(json.loads(row['point']))==3
            assert len(json.loads(row['q_requested']))==7
            float(row['distance']); int(row['batch_pair']); int(row['timestep'])
        except (AssertionError,KeyError,ValueError): continue
        existing=next((r for r in witnesses[row['query_stamp_ns']] if r['batch_pair']==row['batch_pair']),None)
        if existing:
            assert all(existing[k]==row[k] for k in ('point','gradient','distance','q_requested','q_linearization','timestep','plan_seq'))
        else:
            witnesses[row['query_stamp_ns']].append(row)
            recovered_rows.append(dict(query=row['query_stamp_ns'],pair=int(row['batch_pair']),source='controlled.log'))
    queries={r['field.header.stamp']:r for r in read('local_gcdf_query', True)}
    frontiers=read('visibility_frontier_target', True)
    # Exact point feedback/rollback values from the production log, matched to
    # trace coordinates at the production point-identity tolerance (1e-5).
    margins={}; margin_events=[]
    for line in log.splitlines():
        if 'exact VBC point feedback point=[' in line or 'point_margin_rollback point=[' in line:
            point=[float(v) for v in re.search(r'point=\[([^]]+)\]',line)[1].split()]
            is_rollback='point_margin_rollback' in line
            value=float(re.search(r'(?:to=|cdf_margin=)([\d.e+-]+)',line)[1])
            timestamp=float(re.search(r', ([\d.]+)\]',line)[1])
            if timestamp>19.107: raise AssertionError('Unexpected later margin mutation')
            margins[key(point)]=value
            margin_events.append(dict(ros_s=timestamp,point=point,margin=value,rollback=is_rollback))
    assert len(margins)==10
    plan_rows=defaultdict(list)
    for s in solved: plan_rows[s['plan_seq']].append(s)
    records=[]; cpp_inputs=[]; previous_target=None; previous_desired=None
    plan_frontier_candidates={}
    for s in solved:
        stamp=s['query_stamp_ns']; query=queries[stamp]
        q0=np.array([float(query['field.points0.positions'+str(i)]) for i in range(7)])
        target=np.array(json.loads(s['recovery_frontier_target']))
        ws=sorted(witnesses[stamp],key=lambda r:int(r['batch_pair']))
        assert len(ws)==int(s['batch_pairs'])==int(s['cdf_rows'])==200
        assert [int(r['batch_pair']) for r in ws]==list(range(200))
        assert all(r['status']=='evaluated_unknown' for r in ws)
        assert all(r['plan_seq']==s['plan_seq'] for r in ws)
        k2=[]
        for w in ws:
            qbar=np.array([float(query['field.points'+w['timestep']+'.positions'+str(i)]) for i in range(7)])
            assert np.max(abs(qbar-np.array(json.loads(w['q_requested']))))<1e-12
            if w['timestep']!='2': continue
            point=json.loads(w['point']); gradient=np.array(json.loads(w['gradient']))
            qlin=np.array(json.loads(w['q_linearization']))
            at_start=float(w['distance'])+gradient@(q0-qlin)
            margin=max(.05,margins.get(key(point),.01))
            k2.append(dict(point=point,gradient=gradient.tolist(),distance=float(w['distance']),
                           q_linearization=qlin.tolist(),margin=margin,at_start=float(at_start),
                           clearance=float(at_start-margin), source='UNKNOWN',batch_pair=int(w['batch_pair'])))
        tight=[r for r in k2 if r['clearance']<.005 and np.linalg.norm(r['gradient'])>1e-9]
        assert len(tight)==int(s['recovery_frontier_normals'])
        gradients=[np.array(r['gradient']) for r in tight]
        # Frontier messages are not stamped. Identify the input by reproducing
        # the full-precision saved target, not by assuming nearest receipt wins.
        first=plan_rows[s['plan_seq']][0]
        anchor=int(first['query_stamp_ns'])
        candidates=[r for r in frontiers if anchor-200000000<=int(r['%time'])<=anchor+20000000]
        unique={tuple(float(r['field.data'+str(i)]) for i in range(3,10)):r for r in candidates}
        matches=[]
        for desired_tuple,f in unique.items():
            desired=np.array(desired_tuple)
            reproduced,axis,projected_norm=tangent(q0,desired,gradients,int(s['safe_stall_attempts']))
            err=float(np.max(abs(reproduced-target)))
            if err<1e-11: matches.append((err,desired,f,axis,projected_norm))
        reused=previous_target is not None and np.max(abs(target-previous_target))<1e-14
        if matches:
            matches.sort(key=lambda m:m[0]); err,desired,f,axis,projected_norm=matches[0]
            # Any indistinguishable duplicates must agree on the direction conclusions.
            for _,alternative,_,_,_ in matches:
                for r in tight: assert np.dot(r['gradient'],alternative-q0)>0
            frontier_receipt=int(f['%time'])
            cpp_inputs.append(dict(q0=q0.tolist(),desired=desired.tolist(),gradients=[g.tolist() for g in gradients],
                                   attempt=int(s['safe_stall_attempts']),step=.03,expected=target.tolist(),query=stamp))
            mode='recomputed_target_match'
            desired_candidates=[m[1] for m in matches]
            frontier_binding='unique_recorded_target_match' if len(matches)==1 else 'compatible_target_matches'
        else:
            assert reused, ('No trace target reproduction',stamp)
            desired=previous_desired; err=None; axis=None; projected_norm=None; frontier_receipt=None
            mode='retained_target'
            desired_candidates=[np.array(v) for v in unique]
            assert desired_candidates
            frontier_binding='unstamped_frontier_receipt_window_only'
        # plan_frontier is copied once at plan start, including on subsequent SCP iterations.
        if s is not first:
            desired=next(r['_desired'] for r in records if r['plan_seq']==s['plan_seq'])
            desired_candidates,frontier_binding=plan_frontier_candidates[s['plan_seq']]
        else:
            plan_frontier_candidates[s['plan_seq']]=(desired_candidates,frontier_binding)
        delta=desired-q0
        for r in k2:
            g=np.array(r['gradient'])
            r['forward_dot']=float(g@delta)
            r['forward_clearance']=float(r['clearance']+g@delta)
            r['detour_dot']=float(g@(target-q0))
            r['dependency_condition']=bool(r['clearance']<.005 and g@delta < -1e-9 and r['forward_clearance']<0)
            alternatives=[float(g@(candidate-q0)) for candidate in desired_candidates]
            r['forward_dot_range']=[min(alternatives),max(alternatives)]
            r['forward_clearance_range']=[r['clearance']+min(alternatives),r['clearance']+max(alternatives)]
            r['dependency_possible_in_receipt_window']=bool(r['clearance']<.005 and min(alternatives)<-1e-9 and r['clearance']+min(alternatives)<0)
        record=dict(query_stamp_ns=stamp,plan_seq=s['plan_seq'],attempt=int(s['safe_stall_attempts']),
                    scp_iter=int(s['scp_iter']),q0=q0.tolist(),forward_target=desired.tolist(),
                    recovery_target=target.tolist(),input_binding=mode,target_match_inf=err,
                    frontier_binding=frontier_binding,compatible_forward_targets=[v.tolist() for v in desired_candidates],
                    frontier_receipt_ns=frontier_receipt,matching_frontier_values=len(matches),
                    axis=axis,projected_forward_norm=projected_norm,
                    k2_rows=k2,tight_points=[r['point'] for r in tight],
                    forward_min_k2_clearance=min(r['forward_clearance'] for r in k2),
                    _desired=desired)
        records.append(record); previous_target=target; previous_desired=desired
    cpp_path=out/'cpp_inputs.json'; cpp_path.write_text(json.dumps(cpp_inputs,indent=2)+'\n')
    process=subprocess.run([str(args.binary.resolve()),str(cpp_path)],text=True,capture_output=True,check=True)
    cpp=json.loads(process.stdout); assert len(cpp)==len(cpp_inputs)
    cpp_errors=[float(np.max(abs(np.array(actual)-item['expected']))) for actual,item in zip(cpp,cpp_inputs)]
    assert max(cpp_errors)<1e-11
    for r in records: del r['_desired']
    exact_records=[r for r in records if r['frontier_binding']=='unique_recorded_target_match']
    assert len(exact_records)==45
    projection_ratios=[r['projected_forward_norm']/np.linalg.norm(np.array(r['forward_target'])-r['q0'])
                       for r in records if r['attempt']>=3 and r['projected_forward_norm'] is not None]
    point_summary=[]
    for p in sorted({key(row['point']) for r in records for row in r['k2_rows']}):
        rr=[row for r in exact_records for row in r['k2_rows'] if key(row['point'])==p]
        point_summary.append(dict(point=p,source='UNKNOWN',margin=rr[0]['margin'],
                                  exact_forward_records=len(rr),tight_records=sum(row['clearance']<.005 for row in rr),
                                  clearance_range=[min(row['clearance'] for row in rr),max(row['clearance'] for row in rr)],
                                  forward_dot_range=[min(row['forward_dot'] for row in rr),max(row['forward_dot'] for row in rr)]))
    fields=['query_stamp_ns','plan_seq','attempt','scp_iter','frontier_binding','point','margin','at_start',
            'clearance','forward_dot','forward_clearance','detour_dot','dependency_condition']
    with (out/'k2_constraint_rows.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in records:
            for row in r['k2_rows']:
                writer.writerow({k:(row[k] if k in row else r[k]) for k in fields})
    summary=dict(solved_records=len(records),plans=len(plan_rows),cpp_function_replays=len(cpp),
                 cpp_max_target_error=max(cpp_errors),binding_counts=Counter(r['input_binding'] for r in records),
                 tight_point_counts=Counter(str(key(p)) for r in records for p in r['tight_points']),
                 dependency_conditions=sum(row['dependency_condition'] for r in records for row in r['k2_rows']),
                 all_tight_forward_improving=all(row['forward_dot']>0 for r in records for row in r['k2_rows'] if row['clearance']<.005),
                 all_tight_forward_improving_across_compatible_inputs=all(row['forward_dot_range'][0]>0 for r in records for row in r['k2_rows'] if row['clearance']<.005),
                 frontier_binding_counts=Counter(r['frontier_binding'] for r in records),
                 possible_dependency_records=sum(any(row['dependency_possible_in_receipt_window'] for row in r['k2_rows']) for r in records),
                 all_k2_forward_feasible_records=sum(r['forward_min_k2_clearance']>=0 for r in records),
                 exact_forward_all_k2_feasible_records=sum(r['forward_min_k2_clearance']>=0 for r in exact_records),
                 conservative_receipt_window_all_k2_feasible_records=sum(all(row['forward_clearance_range'][0]>=0 for row in r['k2_rows']) for r in records),
                 attempts_3_to_5_projected_forward_norm_ratio=[min(projection_ratios),max(projection_ratios)],
                 axes_by_attempt={a:dict(Counter(r['axis'] for r in records if r['attempt']==a and r['axis'] is not None)) for a in range(2,6)})
    result=dict(batch=BATCH,window_ros_s=[19.107,22.518],numpy_version=np.__version__,summary=summary,margin_events=margin_events,records=records,
                point_summary_exact_frontier_binding=point_summary,
                diagnostic_rows_recovered_from_log=recovered_rows,
                limitations=['Replays unchanged production tangent function and recorded row classification, not a full PIQP solve or simulator replay.',
                             'Frontier topic has no source stamp: unique compatible input is identified by target reproduction for 24 plans and reused for their SCP iterations. Four plans retain old recovery targets: their copied forward frontier cannot be uniquely recovered; report ranges over all recorded inputs within -200/+20 ms of first query.',
                             'Positive directional derivative and k=2 linear feasibility do not certify a nonlinear continuous trajectory. All normal safety gates remain required.'],
                input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},binary_sha256=digest(args.binary.resolve()))
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
