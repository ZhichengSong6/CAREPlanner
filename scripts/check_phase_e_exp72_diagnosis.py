#!/usr/bin/env python3
"""Independent algebraic checks of saved exp72 diagnostic evidence."""
import argparse
import json
from pathlib import Path
import numpy as np

def matrix(m):
    a=np.zeros(m['shape'])
    for i,j,v in m['coo']:a[int(i),int(j)]+=v
    return a

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence-dir',type=Path,required=True)
    ap.add_argument('--mechanism-dir',type=Path,required=True);args=ap.parse_args()
    read=lambda p:json.loads(p.read_text())
    r=read(args.evidence_dir/'report.json');m=read(args.mechanism_dir/'report.json')
    assert not r['unverified_plans'] and len(r['plans'])==27
    checks={}
    for n,info in r['plans'].items():
        p=read(args.evidence_dir/n/'problem.json');s=read(args.evidence_dir/n/'solution.json');x=np.array(s['x'])
        P,A,G=[matrix(p[k]) for k in ('P','A','G')]
        primal=max(np.max(np.abs(A@x-p['b'])),np.max(np.array(p['h_l'])-G@x),np.max(G@x-p['h_u']),np.max(np.array(p['x_l'])-x),np.max(x-p['x_u']))
        station=P@x+p['c']+A.T@np.array(s['y'])+G.T@(np.array(s['z_u'])-s['z_l'])+np.array(s['z_bu'])-s['z_bl']
        assert primal<1e-8 and np.max(np.abs(station))<1e-3
        assert info['iterations']==info['logged_iterations'] and info['step_error']<1e-7
        matches=[z for z in info['matches'] if z['shape_error']<1e-8 and z['velocity_error']<1e-7]
        assert len(matches)==1 and matches[0]['tracker_records']>0 and not matches[0]['tracker_aborted']
        assert m['originals'][n]['nonlinear']['violated_over_1e5']==0
        checks[n]=dict(primal=float(primal),stationarity_inf=float(np.max(np.abs(station))))
    for c in r['cycles']:
        assert c['seen_fraction_max']==0 and c['net_displacement_inf']<.001
    for label,t in m['qvis_targets'].items():
        assert t['violating_point_count']==1 and t['cdf_min_margin_residual']<-.04
        assert t['ideal_fov']['inside_any_fov_count']==5
        projected=m['projected_endpoint_diagnostics'][label]
        assert projected['optimizer_success'] and projected['min_endpoint_cdf_residual']>0
        assert projected['ideal_fov']['inside_any_fov_count']==5
        assert projected['sampled_straight_path_min_cdf_residual']<-.09
    for c in m['counterfactuals'].values():
        for v in c['variants'].values():
            assert max(v['original_constraints'].values())<1e-8 and v['nonlinear']['violated_over_1e5']==0
    tail=m['tail_hold_objective_diagnostic']
    assert tail['prefix_brake_hold_span']<.032 and tail['nonlinear']['violated_over_1e5']==0
    assert max(tail['original_constraints'].values())<1e-8
    with (args.mechanism_dir/'independent_checks.json').open('x') as f:json.dump(checks,f,indent=2)
    print('PASS: 27 QP KKT/trajectory bindings, six return cycles, nonlinear original/counterfactual rows, endpoint/path distinction and tail diagnostic')

if __name__=='__main__':main()
