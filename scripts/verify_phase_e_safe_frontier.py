#!/usr/bin/env python3
"""Replay production tangent targets in the validated exp71 QP, no ROS graph."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import numpy as np
import diagnose_phase_e_case001_exact_witness as d


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--target-tool',type=Path,required=True)
    ap.add_argument('--solver',type=Path,required=True)
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    read=lambda n:json.loads((args.evidence_dir/n).read_text())
    p=read('baseline_problem.json');r=read('report.json')
    q0=np.array(r['q0']);old=np.array(r['frontier']);g=np.array(r['active_rows'][0]['gradient'])
    classes=d.continuity_methods();geom=d.Geometry()
    rows=read('selected_rows.json')
    points=np.array(list({d.solver_point_key(json.loads(x['point'])):json.loads(x['point']) for x in rows}.values()))
    summary={}
    for attempt in range(1,5):
        command=[str(args.target_tool),'--target',str(attempt)]+[str(x) for v in (q0,old,g) for x in v]
        target=np.array(json.loads(subprocess.run(command,capture_output=True,text=True,check=True).stdout))
        assert target.shape==(7,) and np.max(np.abs(target-q0))<=.030000001
        assert abs(g@(target-q0))<1e-8
        cp=copy.deepcopy(p);c=np.array(cp['c']);c[7:14]+=6000*(old-target);cp['c']=c.tolist()
        for key in p:
            if key!='c': assert cp[key]==p[key],key
        sol=d.solve(args.solver,cp);assert sol['status']=='solved'
        x=np.array(sol['x']);q=np.vstack((q0,x[:140].reshape(20,7)));u=x[140:].reshape(20,7)
        residual=d.residuals(p,x)
        assert max(residual.values())<1e-8
        assert np.max(np.abs(q[2]-q0))>.01
        msg=classes[0]._repair_prefix_with_braking_tail(d.trajectory(q,u,classes))
        ds=np.array([geom.minimum(s['q'],points)[0] for s in geom.sweep(msg)])
        summary[attempt]=dict(target=target.tolist(),q2_delta_inf=float(np.max(np.abs(q[2]-q0))),
            original_constraints=residual,local_exact_min_m=float(ds.min()),
            full_gate='NOT_RUN_missing_frozen_map',nonlinear_cdf='REQUIRES_SCP_REQUERY')
        d.save(args.output_dir/f'attempt_{attempt}_problem.json',cp)
        d.save(args.output_dir/f'attempt_{attempt}_solution.json',sol)
    d.save(args.output_dir/'report.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
