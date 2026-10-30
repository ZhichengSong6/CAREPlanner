#!/usr/bin/env python3
"""Read-only fixed-model diagnostic; no changed bounds, safety or model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src/care_collision_cdf/scripts'))
from collision_cdf_model import CollisionCDF


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    args=ap.parse_args(); torch.set_num_threads(1)
    ck=REPO/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'
    model=CollisionCDF(str(ck),device='cpu',activation='gelu')
    src=REPO/'outputs/phase_e_fix34_tests/regression_final/report.json'
    results=[]
    for case in json.loads(src.read_text())['cases']:
        if case['case'][-3:] not in ('004','015'): continue
        snap=case['snapshots'][-1];d=json.loads((REPO/snap['path']).read_text())
        p=torch.tensor(d['point_flat']).reshape(-1,3)
        q=torch.tensor(d['q_linearization_flat']).reshape(-1,7)
        old,g=model.pair_distance_and_gradient(p,q)
        measured=torch.tensor(d['b'][:7]).repeat(len(p),1)
        fresh,gfresh=model.pair_distance_and_gradient(p,measured)
        indices=[pair for _,pair in d['selected_pairs']]
        saved=np.asarray(d['distance']);grad=np.asarray(d['gradient_flat']).reshape(-1,7)
        # The snapshot's q box is un-clipped here: half-width 0.05 radians.
        fresh_linear_max=fresh+gfresh.abs().sum(dim=1)*.05
        row_margin=[]
        for row,pair in d['selected_pairs']:
            gl=np.asarray(d['gradient_flat'][pair*7:(pair+1)*7]);ql=np.asarray(d['q_linearization_flat'][pair*7:(pair+1)*7])
            row_margin.append(float(d['h_l'][row])+saved[pair]-float(gl@ql))
        result=dict(case=case['case'],snapshot=snap['path'],
            value_reproduction_max_error=float(np.max(np.abs(old.numpy()-saved))),
            gradient_reproduction_max_error=float(np.max(np.abs(g.numpy()-grad))),
            old_min=float(old[indices].min()),measured_min_same_points=float(fresh[indices].min()),
            same_point_measured_box_conflicts=int((fresh_linear_max[indices]<torch.tensor(row_margin)-1e-7).sum()),
            max_reference_ahead_q=float((q-measured).abs().max()),
            unchanged_safety_margin_min=min(row_margin),unchanged_safety_margin_max=max(row_margin))
        assert result['value_reproduction_max_error']<1e-4 and result['gradient_reproduction_max_error']<1e-4
        results.append(result)
    report=dict(checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),cases=results,
                limitation='Same historical points only. New seed requires fresh online pair selection and full certification; no execution success claim.')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
