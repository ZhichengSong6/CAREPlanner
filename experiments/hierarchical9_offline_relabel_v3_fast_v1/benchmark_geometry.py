#!/usr/bin/env python3
"""Numerical parity and timing benchmark: RepoOracle autograd geometry vs analytic kinematics."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
V3=HERE.parent/"hierarchical9_offline_relabel_v3"
for p in (V3,HERE):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import common as v3
from analytic_geometry import AnalyticGeometry

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--repo",type=Path,required=True)
    ap.add_argument("--urdf",type=Path,required=True);ap.add_argument("--samples",type=int,default=512);a=ap.parse_args()
    src,spec,bank,q,digest=v3.open_run(a.source)
    oracle=v3.RepoOracle(a.repo,a.urdf,"cpu",joint_names=bank.joints,sensor_frames=bank.sensors)
    rng=np.random.default_rng(260926);pairs=[]
    ids=rng.choice(np.unique(q["x_index"]),min(32,len(np.unique(q["x_index"]))),replace=False)
    for _ in range(a.samples):
        xi=int(rng.choice(ids));s=int(rng.integers(0,8));qq=rng.uniform(bank.lo,bank.hi);pairs.append((xi,s,qq))
    old=[];new=[];max_h=0.;max_j=0.;rel_j=0.
    t=time.perf_counter()
    for xi,s,qq in pairs:old.append(oracle.geometry(np.asarray(bank.x[xi],float),s)(qq))
    old_s=time.perf_counter()-t
    cache={}
    t=time.perf_counter()
    for xi,s,qq in pairs:
        key=(xi,s)
        if key not in cache:cache[key]=AnalyticGeometry(oracle,np.asarray(bank.x[xi],float),s)
        new.append(cache[key](qq))
    new_s=time.perf_counter()-t
    for (h0,j0),(h1,j1) in zip(old,new):
        max_h=max(max_h,float(np.max(np.abs(h0-h1))))
        max_j=max(max_j,float(np.max(np.abs(j0-j1))))
        rel_j=max(rel_j,float(np.linalg.norm(j0-j1)/max(np.linalg.norm(j0),1e-12)))
    rep=dict(samples=len(pairs),margin_max_abs=max_h,jacobian_max_abs=max_j,jacobian_relative_max=rel_j,
             old_seconds=old_s,analytic_seconds=new_s,speedup=old_s/new_s)
    print(json.dumps(rep,indent=2))
    if max_h>1e-10 or max_j>1e-9 or rel_j>1e-9:
        raise SystemExit("ANALYTIC_GEOMETRY_PARITY_FAILED")
    print("[PASS] analytic geometry parity")
if __name__=="__main__":main()
