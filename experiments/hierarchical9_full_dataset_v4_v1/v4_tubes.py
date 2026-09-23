#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from prod_common import *

def worker(a):
 out,m,bank,plan=open_run(a.out);cfg=Config(**m["config"]);n=len(plan["x_index"]);nsh=shard_count(n,cfg.x_shard_size)
 oracle=RepoOracle(ROOT,a.urdf,"cpu",joint_names=bank.joints,sensor_frames=bank.sensors)
 offsets=(0.,)+tuple(v for r in TUBE_RADII for v in (-r,r))
 for sh in range(a.rank,nsh,a.world):
  apath,aj=stage_paths(out,"v4_anchors",sh);tpath,tj=stage_paths(out,"v4_tubes",sh)
  if apath.exists() and aj.exists() and tpath.exists() and tj.exists():
   if sha256_file(apath)!=read_json(aj)["sha256"] or sha256_file(tpath)!=read_json(tj)["sha256"]:raise RuntimeError("bad v4 shard")
   print(f"[resume] v4 shard {sh}",flush=True);continue
  a0,b0=shard_bounds(n,sh,cfg.x_shard_size);anchors=[];tubes=[];t0=time.perf_counter()
  for row in range(a0,b0):
   xi=int(plan["x_index"][row]);split=int(plan["split"][row]);x=np.asarray(bank.x[xi],float)
   for s in np.flatnonzero(plan["support"][row]):
    s=int(s);geo=AnalyticGeometry(oracle,x,s);slots=select_anchor_slots(bank,xi,s,cfg.anchors_per_sensor)
    for slot in slots:
     q0=np.asarray(bank.q[xi,int(slot),:,s],float);res=refine_anchor(q0,bank.masks[s],bank.lo,bank.hi,geo)
     qstar=np.asarray(res["q_star"],float);normal=np.full(7,np.nan);face=-1;gap=np.nan;jnorm=np.nan;jmargin=np.nan;regular=False
     if res["success"]:
      h,j=geo(qstar);order=np.argsort(h);face=int(order[0]);gap=float(h[order[1]]-h[order[0]]) if len(h)>1 else np.inf
      nn=np.asarray(j[face],float)*bank.masks[s];jnorm=float(np.linalg.norm(nn));active=np.flatnonzero(bank.masks[s])
      jmargin=float(np.minimum(qstar-bank.lo,bank.hi-qstar)[active].min())
      if jnorm>1e-8:normal=nn/jnorm
      regular=bool(gap>1e-4 and jnorm>1e-8 and jmargin>max(TUBE_RADII)+.002)
     anchors.append((xi,split,s,int(slot),q0,qstar,normal,res["success"],regular,face,gap,jnorm,jmargin,res["g"],res["iterations"],res["calls"],res["reason"]))
     if regular:
      for off in offsets:
       q=qstar+off*normal
       if not(np.isfinite(q).all() and np.all(q>=bank.lo) and np.all(q<=bank.hi)):continue
       h,_=geo(q);sf=int(np.argmin(h));gm=float(np.min(h));same=sf==face
       signok=(abs(off)<1e-15 and abs(gm)<=2e-6) or (off>0 and gm>0) or (off<0 and gm<0)
       if same and signok:tubes.append((xi,split,s,int(slot),off,q,normal,gm))
  aa=dict(x_index=np.asarray([r[0] for r in anchors],np.int64),split=np.asarray([r[1] for r in anchors],np.uint8),
    sensor=np.asarray([r[2] for r in anchors],np.uint8),slot=np.asarray([r[3] for r in anchors],np.int32),
    q0=np.asarray([r[4] for r in anchors],np.float32),q_star=np.asarray([r[5] for r in anchors],np.float32),
    normal=np.asarray([r[6] for r in anchors],np.float32),refined=np.asarray([r[7] for r in anchors],bool),
    regular=np.asarray([r[8] for r in anchors],bool),active_face=np.asarray([r[9] for r in anchors],np.int8),
    plane_gap_m=np.asarray([r[10] for r in anchors],np.float32),normal_norm=np.asarray([r[11] for r in anchors],np.float32),
    joint_margin_rad=np.asarray([r[12] for r in anchors],np.float32),boundary_g_m=np.asarray([r[13] for r in anchors],np.float32),
    iterations=np.asarray([r[14] for r in anchors],np.int16),geometry_calls=np.asarray([r[15] for r in anchors],np.int16),
    reason=np.asarray([r[16] for r in anchors],dtype="U32"))
  tt=dict(x_index=np.asarray([r[0] for r in tubes],np.int64),split=np.asarray([r[1] for r in tubes],np.uint8),
    sensor=np.asarray([r[2] for r in tubes],np.uint8),source_slot=np.asarray([r[3] for r in tubes],np.int32),
    value=np.asarray([r[4] for r in tubes],np.float32),q=np.asarray([r[5] for r in tubes],np.float32),
    grad=np.asarray([r[6] for r in tubes],np.float32),g_m=np.asarray([r[7] for r in tubes],np.float32))
  elapsed=time.perf_counter()-t0
  install_shard(out,"v4_anchors",sh,aa,dict(stage="v4_anchors",shard=sh,anchors=len(anchors),elapsed_s=elapsed))
  install_shard(out,"v4_tubes",sh,tt,dict(stage="v4_tubes",shard=sh,tubes=len(tubes),elapsed_s=elapsed))
  print(f"[v4 rank{a.rank}] shard={sh+1}/{nsh} anchors={len(anchors)} tubes={len(tubes)} elapsed={elapsed:.1f}s",flush=True)

def merge(a):
 out,m,bank,plan=open_run(a.out);cfg=Config(**m["config"]);nsh=shard_count(len(plan["x_index"]),cfg.x_shard_size)
 ae=verify_stage(out,"v4_anchors",nsh);te=verify_stage(out,"v4_tubes",nsh);anchors=refined=regular=tubes=0;reason=Counter();by={f"S{s}":[0,0,0,0] for s in range(8)}
 for sh in range(nsh):
  ap,_=stage_paths(out,"v4_anchors",sh);tp,_=stage_paths(out,"v4_tubes",sh)
  with np.load(ap,allow_pickle=False) as z:
   anchors+=len(z["sensor"]);refined+=int(z["refined"].sum());regular+=int(z["regular"].sum());reason.update(z["reason"].tolist())
   for s in range(8):
    mm=z["sensor"]==s;by[f"S{s}"][0]+=int(mm.sum());by[f"S{s}"][1]+=int((mm&z["refined"]).sum());by[f"S{s}"][2]+=int((mm&z["regular"]).sum())
  with np.load(tp,allow_pickle=False) as z:
   tubes+=len(z["sensor"])
   for s in range(8):by[f"S{s}"][3]+=int((z["sensor"]==s).sum())
 summary=dict(status="COMPLETE",anchors=anchors,refined=refined,regular=regular,valid_tubes=tubes,
   refined_fraction=refined/max(anchors,1),regular_fraction=regular/max(anchors,1),reasons=dict(reason),by_sensor=by,
   shard_worker_seconds=float(sum(e["elapsed_s"] for e in ae)))
 write_json(out/"v4_summary.json",summary);print(json.dumps(summary,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--urdf",type=Path,default=ROOT/"src/arm_description/urdf/Arm.urdf")
 ap.add_argument("--rank",type=int,default=0);ap.add_argument("--world",type=int,default=1);ap.add_argument("--merge",action="store_true");a=ap.parse_args()
 if a.merge:merge(a)
 else:worker(a)
if __name__=="__main__":main()
