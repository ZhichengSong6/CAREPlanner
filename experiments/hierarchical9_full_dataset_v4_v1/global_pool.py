#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from prod_common import *

def worker(a):
 out,m,bank,plan=open_run(a.out);cfg=Config(**m["config"]);n=len(plan["x_index"]);nsh=shard_count(n,cfg.x_shard_size)
 oracle=RepoOracle(ROOT,a.urdf,a.device,joint_names=bank.joints,sensor_frames=bank.sensors)
 done=0
 for sh in range(a.rank,nsh,a.world):
  p,j=stage_paths(out,"global_pool",sh)
  if p.exists() and j.exists():
   old=read_json(j)
   if sha256_file(p)!=old["sha256"]:raise RuntimeError(p)
   print(f"[resume] global shard {sh}",flush=True);continue
  lo,hi=bank.lo,bank.hi;a0,b0=shard_bounds(n,sh,cfg.x_shard_size)
  xis=plan["x_index"][a0:b0];spl=plan["split"][a0:b0];sup=plan["support"][a0:b0]
  qs=[];gs=[];sel=[];avail=[];cover=[];t0=time.perf_counter()
  for k,xi in enumerate(xis):
   rng=np.random.default_rng(task_seed(cfg.seed,int(xi)))
   q=rng.uniform(lo,hi,size=(cfg.candidate_q_per_x,7)).astype(np.float32)
   g=oracle.reference_margins(np.asarray(bank.x[int(xi)],float),q).astype(np.float32)
   si,av,co=select_v3_indices(q,g,sup[k],cfg.v3_q_per_x,lo,hi)
   qs.append(q);gs.append(g);sel.append(si);avail.append(av);cover.append(co)
  arr=dict(x_index=xis.astype(np.int64),split=spl.astype(np.uint8),support=sup.astype(bool),
      q_pool=np.asarray(qs,np.float32),g_pool=np.asarray(gs,np.float32),v3_selected_index=np.asarray(sel,np.int16),
      sign_state_available=np.asarray(avail,bool),sign_state_covered=np.asarray(cover,bool))
  install_shard(out,"global_pool",sh,arr,dict(stage="global_pool",shard=sh,x_start=a0,x_stop=b0,rows=len(xis),elapsed_s=time.perf_counter()-t0))
  print(f"[global rank{a.rank}] shard={sh+1}/{nsh} x={len(xis)} elapsed={time.perf_counter()-t0:.2f}s",flush=True);done+=1
 return done

def merge(a):
 out,m,bank,plan=open_run(a.out);cfg=Config(**m["config"]);nsh=shard_count(len(plan["x_index"]),cfg.x_shard_size);entries=verify_stage(out,"global_pool",nsh)
 counts=np.zeros((8,2),np.int64);selected=np.zeros((8,2),np.int64);available=np.zeros((8,2),np.int64);covered=np.zeros((8,2),np.int64)
 for sh in range(nsh):
  p,_=stage_paths(out,"global_pool",sh)
  with np.load(p,allow_pickle=False) as z:
   g=z["g_pool"];idx=z["v3_selected_index"];sup=z["support"]
   for s in range(8):
    counts[s,0]+=int((g[:,:,s]<0).sum());counts[s,1]+=int((g[:,:,s]>=0).sum())
    for i in range(len(g)):
     selected[s,0]+=int((g[i,idx[i],s]<0).sum());selected[s,1]+=int((g[i,idx[i],s]>=0).sum())
   available += z["sign_state_available"].sum(axis=0);covered += z["sign_state_covered"].sum(axis=0)
 summary=dict(status="COMPLETE",shards=nsh,global_sign_counts=counts,selected_v3_sign_counts=selected,
   supported_state_available=available,supported_state_covered=covered,
   supported_state_coverage=float(covered.sum()/max(available.sum(),1)),worker_seconds=float(sum(e["elapsed_s"] for e in entries)))
 write_json(out/"global_pool_summary.json",summary);print(json.dumps(old_io.clean_json(summary),indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--urdf",type=Path,default=ROOT/"src/arm_description/urdf/Arm.urdf")
 ap.add_argument("--device",default="cpu");ap.add_argument("--rank",type=int,default=0);ap.add_argument("--world",type=int,default=1);ap.add_argument("--merge",action="store_true");a=ap.parse_args()
 if a.merge:merge(a)
 else:worker(a)
if __name__=="__main__":
 old_io=cache_io;main()
