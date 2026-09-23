#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from prod_common import *

def reason_code(new):
 if new.get("reference_check") is not None and not new["reference_check"].get("passed",False):return "REFERENCE_GATE"
 if new["selected"] is None:return "NO_FEASIBLE"
 if not new["selected"]["qualified"]:return "NOT_QUALIFIED"
 if abs(float(new["query_g_m"]))<=core.SolverConfig().sign_guard_m and float(new["value"] or 0)!=0:return "SIGN_UNRESOLVED"
 return "VALID" if new["value_valid"] else "OTHER_INVALID"

def worker(a):
 out,m,bank,plan=open_run(a.out);cfg=Config(**m["config"]);solver=core.SolverConfig(**m["solver"]);vc=core.VerifyConfig(**m["verify"])
 nsh=shard_count(len(plan["x_index"]),cfg.x_shard_size)
 oracle=RepoOracle(ROOT,a.urdf,"cpu",joint_names=bank.joints,sensor_frames=bank.sensors)
 for sh in range(a.rank,nsh,a.world):
  op,oj=stage_paths(out,"v3_labels",sh)
  if op.exists() and oj.exists():
   if sha256_file(op)!=read_json(oj)["sha256"]:raise RuntimeError(op)
   print(f"[resume] v3 shard {sh}",flush=True);continue
  gp,_=stage_paths(out,"global_pool",sh)
  if not gp.is_file():raise FileNotFoundError(gp)
  with np.load(gp,allow_pickle=False) as z:g={k:z[k] for k in z.files}
  rows=[];t0=time.perf_counter()
  for i,xi in enumerate(g["x_index"]):
   x=np.asarray(bank.x[int(xi)],float);support=g["support"][i];qidx=g["v3_selected_index"][i]
   for s in np.flatnonzero(support):
    s=int(s);points=bank.sensor_bank(int(xi),s);mask=bank.masks[s];geo=AnalyticGeometry(oracle,x,s)
    for slot,cidx in enumerate(qidx):
     q=np.asarray(g["q_pool"][i,int(cidx)],float);ref=float(g["g_pool"][i,int(cidx),s])
     old=core.legacy.old_bank_label(q,points,mask,1 if ref>=0 else -1)
     ts=time.perf_counter();attempts=core.solve_attempts(q,points,mask,bank.lo,bank.hi,geo,solver)
     new=core.screen_candidates(q,mask,bank.lo,bank.hi,geo,attempts,solver,vc);new=v3.reference_gate(oracle,x,s,q,new,solver)
     elapsed=1000*(time.perf_counter()-ts);cand=np.asarray(new["gradient_candidate"],float)
     blockers=[r for r in new["gradient_reasons"] if r!="DISTANCE_DERIVATIVE_NOT_VERIFIED"]
     rows.append(dict(x_index=int(xi),split=int(g["split"][i]),q_slot=slot,pool_index=int(cidx),sensor=s,q=q,
       reference_g_m=ref,old_value=float(old["value"]),new_value=float(new["value"]) if new["value"] is not None else np.nan,
       value_valid=bool(new["value_valid"]),q_star=np.asarray(new["q_star"],float),gradient_candidate=cand,
       gradient_candidate_valid=bool(new["value_valid"] and np.isfinite(cand).all() and not blockers),
       ambiguity=bool(new["ambiguity"]),uncertain=bool(new["uncertain_competitor"]),reason=reason_code(new),
       selected_plane=int(new["selected"]["plane"]) if new["selected"] is not None else -1,
       distance_rad=float(new["distance_rad"]) if new["distance_rad"] is not None else np.nan,
       stationarity=float(new["selected"]["stationarity_relative"]) if new["selected"] is not None else np.nan,
       attempts=len(attempts),successes=sum(bool(x.get("success")) for x in attempts),
       geometry_calls=sum(int(x.get("geometry_calls",0)) for x in attempts),elapsed_ms=elapsed))
  if rows:
   arr={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
   for k in ("q","q_star","gradient_candidate"):arr[k]=arr[k].astype(np.float32)
   for k in ("reference_g_m","old_value","new_value","distance_rad","stationarity","elapsed_ms"):arr[k]=arr[k].astype(np.float32)
  else:raise RuntimeError("empty v3 shard")
  install_shard(out,"v3_labels",sh,arr,dict(stage="v3_labels",shard=sh,tasks=len(rows),elapsed_s=time.perf_counter()-t0))
  print(f"[v3 rank{a.rank}] shard={sh+1}/{nsh} tasks={len(rows)} elapsed={time.perf_counter()-t0:.1f}s",flush=True)

def merge(a):
 out,m,bank,plan=open_run(a.out);cfg=Config(**m["config"]);nsh=shard_count(len(plan["x_index"]),cfg.x_shard_size);entries=verify_stage(out,"v3_labels",nsh)
 total=valid=0;ms=[];reasons=Counter();by={f"S{s}":[0,0] for s in range(8)}
 for sh in range(nsh):
  p,_=stage_paths(out,"v3_labels",sh)
  with np.load(p,allow_pickle=False) as z:
   total+=len(z["sensor"]);valid+=int(z["value_valid"].sum());ms.extend(z["elapsed_ms"].astype(float).tolist());reasons.update(z["reason"].tolist())
   for s in range(8):
    mask=z["sensor"]==s;by[f"S{s}"][0]+=int(mask.sum());by[f"S{s}"][1]+=int((mask&z["value_valid"]).sum())
 ms=np.asarray(ms,float)
 summary=dict(status="COMPLETE",tasks=total,valid_values=valid,valid_fraction=valid/max(total,1),reasons=dict(reasons),by_sensor=by,
   median_task_s=float(np.median(ms)/1000),p95_task_s=float(np.quantile(ms,.95)/1000),worker_hours=float(ms.sum()/3.6e6),
   shard_worker_seconds=float(sum(e["elapsed_s"] for e in entries)))
 write_json(out/"v3_summary.json",summary);print(json.dumps(summary,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--urdf",type=Path,default=ROOT/"src/arm_description/urdf/Arm.urdf")
 ap.add_argument("--rank",type=int,default=0);ap.add_argument("--world",type=int,default=1);ap.add_argument("--merge",action="store_true");a=ap.parse_args()
 if a.merge:merge(a)
 else:worker(a)
if __name__=="__main__":main()
