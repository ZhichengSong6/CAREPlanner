#!/usr/bin/env python3
"""Benchmark accelerated V3 on frozen completed base labels; no new label cache is installed."""
from __future__ import annotations
import argparse,json,math,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
V3=HERE.parent/"hierarchical9_offline_relabel_v3"
for p in (V3,HERE):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import common as v3
import fast_core
core=fast_core.core

def select_tasks(q,per,seed):
    rng=np.random.default_rng(seed);out=[]
    for split in (0,1):
      for group in np.unique(q["query_group"]):
       for s in range(8):
        g=q["reference_g_m"][:,s]
        for sign in (-1,1):
          ids=np.flatnonzero((q["split"]==split)&(q["query_group"]==group)&q["support"][:,s]&((g<0) if sign<0 else (g>=0)))
          if len(ids):
            for i in rng.permutation(ids)[:per]:out.append((int(q["query_id"][i]),s))
    # Stable unique order.
    seen=set();return [x for x in out if not (x in seen or seen.add(x))]

def cosine(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.linalg.norm(a)<1e-12 or np.linalg.norm(b)<1e-12:return None
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--repo",type=Path,required=True);ap.add_argument("--urdf",type=Path,required=True)
 ap.add_argument("--output",type=Path,required=True);ap.add_argument("--rank",type=int,default=0);ap.add_argument("--world",type=int,default=1);ap.add_argument("--per-stratum",type=int,default=4);ap.add_argument("--seed",type=int,default=260925)
 ap.add_argument("--merge",action="store_true");a=ap.parse_args()
 out=a.output.resolve()
 if a.merge:
  rows=[]
  for r in range(a.world):
   p=out/f"rank{r:02d}.jsonl"
   rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  rows.sort(key=lambda z:z["bench_id"])
  if [r["bench_id"] for r in rows]!=list(range(len(rows))):raise ValueError("benchmark shards incomplete")
  both=[r for r in rows if r["old_value_valid"] and r["fast_value_valid"]]
  vd=np.asarray([abs(r["old_value"]-r["fast_value"]) for r in both],float) if both else np.array([])
  old_ms=np.asarray([r["old_elapsed_ms"] for r in rows],float);fast_ms=np.asarray([r["fast_elapsed_ms"] for r in rows],float)
  rep=dict(status="COMPLETE",tasks=len(rows),
    validity_agreement=sum(r["old_value_valid"]==r["fast_value_valid"] for r in rows)/len(rows),
    old_valid=sum(r["old_value_valid"] for r in rows),fast_valid=sum(r["fast_value_valid"] for r in rows),
    value_abs_diff=dict(n=len(vd),median=float(np.median(vd)) if len(vd) else None,p95=float(np.quantile(vd,.95)) if len(vd) else None,max=float(vd.max()) if len(vd) else None),
    qstar_close_1e6=sum((r["qstar_l2"] is not None and r["qstar_l2"]<=1e-6) for r in both),
    grad_cosine=dict(n=sum(r["grad_cosine"] is not None for r in both),median=float(np.median([r["grad_cosine"] for r in both if r["grad_cosine"] is not None])) if any(r["grad_cosine"] is not None for r in both) else None),
    timing=dict(old_worker_s=float(old_ms.sum()/1000),fast_worker_s=float(fast_ms.sum()/1000),
      speedup_total=float(old_ms.sum()/fast_ms.sum()),old_median_s=float(np.median(old_ms)/1000),fast_median_s=float(np.median(fast_ms)/1000),
      fast_p95_s=float(np.quantile(fast_ms,.95)/1000)),
    attempts=dict(old=sum(r["old_attempts"] for r in rows),fast=sum(r["fast_attempts"] for r in rows),
      fast_fallback_tasks=sum(bool(r["fallback_planes"]) for r in rows)),
    by_sensor={})
  for s in range(8):
   rr=[r for r in rows if r["sensor"]==s]
   if rr:rep["by_sensor"][f"S{s}"]=dict(n=len(rr),validity_agreement=sum(r["old_value_valid"]==r["fast_value_valid"] for r in rr)/len(rr),
     speedup=sum(r["old_elapsed_ms"] for r in rr)/sum(r["fast_elapsed_ms"] for r in rr))
  v3.old_io.write_json(out/"report.json",rep)
  lines=["# V3-fast parity benchmark","",f"Tasks: {rep['tasks']}; validity agreement: {rep['validity_agreement']:.6f}.",
    f"Both-valid value |diff| median={rep['value_abs_diff']['median']}, p95={rep['value_abs_diff']['p95']}, max={rep['value_abs_diff']['max']}.",
    f"Old worker time={rep['timing']['old_worker_s']:.1f}s; fast={rep['timing']['fast_worker_s']:.1f}s; total speedup={rep['timing']['speedup_total']:.2f}x.",
    f"Median/task old={rep['timing']['old_median_s']:.2f}s; fast={rep['timing']['fast_median_s']:.2f}s; fast p95={rep['timing']['fast_p95_s']:.2f}s.",
    f"Attempts old={rep['attempts']['old']}; fast={rep['attempts']['fast']}; fallback tasks={rep['attempts']['fast_fallback_tasks']}.","",
    "| sensor | n | validity agreement | speedup |","|---|---:|---:|---:|"]
  for s,z in rep["by_sensor"].items():lines.append(f"| {s} | {z['n']} | {z['validity_agreement']:.6f} | {z['speedup']:.2f}x |")
  (out/"summary.md").write_text("\n".join(lines)+"\n");print((out/"summary.md").read_text());return
 src,spec,bank,q,digest=v3.open_run(a.source)
 tasks=select_tasks(q,a.per_stratum,a.seed)
 if a.rank==0:
  if out.exists():raise FileExistsError(out)
  out.mkdir(parents=True);v3.old_io.write_json(out/"plan.json",dict(tasks=tasks,source=str(src),spec_sha256=digest,per_stratum=a.per_stratum,seed=a.seed))
 while not out.exists():time.sleep(.05)
 oracle=v3.RepoOracle(a.repo,a.urdf,"cpu",joint_names=bank.joints,sensor_frames=bank.sensors)
 face_cache=fast_core.FaceBankCache(oracle);loc={int(v):i for i,v in enumerate(q["query_id"])}
 path=out/f"rank{a.rank:02d}.jsonl"
 with path.open("w") as f:
  for bid in range(a.rank,len(tasks),a.world):
   qid,s=tasks[bid];i=loc[qid];xi=int(q["x_index"][i]);qq=np.asarray(q["q_query"][i],float);x=np.asarray(bank.x[xi],float);points=bank.sensor_bank(xi,s)
   old,_=v3.read_record(v3.record_path(src,"base",qid,s),digest,[qid,s]);oldnew=old["new"]
   faces=face_cache.faces(xi,s,x,points);geo=oracle.geometry(x,s);t=time.perf_counter()
   attempts,new=fast_core.solve_fast(qq,points,bank.masks[s],bank.lo,bank.hi,geo,faces,
      core.SolverConfig(**spec["solver"]),core.VerifyConfig(**spec["verify"]))
   new=v3.reference_gate(oracle,x,s,qq,new,core.SolverConfig(**spec["solver"]));elapsed=1000*(time.perf_counter()-t)
   both=bool(oldnew["value_valid"] and new["value_valid"]);qdist=None;gc=None
   if both:
    qdist=float(np.linalg.norm(np.asarray(oldnew["q_star"],float)-np.asarray(new["q_star"],float)))
    gc=cosine(oldnew["gradient_candidate"],new["gradient_candidate"])
   row=dict(bench_id=bid,query_id=qid,sensor=s,x_index=xi,old_value_valid=bool(oldnew["value_valid"]),fast_value_valid=bool(new["value_valid"]),
     old_value=float(oldnew["value"]) if oldnew["value"] is not None else None,fast_value=float(new["value"]) if new["value"] is not None else None,
     qstar_l2=qdist,grad_cosine=gc,old_elapsed_ms=float(old["elapsed_ms"]),fast_elapsed_ms=elapsed,old_attempts=len(old["attempts"]),fast_attempts=len(attempts),
     fallback_planes=new["fast_diagnostics"]["fallback_planes"],memo_calls=new["fast_diagnostics"]["memo_geometry_calls"],memo_hits=new["fast_diagnostics"]["memo_hits"])
   f.write(json.dumps(v3.old_io.clean_json(row),allow_nan=False)+"\n");f.flush()
   print(f"[rank{a.rank}] {bid+1}/{len(tasks)} fast={elapsed/1000:.2f}s attempts={len(attempts)} fallback={row['fallback_planes']}",flush=True)

if __name__=="__main__":
 main()
