#!/usr/bin/env python3
"""Benchmark exact original V3 search policy with analytic geometry, plus analytic hybrid.

Uses the same frozen task stratification as benchmark.py. No training labels are installed.
"""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
V3=HERE.parent/"hierarchical9_offline_relabel_v3"
for p in (V3,HERE):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import common as v3
import fast_core
from analytic_geometry import AnalyticGeometry
from benchmark import select_tasks,cosine
core=fast_core.core

def one_mode(mode,qq,points,mask,bank,x,s,oracle,spec,faces=None):
    cfg=core.SolverConfig(**spec["solver"]);vc=core.VerifyConfig(**spec["verify"])
    geo=AnalyticGeometry(oracle,x,s)
    t=time.perf_counter()
    if mode=="legacy_analytic":
        attempts=core.solve_attempts(qq,points,mask,bank.lo,bank.hi,geo,cfg)
        new=core.screen_candidates(qq,mask,bank.lo,bank.hi,geo,attempts,cfg,vc)
    elif mode=="hybrid_analytic":
        attempts,new=fast_core.solve_fast(qq,points,mask,bank.lo,bank.hi,geo,faces,cfg,vc)
    else: raise ValueError(mode)
    new=v3.reference_gate(oracle,x,s,qq,new,cfg)
    return attempts,new,1000*(time.perf_counter()-t)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--source",type=Path,required=True);ap.add_argument("--repo",type=Path,required=True);ap.add_argument("--urdf",type=Path,required=True)
 ap.add_argument("--output",type=Path,required=True);ap.add_argument("--rank",type=int,default=0);ap.add_argument("--world",type=int,default=1)
 ap.add_argument("--per-stratum",type=int,default=4);ap.add_argument("--seed",type=int,default=260925);ap.add_argument("--merge",action="store_true")
 a=ap.parse_args();out=a.output.resolve()
 if a.merge:
  rows=[]
  for r in range(a.world):
   p=out/f"rank{r:02d}.jsonl";rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  rows.sort(key=lambda z:z["bench_id"])
  if [r["bench_id"] for r in rows]!=list(range(len(rows))):raise ValueError("incomplete shards")
  rep={"status":"COMPLETE","tasks":len(rows),"modes":{}}
  for mode in ("legacy_analytic","hybrid_analytic"):
   vv=[r for r in rows if r["old_value_valid"] and r[f"{mode}_value_valid"]]
   dif=np.asarray([abs(r["old_value"]-r[f"{mode}_value"]) for r in vv],float) if vv else np.array([])
   old_ms=np.asarray([r["old_elapsed_ms"] for r in rows],float)
   ms=np.asarray([r[f"{mode}_elapsed_ms"] for r in rows],float)
   mism=[r for r in rows if r["old_value_valid"]!=r[f"{mode}_value_valid"]]
   rep["modes"][mode]=dict(
      validity_agreement=1-len(mism)/len(rows),mismatch_count=len(mism),
      old_valid=sum(r["old_value_valid"] for r in rows),mode_valid=sum(r[f"{mode}_value_valid"] for r in rows),
      value_abs_diff=dict(n=len(dif),median=float(np.median(dif)) if len(dif) else None,p95=float(np.quantile(dif,.95)) if len(dif) else None,max=float(dif.max()) if len(dif) else None),
      timing=dict(worker_s=float(ms.sum()/1000),median_s=float(np.median(ms)/1000),p95_s=float(np.quantile(ms,.95)/1000),
                  speedup_vs_old=float(old_ms.sum()/ms.sum())),
      attempts=sum(r[f"{mode}_attempts"] for r in rows),
      extrapolation_550k=dict(worker_hours=float(ms.mean()/1000*550000/3600),wall_hours_16=float(ms.mean()/1000*550000/3600/16)),
      mismatches=[dict(bench_id=r["bench_id"],query_id=r["query_id"],sensor=r["sensor"],
         old_valid=r["old_value_valid"],mode_valid=r[f"{mode}_value_valid"],old_value=r["old_value"],mode_value=r[f"{mode}_value"])
         for r in mism])
  v3.old_io.write_json(out/"report.json",rep)
  lines=["# V3-fast v2 analytic-solver benchmark","",f"Tasks: {rep['tasks']}.","",
         "| mode | validity agreement | mismatch | median s | p95 s | speedup | 550k wall h @16 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
  for mode,z in rep["modes"].items():
   lines.append(f"| {mode} | {z['validity_agreement']:.6f} | {z['mismatch_count']} | {z['timing']['median_s']:.3f} | {z['timing']['p95_s']:.3f} | {z['timing']['speedup_vs_old']:.2f}x | {z['extrapolation_550k']['wall_hours_16']:.1f} |")
  lines+=[""]
  for mode,z in rep["modes"].items():
   lines += [f"## {mode}",f"Both-valid value |diff|: median={z['value_abs_diff']['median']}, p95={z['value_abs_diff']['p95']}, max={z['value_abs_diff']['max']}.",
             f"Attempts={z['attempts']}; worker-hours@550k={z['extrapolation_550k']['worker_hours']:.1f}.",
             f"Mismatches={json.dumps(z['mismatches'],ensure_ascii=False)}",""]
  (out/"summary.md").write_text("\n".join(lines)+"\n");print((out/"summary.md").read_text());return

 src,spec,bank,q,digest=v3.open_run(a.source);tasks=select_tasks(q,a.per_stratum,a.seed)
 if a.rank==0:
  if out.exists():raise FileExistsError(out)
  out.mkdir(parents=True);v3.old_io.write_json(out/"plan.json",dict(tasks=tasks,source=str(src),spec_sha256=digest,seed=a.seed,per_stratum=a.per_stratum))
 while not out.exists():time.sleep(.05)
 oracle=v3.RepoOracle(a.repo,a.urdf,"cpu",joint_names=bank.joints,sensor_frames=bank.sensors)
 face_cache=fast_core.FaceBankCache(oracle);loc={int(v):i for i,v in enumerate(q["query_id"])}
 path=out/f"rank{a.rank:02d}.jsonl"
 with path.open("w") as f:
  for bid in range(a.rank,len(tasks),a.world):
   qid,s=tasks[bid];i=loc[qid];xi=int(q["x_index"][i]);qq=np.asarray(q["q_query"][i],float);x=np.asarray(bank.x[xi],float);points=bank.sensor_bank(xi,s);mask=bank.masks[s]
   old,_=v3.read_record(v3.record_path(src,"base",qid,s),digest,[qid,s]);oldnew=old["new"]
   faces=face_cache.faces(xi,s,x,points)
   row=dict(bench_id=bid,query_id=qid,sensor=s,x_index=xi,old_value_valid=bool(oldnew["value_valid"]),
            old_value=float(oldnew["value"]) if oldnew["value"] is not None else None,old_elapsed_ms=float(old["elapsed_ms"]),old_attempts=len(old["attempts"]))
   for mode in ("legacy_analytic","hybrid_analytic"):
    attempts,new,elapsed=one_mode(mode,qq,points,mask,bank,x,s,oracle,spec,faces)
    row[f"{mode}_value_valid"]=bool(new["value_valid"]);row[f"{mode}_value"]=float(new["value"]) if new["value"] is not None else None
    row[f"{mode}_elapsed_ms"]=elapsed;row[f"{mode}_attempts"]=len(attempts)
    if mode=="hybrid_analytic":row["hybrid_fallback_planes"]=new["fast_diagnostics"]["fallback_planes"]
   f.write(json.dumps(v3.old_io.clean_json(row),allow_nan=False)+"\n");f.flush()
   print(f"[rank{a.rank}] {bid+1}/{len(tasks)} old={row['old_elapsed_ms']/1000:.2f}s legacyA={row['legacy_analytic_elapsed_ms']/1000:.2f}s hybridA={row['hybrid_analytic_elapsed_ms']/1000:.2f}s",flush=True)

if __name__=="__main__":main()
