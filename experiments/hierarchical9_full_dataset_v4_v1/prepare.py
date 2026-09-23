#!/usr/bin/env python3
from __future__ import annotations
import argparse,math
from dataclasses import asdict
from pathlib import Path
import numpy as np
from prod_common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--fresh-starts",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
 ap.add_argument("--candidate-q-per-x",type=int,default=128);ap.add_argument("--v3-q-per-x",type=int,default=4);ap.add_argument("--anchors-per-sensor",type=int,default=32);ap.add_argument("--x-shard-size",type=int,default=128);ap.add_argument("--seed",type=int,default=260927)
 a=ap.parse_args();cfg=Config(a.candidate_q_per_x,a.v3_q_per_x,a.anchors_per_sensor,a.x_shard_size,a.seed);cfg.validate()
 source,sm,bank,_=cache_io.open_job(a.source);out=a.out.resolve()
 if out.exists():raise FileExistsError(out)
 fresh=fresh_x_indices(a.fresh_starts)
 val_count=int(sm["sampling"]["val_count"]);split_seed=int(sm["sampling"]["split_seed"])
 tr,va=bank.original_split(val_count,split_seed);freshset=set(fresh.tolist())
 rows=[];valid_counts=[]
 for split,pool in ((0,tr),(1,va)):
  for xi in pool:
   if int(xi) in freshset:continue
   vc=np.asarray(bank.valid[int(xi)],bool).sum(axis=0).astype(np.uint16)
   rows.append((int(xi),split));valid_counts.append(vc)
 x_index=np.asarray([r[0] for r in rows],np.int64);split=np.asarray([r[1] for r in rows],np.uint8)
 valid_count=np.asarray(valid_counts,np.uint16);support=valid_count>0
 order=np.argsort(x_index,kind="stable");x_index=x_index[order];split=split[order];valid_count=valid_count[order];support=support[order]
 out.mkdir(parents=True)
 write_npz(out/"x_plan.npz",dict(x_index=x_index,split=split,valid_count=valid_count,support=support,fresh_excluded_x=fresh))
 anchors=int(np.minimum(valid_count,cfg.anchors_per_sensor).sum())
 supported_pairs=int(support.sum());n=len(x_index);nsh=shard_count(n,cfg.x_shard_size)
 manifest=dict(format=FORMAT,status="PREPARED",source=str(Path(a.source).resolve()),fresh_starts=str(a.fresh_starts.resolve()),
   original_data_sha256=sm["original_data_sha256"],config=asdict(cfg),solver=asdict(core.SolverConfig()),verify=asdict(core.VerifyConfig()),
   original_split=dict(val_count=val_count,split_seed=split_seed),x_plan_sha256=sha256_file(out/"x_plan.npz"),
   code_identity=code_identity(),counts=dict(x=n,train_x=int((split==0).sum()),val_x=int((split==1).sum()),fresh_excluded_x=len(fresh),
      supported_x_sensor=supported_pairs,unresolved_x_sensor=int(n*8-supported_pairs),x_shards=nsh,
      global_xq=n*cfg.candidate_q_per_x,global_sensor_sign_labels=n*cfg.candidate_q_per_x*8,
      v3_xq=n*cfg.v3_q_per_x,v3_query_sensor_tasks=supported_pairs*cfg.v3_q_per_x,
      v4_boundary_anchors=anchors,v4_tube_candidates=anchors*(1+2*len(TUBE_RADII))),
   semantics=dict(support="old bank has >=1 valid q0; missing bank is UNRESOLVED, never proof of infeasibility",
      v3="original V3 six-face/original-seed SLSQP and screening, analytic geometry only",
      v4="existing boundary q0 coverage -> deterministic active-joint FPS anchors -> strict zero refinement -> local normal tubes",
      global_sign="analytic FOV sign for all 8 sensors; no distance target implied"),
   training_ready=False)
 write_json(out/"manifest.json",manifest)
 print("[prepared]",out)
 print(json.dumps(manifest["counts"],indent=2))
if __name__=="__main__":
 import json;main()
