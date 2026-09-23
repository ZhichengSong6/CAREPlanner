#!/usr/bin/env python3
"""Frozen-start solver comparison: original R1 vs paired-trained new_value."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from collections import Counter
from pathlib import Path
import numpy as np,torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
R012=REPO/"experiments/hierarchical9_scratch50k_r012_v1"
PAIR=REPO/"experiments/hierarchical9_paired_training_v1"
AUDIT=REPO/"experiments/hierarchical9_boundary_audit_v1"
SCRIPTS=REPO/"src/care_visibility_cdf/scripts"
for p in (PAIR,AUDIT,SCRIPTS,R012):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import common as pair_common
import runtime_probe
import model as r012_model
from hierarchical_visibility_cdf_model import HierarchicalSensorView
from train_signed_visibility_cdf_pairwise_replace import VisibilityQ0Dataset,DEFAULT_JOINT_NAMES,DEFAULT_SENSOR_FRAMES
from oracle import SensorOracle

NAMES=("R1","NEW_VALUE")

def sha256(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def fraction(k,n):return {"passed":int(k),"count":int(n),"rate":float(k/n) if n else None}
def exact_p(a,b):
 n=a+b
 if not n:return 1.0
 k=min(a,b)
 return min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
def aggregate(rows):
 out={}
 for cohort,chosen in (("local",[r for r in rows if "/local_" in r["group"]]),("uniform",[r for r in rows if r["group"].endswith("uniform_outside")]),("all",rows)):
  out[cohort]={n:fraction(sum(bool(r["models"][n]["fov_pass"]) for r in chosen),len(chosen)) for n in NAMES}
 return out
def per_sensor(rows):
 return {f"S{s}":{n:fraction(sum(bool(r["models"][n]["fov_pass"]) for r in rows if r["sensor"]==s),sum(r["sensor"]==s for r in rows)) for n in NAMES} for s in range(8)}
def paired(rows):
 c=Counter()
 for r in rows:
  a=bool(r["models"]["NEW_VALUE"]["fov_pass"]);b=bool(r["models"]["R1"]["fov_pass"])
  c["both_pass" if a and b else "NEW_VALUE_only" if a else "R1_only" if b else "both_fail"]+=1
 ao,bo=c["NEW_VALUE_only"],c["R1_only"]
 return {**dict(c),"net_NEW_VALUE_minus_R1":ao-bo,"exact_sign_p":exact_p(ao,bo)}
def failures(rows,n):
 return dict(Counter(r["models"][n]["failure_stage"] for r in rows))
def load_candidate(path,device):
 cp=torch.load(path,map_location="cpu",weights_only=False)
 if cp.get("format")!="care_h9_paired_training_v1" or cp.get("arm")!="new_value" or not cp.get("completed"):
  raise ValueError("candidate must be completed paired-training new_value checkpoint")
 if cp.get("parent_sha256")!=pair_common.R1_SHA:raise ValueError("candidate parent is not fixed R1")
 m=r012_model.build_model("R1");m.load_state_dict(cp["model_state"],strict=True)
 return m.to(device).eval().requires_grad_(False),sha256(path)
def markdown(r):
 f=lambda x:"N/A" if x is None else f"{x:.5f}"
 lines=["# Frozen fresh-holdout solver comparison: R1 vs NEW_VALUE","",f"Starts SHA256: {r['starts_sha256']}. Reused exactly; no resampling.","",
 "| cohort | N | R1 | NEW_VALUE |","|---|---:|---:|---:|"]
 for c in ("local","uniform","all"):
  x=r["aggregate"][c];lines.append(f"| {c} | {x['R1']['count']} | {f(x['R1']['rate'])} | {f(x['NEW_VALUE']['rate'])} |")
 p=r["paired"];lines+=["","## Paired outcomes","",f"NEW_VALUE-only: {p.get('NEW_VALUE_only',0)}; R1-only: {p.get('R1_only',0)}; net: {p['net_NEW_VALUE_minus_R1']}; exact sign p={p['exact_sign_p']:.6f}.","","## Failure stages",""]
 for n in NAMES:lines.append(f"- {n}: {json.dumps(r['failures'][n],sort_keys=True)}")
 lines+=["","## Per sensor",""]
 for s in range(8):
  x=r["per_sensor"][f"S{s}"];lines.append(f"- S{s}: R1={f(x['R1']['rate'])}, NEW_VALUE={f(x['NEW_VALUE']['rate'])}, N={x['R1']['count']}")
 lines+=["","FOV-only learned-branch solver comparison. LOS/collision/trajectory/actual-seen are NOT_RUN; this does not promote a production checkpoint."]
 return "\n".join(lines)+"\n"

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--r012-root",type=Path,required=True);ap.add_argument("--candidate",type=Path,required=True);ap.add_argument("--starts",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--rank",type=int,default=0);ap.add_argument("--world",type=int,default=1);ap.add_argument("--merge",action="store_true");a=ap.parse_args()
 out=a.output.resolve()
 if a.merge:
  rows=[]
  for r in range(a.world):
   p=out/f"solves.rank{r}.jsonl"
   rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  rows.sort(key=lambda x:x["case_id"]);starts=[json.loads(x) for x in a.starts.read_text().splitlines() if x.strip()]
  if [r["case_id"] for r in rows]!=list(range(len(starts))):raise ValueError("incomplete/duplicate rank shards")
  m=json.loads((out/"manifest.json").read_text());rep={"status":"COMPLETE","starts_sha256":m["starts_sha256"],"count":len(rows),"aggregate":aggregate(rows),"per_sensor":per_sensor(rows),"paired":paired(rows),"failures":{n:failures(rows,n) for n in NAMES}}
  pair_common.write_json(out/"report.json",rep);(out/"summary.md").write_text(markdown(rep));m["status"]="COMPLETE";pair_common.write_json(out/"manifest.json",m);print((out/"summary.md").read_text());return
 if not torch.cuda.is_available():raise RuntimeError("CUDA required")
 if not 0<=a.rank<a.world:raise ValueError("bad rank/world")
 device=torch.device("cuda",0);torch.cuda.set_device(0);torch.set_num_threads(2)
 r1,rcp,rsha=pair_common.load_r1(a.r012_root,device,False);cand,csha=load_candidate(a.candidate,device)
 data=Path(rcp["args"]["data"]);urdf=Path(rcp["args"]["urdf"]);dataset=VisibilityQ0Dataset(str(data),val_count=int(rcp["args"].get("val_count",1000)),seed=int(rcp["args"].get("seed",0)))
 lo,hi=dataset.q_limits(device);masks=dataset.sensor_masks(device);oracle=SensorOracle(urdf,device,DEFAULT_JOINT_NAMES,DEFAULT_SENSOR_FRAMES)
 probes={"R1":runtime_probe.make_probe(HierarchicalSensorView(r1),masks,lo,hi),"NEW_VALUE":runtime_probe.make_probe(HierarchicalSensorView(cand),masks,lo,hi)}
 starts_text=a.starts.read_text();starts=[json.loads(x) for x in starts_text.splitlines() if x.strip()];starts_sha=hashlib.sha256(starts_text.encode()).hexdigest()
 old_manifest=json.loads((a.starts.parent/"manifest.json").read_text())
 if starts_sha!=old_manifest.get("starts_sha256"):raise ValueError("starts hash differs from original fresh holdout manifest")
 if a.rank==0:
  if out.exists():raise FileExistsError(f"No overwrite: {out}")
  out.mkdir(parents=True)
  pair_common.write_json(out/"manifest.json",{"status":"RUNNING","starts_sha256":starts_sha,"start_count":len(starts),"source_starts":str(a.starts.resolve()),"r1_sha256":rsha,"candidate_sha256":csha,"candidate_parent_sha256":pair_common.R1_SHA,"world":a.world,"solver_semantics":"unchanged runtime_probe parameters"})
 while not out.exists():time.sleep(.1)
 p=out/f"solves.rank{a.rank}.jsonl"
 with p.open("w") as f:
  for i in range(a.rank,len(starts),a.world):
   spec=starts[i];x=torch.tensor(spec["x"],device=device,dtype=torch.float32);q=torch.tensor(spec["q_init"],device=device,dtype=torch.float32);s=int(spec["sensor"])
   row={**spec,"case_id":i,"models":{}}
   order=NAMES if i%2==0 else NAMES[::-1]
   for n in order:row["models"][n]=runtime_probe.run_probe(probes[n],oracle,x,q,s)
   f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()
   if ((i-a.rank)//a.world+1)%100==0:print(f"[rank{a.rank}] cases={(i-a.rank)//a.world+1}",flush=True)
 print(f"[done] rank={a.rank} shard={p}",flush=True)
if __name__=="__main__":main()
