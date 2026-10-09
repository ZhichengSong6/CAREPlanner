#!/usr/bin/env python3
"""Full frozen fresh holdout with a scale-aware V2 continuous-distance solver."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from collections import Counter
from pathlib import Path
import torch

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
import core as audit_core
import model as r012_model
from hierarchical_visibility_cdf_model import HierarchicalSensorView
from train_signed_visibility_cdf_pairwise_replace import VisibilityQ0Dataset,DEFAULT_JOINT_NAMES,DEFAULT_SENSOR_FRAMES
from oracle import SensorOracle

NAMES=("R1","V2_50K")

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

def cohort_rows(rows,cohort):
    if cohort=="local":return [r for r in rows if "/local_" in r["group"]]
    if cohort=="uniform":return [r for r in rows if r["group"].endswith("uniform_outside")]
    if cohort=="all":return rows
    raise ValueError(cohort)

def aggregate(rows):
    out={}
    for c in ("local","uniform","all"):
        chosen=cohort_rows(rows,c)
        out[c]={n:fraction(sum(bool(r["models"][n]["fov_pass"]) for r in chosen),len(chosen)) for n in NAMES}
    return out

def per_sensor(rows):
    return {f"S{s}":{n:fraction(sum(bool(r["models"][n]["fov_pass"]) for r in rows if r["sensor"]==s),sum(r["sensor"]==s for r in rows)) for n in NAMES} for s in range(8)}

def paired(rows):
    c=Counter()
    for r in rows:
        a=bool(r["models"]["V2_50K"]["fov_pass"]);b=bool(r["models"]["R1"]["fov_pass"])
        c["both_pass" if a and b else "V2_50K_only" if a else "R1_only" if b else "both_fail"]+=1
    ao,bo=c["V2_50K_only"],c["R1_only"]
    return {**dict(c),"net_V2_50K_minus_R1":ao-bo,"exact_sign_p":exact_p(ao,bo)}

def failures(rows,n):return dict(Counter(r["models"][n]["failure_stage"] for r in rows))

def load_v2(path,device):
    path=path.resolve()
    cp=torch.load(path,map_location="cpu",weights_only=False)
    if cp.get("format")!="care_h9_v4_scratch_v2" or not cp.get("completed"):
        raise ValueError("candidate must be completed care_h9_v4_scratch_v2 final checkpoint")
    if int(cp.get("step",-1))!=50000 or cp.get("initialization")!="random_from_scratch":
        raise ValueError("candidate must be formal random-init 50k")
    run=json.loads((path.parent/"run.json").read_text())
    digest=sha256(path)
    if run.get("status")!="COMPLETE" or int(run.get("successful_updates",-1))!=50000 or run.get("final_sha256")!=digest:
        raise ValueError("candidate run provenance mismatch")
    m=r012_model.build_model("R1");m.load_state_dict(cp["model_state"],strict=True)
    return m.to(device).eval().requires_grad_(False),digest,run

def markdown(r):
    f=lambda x:"N/A" if x is None else f"{x:.5f}"
    lines=["# Frozen fresh-holdout: R1 legacy solver vs V2-50K scale-aware solver","",f"Starts SHA256: {r['starts_sha256']}. Reused exactly; no resampling.","",
           "| cohort | N | R1 | V2-50K | delta pp |","|---|---:|---:|---:|---:|"]
    for c in ("local","uniform","all"):
        x=r["aggregate"][c];a=x["R1"]["rate"];b=x["V2_50K"]["rate"];d=None if a is None or b is None else 100*(b-a)
        lines.append(f"| {c} | {x['R1']['count']} | {f(a)} | {f(b)} | {'N/A' if d is None else f'{d:+.3f}'} |")
    lines+=["","## Paired outcomes by cohort","","| cohort | V2-only | R1-only | net V2-R1 | exact sign p |","|---|---:|---:|---:|---:|"]
    for c in ("local","uniform","all"):
        p=r["paired"][c]
        lines.append(f"| {c} | {p.get('V2_50K_only',0)} | {p.get('R1_only',0)} | {p['net_V2_50K_minus_R1']} | {p['exact_sign_p']:.6g} |")
    lines+=["","## Failure stages",""]
    for n in NAMES:lines.append(f"- {n}: {json.dumps(r['failures'][n],sort_keys=True)}")
    lines+=["","## Per sensor",""]
    for s in range(8):
        x=r["per_sensor"][f"S{s}"];a=x["R1"]["rate"];b=x["V2_50K"]["rate"];d=None if a is None or b is None else 100*(b-a)
        lines.append(f"- S{s}: R1={f(a)}, V2-50K={f(b)}, delta_pp={'N/A' if d is None else f'{d:+.3f}'}, N={x['R1']['count']}")
    lines+=["","FOV-only learned-branch solver comparison on the exact frozen fresh starts. LOS/collision/trajectory/actual-seen are NOT_RUN."]
    return "\n".join(lines)+"\n"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--r012-root",type=Path,required=True);ap.add_argument("--candidate",type=Path,required=True);ap.add_argument("--starts",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--rank",type=int,default=0);ap.add_argument("--world",type=int,default=1);ap.add_argument("--merge",action="store_true")
    a=ap.parse_args();out=a.output.resolve()
    if a.merge:
        rows=[]
        for r in range(a.world):
            p=out/f"solves.rank{r}.jsonl";rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        rows.sort(key=lambda x:x["case_id"])
        starts=[json.loads(x) for x in a.starts.read_text().splitlines() if x.strip()]
        if [r["case_id"] for r in rows]!=list(range(len(starts))):raise ValueError("incomplete/duplicate rank shards")
        m=json.loads((out/"manifest.json").read_text())
        rep={"status":"COMPLETE","starts_sha256":m["starts_sha256"],"count":len(rows),"aggregate":aggregate(rows),"per_sensor":per_sensor(rows),"paired":{c:paired(cohort_rows(rows,c)) for c in ("local","uniform","all")},"failures":{n:failures(rows,n) for n in NAMES}}
        pair_common.write_json(out/"report.json",rep);(out/"summary.md").write_text(markdown(rep));m["status"]="COMPLETE";pair_common.write_json(out/"manifest.json",m);print((out/"summary.md").read_text());return
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda",0);torch.cuda.set_device(0);torch.set_num_threads(2)
    r1,rcp,rsha=pair_common.load_r1(a.r012_root,device,False);cand,csha,crun=load_v2(a.candidate,device)
    data=Path(rcp["args"]["data"]);urdf=Path(rcp["args"]["urdf"]);dataset=VisibilityQ0Dataset(str(data),val_count=int(rcp["args"].get("val_count",1000)),seed=int(rcp["args"].get("seed",0)))
    lo,hi=dataset.q_limits(device);masks=dataset.sensor_masks(device);oracle=SensorOracle(urdf,device,DEFAULT_JOINT_NAMES,DEFAULT_SENSOR_FRAMES)
    probes={"R1":runtime_probe.make_probe(HierarchicalSensorView(r1),masks,lo,hi),"V2_50K":runtime_probe.make_probe(HierarchicalSensorView(cand),masks,lo,hi)}\n    # V2 is trained in continuous signed-distance units; use the existing root tolerance\n    # as the projection stopping tolerance instead of the legacy R1 score-scale epsilon.\n    probes["V2_50K"].projection_epsilon_f=probes["V2_50K"].root_tolerance_f
    starts_text=a.starts.read_text();starts=[json.loads(x) for x in starts_text.splitlines() if x.strip()];starts_sha=hashlib.sha256(starts_text.encode()).hexdigest()
    old_manifest=json.loads((a.starts.parent/"manifest.json").read_text())
    if starts_sha!=old_manifest.get("starts_sha256"):raise ValueError("starts hash differs from original fresh holdout")
    if a.rank==0:
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
        pair_common.write_json(out/"manifest.json",{"status":"RUNNING","starts_sha256":starts_sha,"start_count":len(starts),"source_starts":str(a.starts.resolve()),"r1_sha256":rsha,"candidate_sha256":csha,"candidate_code_sha":crun["code_sha"],"candidate_successful_updates":crun["successful_updates"],"world":a.world,"solver_semantics":"R1 legacy runtime_probe; V2 projection_epsilon_f=root_tolerance_f=0.002; all other runtime_probe parameters unchanged; exact original frozen starts"})
    while not out.exists():time.sleep(.1)
    p=out/f"solves.rank{a.rank}.jsonl"
    with p.open("w") as f:
        for i in range(a.rank,len(starts),a.world):
            spec=starts[i];x=torch.tensor(spec["x"],device=device,dtype=torch.float32);q=torch.tensor(spec["q_init"],device=device,dtype=torch.float32);s=int(spec["sensor"])
            row={**spec,"case_id":i,"models":{}};order=NAMES if i%2==0 else NAMES[::-1]
            for n in order:row["models"][n]=runtime_probe.run_probe(probes[n],oracle,x,q,s)
            f.write(json.dumps(audit_core.json_safe(row),allow_nan=False)+"\n");f.flush()
            if ((i-a.rank)//a.world+1)%100==0:print(f"[rank{a.rank}] cases={(i-a.rank)//a.world+1}",flush=True)
    print(f"[done] rank={a.rank} shard={p}",flush=True)

if __name__=="__main__":main()
