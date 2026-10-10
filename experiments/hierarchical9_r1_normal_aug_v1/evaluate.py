#!/usr/bin/env python3
"""Frozen 1963-case FOV-only R1 vs R1+normal-direction comparison.

Exact legacy runtime_probe (epsilon_f=0.03), no parameter tuning, no optimizer.
Analytic SensorOracle supplies actual FOV pass; not trajectory/LOS certification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import Counter
import torch

HERE=Path(__file__).resolve().parent
REPO=Path(os.environ.get("R1A_REPO",str(HERE.parents[1]))).resolve()
R012=REPO/"experiments/hierarchical9_scratch50k_r012_v1"
PAIR=REPO/"experiments/hierarchical9_paired_training_v1"
AUDIT=REPO/"experiments/hierarchical9_boundary_audit_v1"
SCRIPTS=REPO/"src/care_visibility_cdf/scripts"
for location in (PAIR,AUDIT,SCRIPTS,R012):
    if str(location) not in sys.path:
        sys.path.insert(0,str(location))
import common as pair_common
import runtime_probe
import core as audit_core
import model as r012_model
from hierarchical_visibility_cdf_model import HierarchicalSensorView
from train_signed_visibility_cdf_pairwise_replace import VisibilityQ0Dataset,DEFAULT_JOINT_NAMES,DEFAULT_SENSOR_FRAMES
from oracle import SensorOracle
from train import FORMAT, baseline, dependency_fingerprints, sha, json_read, json_atomic

NAMES=("R1","R1_AUG")
EXPECTED_STARTS_SHA="0e4711fc21628fdbb341191ec634415781696800a2386a4ad48e592c34f3198c"


def fraction(rows,arm):
    passed=sum(bool(row["models"][arm]["fov_pass"]) for row in rows)
    return {"pass":passed,"count":len(rows),"rate":passed/len(rows) if rows else None}


def groups(rows):
    return {"local":[r for r in rows if "/local_" in r["group"]],
            "uniform":[r for r in rows if r["group"].endswith("uniform_outside")],
            "all":rows}


def report(rows,starts_sha):
    by=groups(rows)
    aggregates={g:{name:fraction(subset,name) for name in NAMES} for g,subset in by.items()}
    baseline_counts=aggregates
    if (len(rows)!=1963 or baseline_counts["local"]["R1"]["pass"]!=618 or
        baseline_counts["uniform"]["R1"]["pass"]!=673 or
        baseline_counts["all"]["R1"]["pass"]!=1291):
        raise RuntimeError("Frozen original R1 solver baseline NOT reproduced")
    gains=sum(r["models"]["R1_AUG"]["fov_pass"] and not r["models"]["R1"]["fov_pass"] for r in rows)
    losses=sum(r["models"]["R1"]["fov_pass"] and not r["models"]["R1_AUG"]["fov_pass"] for r in rows)
    per_sensor={}
    for s in range(8):
        subset=[r for r in rows if r["sensor"]==s]
        per_sensor[f"S{s}"]={n:fraction(subset,n) for n in NAMES}
    failure={n:dict(Counter(r["models"][n]["failure_stage"] for r in rows)) for n in NAMES}
    return {"status":"COMPLETE","starts_sha256":starts_sha,"count":len(rows),
        "eps_f":0.03,"aggregate":aggregates,"per_sensor":per_sensor,
        "paired":{"aug_only":gains,"R1_only":losses,"net_gain":gains-losses},
        "failures":failure,"scope":"fixed FOV-only solver; no LOS/GCDF/VBC/actual seen"}


def summary(r):
    lines=["# R1 vs R1 + analytic boundary-normal direction (controlled scratch 50k)","",
        "Same original frozen 1963 starts, unchanged legacy runtime and epsilon_f=0.03.",
        "Original R1 must exactly reproduce 618/683 local, 673/1280 uniform, 1291/1963 all.","",
        "| Cohort | N | R1 | R1 + boundary normal | Δ pp |",
        "|---|---:|---:|---:|---:|"]
    for key in ("local","uniform","all"):
        a,b=r["aggregate"][key]["R1"],r["aggregate"][key]["R1_AUG"]
        lines.append(f"| {key} | {a['count']} | {a['rate']:.5f} | {b['rate']:.5f} | {100*(b['rate']-a['rate']):+.2f} |")
    p=r["paired"]
    lines += ["",f"Paired: augment-only={p['aug_only']}, R1-only={p['R1_only']}, net={p['net_gain']:+d}.","",
        "## Per sensor","", "| Sensor | N | R1 | Augmented | Δ pp |",
        "|---|---:|---:|---:|---:|"]
    for s in range(8):
        a,b=r["per_sensor"][f"S{s}"]["R1"],r["per_sensor"][f"S{s}"]["R1_AUG"]
        lines.append(f"| S{s} | {a['count']} | {a['rate']:.5f} | {b['rate']:.5f} | {100*(b['rate']-a['rate']):+.2f} |")
    lines += ["","## Failure stages","",
        "R1: "+json.dumps(r["failures"]["R1"],ensure_ascii=False,sort_keys=True),
        "",
        "Aug: "+json.dumps(r["failures"]["R1_AUG"],ensure_ascii=False,sort_keys=True),
        "","Reused starts are a regression set; no independent fresh generalization test.",
        "No parameter tuning in this evaluator; analytic FOV pass is not execution certification."]
    return "\n".join(lines)+"\n"


def load_candidate(root,device,expected_sha):
    p=root/"final.pt"
    r=json_read(root/"run.json")
    if (r.get("status")!="COMPLETE" or r.get("successful_updates")!=50000 or
        not r.get("baseline_stream_exact_match") or
        r.get("variant")!=FORMAT or sha(p)!=r.get("final_sha256") or
        r.get("training_stream_sha256")!=expected_sha):
        raise ValueError("Augmented final checkpoint/run mismatch")
    cp=torch.load(p,map_location="cpu",weights_only=False)
    if (cp.get("format")!=FORMAT or cp.get("step")!=50000 or
        cp.get("initialization")!="random_from_scratch" or
        cp.get("source_sha256")!=dependency_fingerprints()):
        raise ValueError("Augmented checkpoint code/provenance mismatch")
    model=r012_model.build_model("R1")
    model.load_state_dict(cp["model_state"],strict=True)
    return model.to(device).eval().requires_grad_(False),r["final_sha256"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--r012-root",type=Path,required=True)
    parser.add_argument("--aug-root",type=Path,required=True)
    parser.add_argument("--starts",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--world",type=int,default=4)
    parser.add_argument("--merge",action="store_true")
    a=parser.parse_args()
    out=a.output.resolve()
    frozen=a.starts.read_text()
    if (hashlib.sha256(frozen.encode()).hexdigest()!=EXPECTED_STARTS_SHA or
        json_read(a.starts.parent/"manifest.json").get("starts_sha256")!=EXPECTED_STARTS_SHA):
        raise ValueError("Frozen starts hash mismatch")
    starts=[json.loads(line) for line in frozen.splitlines() if line.strip()]
    if len(starts)!=1963:
        raise ValueError("Wrong frozen start count")
    if a.merge:
        rows=[]
        for rank in range(a.world):
            file=out/f"solves.rank{rank}.jsonl"
            rows.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
        rows.sort(key=lambda z:z["case_id"])
        if [x["case_id"] for x in rows]!=list(range(1963)):
            raise ValueError("Incomplete/duplicate solver cases")
        rep=report(rows,EXPECTED_STARTS_SHA)
        json_atomic(out/"report.json",rep)
        (out/"summary.md").write_text(summary(rep))
        json_atomic(out/"complete.json",{"status":"COMPLETE","rows":1963,
                                       "comparison":"fixed R1 vs augmented eps 0.03"})
        print((out/"summary.md").read_text(),flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for formal solver evaluation")
    device=torch.device("cuda",0)
    torch.cuda.set_device(0);torch.set_num_threads(2)
    r1,rcp,rsha=pair_common.load_r1(a.r012_root,device,False)
    baseline_run=baseline(a.r012_root)
    aug,augsha=load_candidate(a.aug_root,device,baseline_run["training_stream_sha256"])
    ds=VisibilityQ0Dataset(str(rcp["args"]["data"]),
                            val_count=int(rcp["args"].get("val_count",1000)),
                            seed=int(rcp["args"].get("seed",0)))
    lo,hi=ds.q_limits(device)
    masks=ds.sensor_masks(device)
    oracle=SensorOracle(Path(rcp["args"]["urdf"]),device,
                        DEFAULT_JOINT_NAMES,DEFAULT_SENSOR_FRAMES)
    probes={
      "R1":runtime_probe.make_probe(HierarchicalSensorView(r1),masks,lo,hi),
      "R1_AUG":runtime_probe.make_probe(HierarchicalSensorView(aug),masks,lo,hi),
    }
    if any(probe.projection_epsilon_f!=0.03 for probe in probes.values()):
        raise RuntimeError("Legacy solver epsilon changed")
    if a.rank==0:
        if (out/"manifest.json").exists():
            raise FileExistsError("Existing evaluation manifest, no overwrite")
        json_atomic(out/"manifest.json",{"status":"RUNNING","source_starts":str(a.starts),
          "starts_sha256":EXPECTED_STARTS_SHA,"r1_sha256":rsha,"aug_sha256":augsha,
          "solver":"unchanged runtime_probe, epsilon_f=0.03","world":a.world})
    file=out/f"solves.rank{a.rank}.jsonl"
    if file.exists():raise FileExistsError(file)
    with file.open("w") as f:
        count=0
        for i in range(a.rank,len(starts),a.world):
            spec=starts[i]
            x=torch.tensor(spec["x"],device=device,dtype=torch.float32)
            q=torch.tensor(spec["q_init"],device=device,dtype=torch.float32)
            s=int(spec["sensor"])
            row={**spec,"case_id":i,"models":{}}
            names=NAMES if i%2==0 else NAMES[::-1]
            for name in names:
                row["models"][name]=runtime_probe.run_probe(probes[name],oracle,x,q,s)
            f.write(json.dumps(audit_core.json_safe(row),allow_nan=False)+"\n")
            count+=1
            if count%100==0:
                f.flush()
                print(f"[solver rank{a.rank}] solved={count}",flush=True)
    print(f"[done] rank{a.rank} cases={count}",flush=True)

if __name__=="__main__":
    main()
