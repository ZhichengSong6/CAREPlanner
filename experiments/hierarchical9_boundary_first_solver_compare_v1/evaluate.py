#!/usr/bin/env python3
"""Complete formal paired FOV-only comparison: frozen R1 vs boundary-first best/final, fixed and scale-aware."""
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

NAMES=("R1","BF_BEST","BF_FINAL","BF_BEST_002","BF_FINAL_002")

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

def paired(rows, candidate):
    c=Counter()
    for r in rows:
        a=bool(r["models"][candidate]["fov_pass"])
        b=bool(r["models"]["R1"]["fov_pass"])
        c["both_pass" if a and b else "candidate_only" if a else "R1_only" if b else "both_fail"]+=1
    a,b=c["candidate_only"],c["R1_only"]
    return {**dict(c),"net_candidate_minus_R1":a-b,"exact_sign_p":exact_p(a,b)}

def failures(rows,n):return dict(Counter(r["models"][n]["failure_stage"] for r in rows))

def load_boundary(root, filename, device):
    root=root.resolve()
    path=root/filename
    run=json.loads((root/"run.json").read_text())
    if run.get("status")!="COMPLETE" or run.get("successful_updates")!=50000:
        raise ValueError("Boundary-first formal training must be COMPLETE @50000")
    if run.get("initialization")!="random_from_scratch":
        raise ValueError("Not a scratch checkpoint")
    digest=sha256(path)
    expect=run["final_sha256"] if filename=="final.pt" else run["best_val_sha256"]
    guard=json.loads(path.with_suffix(path.suffix+".sha256.json").read_text())
    if digest!=expect or digest!=guard["sha256"]:
        raise ValueError("Boundary-first checkpoint provenance/hash mismatch: "+filename)
    cp=torch.load(path,map_location="cpu",weights_only=False)
    if cp.get("format")!="care_h9_boundary_first_scratch_v1":
        raise ValueError("Wrong candidate format")
    if cp.get("initialization")!="random_from_scratch":
        raise ValueError("Wrong candidate initialization")
    expected_step=50000 if filename=="final.pt" else run["best_step"]
    if cp.get("step")!=expected_step or not 1<=int(expected_step)<=50000:
        raise ValueError("Wrong best/final checkpoint step")
    if cp.get("cache_identity")!=run["cache_identity"] or cp.get("protocol")!=run["protocol"]:
        raise ValueError("Boundary-first candidate metadata mismatch")
    if cp.get("package_identity")!=run["package_identity"]:
        raise ValueError("Package identity mismatch")
    m=r012_model.build_model("R1")
    m.load_state_dict(cp["model_state"],strict=True)
    return m.to(device).eval().requires_grad_(False),digest,cp["step"],run

def markdown(rep):
    f=lambda x: "N/A" if x is None else f"{x:.5f}"
    lines=["# Complete paired FOV solver comparison: R1 vs Boundary-first scratch v1","",
           f"Frozen starts: {rep['starts_sha256']} (no resampling).", "",
           "Primary comparison uses identical runtime solver settings, including epsilon_f=0.03.",
           "BF_*_002 are prespecified score-scale sensitivity checks (epsilon_f=0.002) and NOT equal-solver comparisons.", "",
           "| Cohort | N | R1 .03 | Best .03 | Final .03 | Best .002 | Final .002 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for cohort in ("local","uniform","all"):
        a=rep["aggregate"][cohort]
        lines.append("| "+cohort+" | "+str(a["R1"]["count"])+" | "+" | ".join(
            f(a[n]["rate"]) for n in NAMES)+" |")
    lines += ["","## Paired outcomes against frozen R1","",
              "| Cohort | Candidate | Candidate-only | R1-only | Net | Exact sign p |",
              "|---|---|---:|---:|---:|---:|"]
    for cohort in ("local","uniform","all"):
        for cand in NAMES[1:]:
            p=rep["paired"][cohort][cand]
            lines.append(f"| {cohort} | {cand} | {p.get('candidate_only',0)} | {p.get('R1_only',0)} | {p['net_candidate_minus_R1']} | {p['exact_sign_p']:.6g} |")
    lines += ["","## Failure stages","",""]
    for n in NAMES:
        lines.append(f"- {n}: {json.dumps(rep['failures'][n],sort_keys=True)}")
    lines += ["","## Per sensor","","| Sensor | N | R1 .03 | Best .03 | Final .03 | Best .002 | Final .002 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for sid in range(8):
        row=rep["per_sensor"][f"S{sid}"]
        lines.append(f"| S{sid} | {row['R1']['count']} | "+
                     " | ".join(f(row[n]["rate"]) for n in NAMES)+" |")
    lines += ["","FOV-only learned branch comparison. Primitive LOS, GCDF, VBC, trajectory and actual seen: NOT_RUN.",
              "A learned candidate is never a safety certificate. These reused starts are now a regression set, not untouched final validation."]
    return "\n".join(lines)+"\n"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--r012-root",type=Path,required=True)
    ap.add_argument("--bf-root",type=Path,required=True)
    ap.add_argument("--starts",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--rank",type=int,default=0)
    ap.add_argument("--world",type=int,default=1)
    ap.add_argument("--merge",action="store_true")
    a=ap.parse_args()
    out=a.output.resolve()
    if a.merge:
        rows=[]
        for rank in range(a.world):
            file=out/f"solves.rank{rank}.jsonl"
            rows.extend(json.loads(x) for x in file.read_text().splitlines() if x.strip())
        rows.sort(key=lambda r:r["case_id"])
        starts=[json.loads(x) for x in a.starts.read_text().splitlines() if x.strip()]
        if [r["case_id"] for r in rows]!=list(range(len(starts))):
            raise ValueError("Duplicate or missing rank shards")
        manifest=json.loads((out/"manifest.json").read_text())
        rep={"status":"COMPLETE","starts_sha256":manifest["starts_sha256"],
             "count":len(rows),"model_sha256":manifest["model_sha256"],
             "model_steps":manifest["model_steps"],
             "aggregate":aggregate(rows),"per_sensor":per_sensor(rows),
             "paired":{c:{n:paired(cohort_rows(rows,c),n) for n in NAMES[1:]}
                 for c in ("local","uniform","all")},
             "failures":{n:failures(rows,n) for n in NAMES}}
        # Reproduce the exactly frozen prior baseline to detect a changed solver/oracle.
        base=rep["aggregate"]
        if not (len(rows)==1963 and base["local"]["R1"]["passed"]==618
                and base["uniform"]["R1"]["passed"]==673
                and base["all"]["R1"]["passed"]==1291):
            raise ValueError("Old R1 baseline was not reproduced exactly")
        pair_common.write_json(out/"report.json",rep)
        (out/"summary.md").write_text(markdown(rep))
        manifest["status"]="COMPLETE"
        pair_common.write_json(out/"manifest.json",manifest)
        print((out/"summary.md").read_text(),flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device=torch.device("cuda",0)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    r1,rcp,rsha=pair_common.load_r1(a.r012_root,device,False)
    best,bsha,bstep,brun=load_boundary(a.bf_root,"best_val.pt",device)
    final,fsha,fstep,frun=load_boundary(a.bf_root,"final.pt",device)
    data=Path(rcp["args"]["data"])
    urdf=Path(rcp["args"]["urdf"])
    dataset=VisibilityQ0Dataset(str(data),val_count=int(rcp["args"].get("val_count",1000)),
                                seed=int(rcp["args"].get("seed",0)))
    lo,hi=dataset.q_limits(device)
    masks=dataset.sensor_masks(device)
    oracle=SensorOracle(urdf,device,DEFAULT_JOINT_NAMES,DEFAULT_SENSOR_FRAMES)
    probes={
        "R1":runtime_probe.make_probe(HierarchicalSensorView(r1),masks,lo,hi),
        "BF_BEST":runtime_probe.make_probe(HierarchicalSensorView(best),masks,lo,hi),
        "BF_FINAL":runtime_probe.make_probe(HierarchicalSensorView(final),masks,lo,hi),
        "BF_BEST_002":runtime_probe.make_probe(HierarchicalSensorView(best),masks,lo,hi),
        "BF_FINAL_002":runtime_probe.make_probe(HierarchicalSensorView(final),masks,lo,hi),
    }
    # Primary fixed-solvers: R1/BF_BEST/BF_FINAL all untouched defaults (.03).
    # Secondary: BF-only .002, prespecified from prior V2 scale finding. No sweep.
    probes["BF_BEST_002"].projection_epsilon_f=0.002
    probes["BF_FINAL_002"].projection_epsilon_f=0.002
    for name in ("R1","BF_BEST","BF_FINAL"):
        if probes[name].projection_epsilon_f!=0.03:
            raise RuntimeError("Fixed primary solver settings changed")
    startstext=a.starts.read_text()
    starts=[json.loads(x) for x in startstext.splitlines() if x.strip()]
    digest=hashlib.sha256(startstext.encode()).hexdigest()
    frozen_manifest=json.loads((a.starts.parent/"manifest.json").read_text())
    if digest!=frozen_manifest["starts_sha256"] or digest!="0e4711fc21628fdbb341191ec634415781696800a2386a4ad48e592c34f3198c":
        raise ValueError("Frozen fresh starts identity mismatch")
    if len(starts)!=1963:
        raise ValueError("Expected 1963 frozen cases")
    if a.rank==0:
        if (out/"manifest.json").exists():
            raise FileExistsError("Run manifest already exists")
        pair_common.write_json(out/"manifest.json",
            {"status":"RUNNING","starts_sha256":digest,"count":len(starts),"world":a.world,
             "r012_root":str(a.r012_root.resolve()),"bf_root":str(a.bf_root.resolve()),
             "source_starts":str(a.starts.resolve()),"model_sha256":{"R1":rsha,"BF_BEST":bsha,"BF_FINAL":fsha},
             "model_steps":{"BF_BEST":bstep,"BF_FINAL":fstep},
             "solver_semantics":{"primary":"identical legacy probe, epsilon_f=0.03",
                 "sensitivity":"BF checkpoints only epsilon_f=0.002; not comparable as fixed-solver",
                 "fov":"analytic SensorOracle independent of all learned scores",
                 "other_parameters":"all unchanged"}})
    shard=out/f"solves.rank{a.rank}.jsonl"
    with shard.open("w") as f:
        count=0
        for i in range(a.rank,len(starts),a.world):
            spec=starts[i]
            x=torch.tensor(spec["x"],device=device,dtype=torch.float32)
            q=torch.tensor(spec["q_init"],device=device,dtype=torch.float32)
            sensor=int(spec["sensor"])
            row={**spec,"case_id":i,"models":{}}
            names=NAMES[i%len(NAMES):]+NAMES[:i%len(NAMES)]
            for name in names:
                row["models"][name]=runtime_probe.run_probe(probes[name],oracle,x,q,sensor)
            f.write(json.dumps(audit_core.json_safe(row),allow_nan=False)+"\n")
            count+=1
            if count%100==0:
                f.flush()
                print(f"[rank{a.rank}] {count} cases",flush=True)
    print(f"[done] rank={a.rank}, cases={count}, shard={shard}",flush=True)

if __name__=="__main__":
    main()
