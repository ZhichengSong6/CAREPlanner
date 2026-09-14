#!/usr/bin/env python3
"""Matched offline + planner-facing evaluation for V1/P0/ABC-B/C/E0/E1/E2."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import e012_protocol as proto

# abc_protocol.py is loaded by path from e012_protocol, but its checkpoint loader
# intentionally uses `from abc_model import build_arm`.  When this evaluator is
# launched from the E012 directory, the sibling ABC experiment directory is not
# automatically on sys.path.  Add that directory explicitly for evaluation-only
# baseline reconstruction.  This does not change any E012 training fingerprint.
_abc_dir = str(proto.ABC_DIR)
if _abc_dir not in sys.path:
    sys.path.insert(0, _abc_dir)

old=proto.old
compat=proto._load("e012_eval_compat",proto.ABC_DIR/"eval_compat.py")


def fraction(k,n): return {"passed":int(k),"count":int(n),"rate":float(k/n) if n else None}


def solve_summary(rows,names,stats):
    out={"count":len(rows),"models":{},"paired_vs_V1":{}}
    for name in names:
        rr=[r["models"][name] for r in rows]
        out["models"][name]={"fov_pass":fraction(sum(r["fov_pass"] for r in rr),len(rr)),
            "root_002":fraction(sum(r["predicted_root_within_002"] for r in rr),len(rr)),
            "failure_stages":dict(Counter(r["failure_stage"] for r in rr)),
            "root_source_by_failure":dict(Counter(r["root_source"]+" / "+r["failure_stage"] for r in rr)),
            "solver_ms":stats.finite_dist([r["solver_ms"] for r in rr])}
    if "V1" in names:
        for name in names:
            if name=="V1": continue
            out["paired_vs_V1"][name]=dict(Counter(
                "both_pass" if r["models"]["V1"]["fov_pass"] and r["models"][name]["fov_pass"] else
                name+"_only" if r["models"][name]["fov_pass"] else
                "V1_only" if r["models"]["V1"]["fov_pass"] else "both_fail" for r in rows))
    return out


def aggregate(rows,names):
    result={}
    for key,chosen in (("local",[r for r in rows if "/local_" in r["group"]]),
                       ("uniform",[r for r in rows if r["group"].endswith("uniform_outside")]),
                       ("all",rows)):
        result[key]={n:fraction(sum(r["models"][n]["fov_pass"] for r in chosen),len(chosen)) for n in names}
    return result


def markdown(report,names):
    def f(v): return "N/A" if v is None else f"{v:.5f}"
    lines=["# H9 E0/E1/E2 end-to-end diagnostic","",
        "All E arms start exactly P0-equivalent, but every saved E checkpoint trains shared early, union and sensor paths.",
        "E0=normal end-to-end; E1=conflict-aware one-sensor shared routing; E2=E1 + runtime projection replay.",
        "Development FOV-only evaluation; no automatic Mainline-A promotion.","",
        "## Solve aggregate","","| Cohort | N | "+" | ".join(n+" FOV" for n in names)+" |",
        "|---|---:|"+"---:|"*len(names)]
    for cohort in ("local","uniform","all"):
        row=report["solve_aggregate"][cohort]; n=next(iter(row.values()))["count"]
        lines.append(f"| {cohort} | {n} | "+" | ".join(f(row[nm]["rate"]) for nm in names)+" |")
    lines += ["","## Offline sensor-max field","","| Model | MAE | sign | grad cosine | rank top1 | fallback |",
              "|---|---:|---:|---:|---:|---:|"]
    for n in names:
        fs=report["field_sentinel"][n]; sm=fs["fields"]["sensor_max"]; rk=fs["ranking"]
        lines.append(f"| {n} | {f(sm['mae'])} | {f(sm['sign_accuracy'])} | {f(sm['gradient_cosine_mean'])} | "
                     f"{f(rk['winner_top1_accuracy_or_recall'])} | {f(rk['fallback_accuracy_after_gt_winner_removed'])} |")
    lines += ["","## Planning sensor-max","","| Model | projection | ascent1 | ascent10 |","|---|---:|---:|---:|"]
    for n in names:
        p=report["planning_sentinel"][f"{n}/sensor_max"]
        lines.append(f"| {n} | {f(p['proj_oracle_boundary_030'])} | {f(p['asc1_g_ge_0p03'])} | {f(p['asc10_g_ge_0p03'])} |")
    lines += ["","## E-arm weight drift from P0-equivalent initialization","","| Arm | early rel-L2 | union rel-L2 | sensor tails rel-L2 | sensor heads rel-L2 |",
              "|---|---:|---:|---:|---:|"]
    for arm in proto.ARMS:
        d=report["training"][arm]["weight_drift"]
        lines.append(f"| {arm} | {d['early']['relative_l2']:.6e} | {d['union']['relative_l2']:.6e} | "
                     f"{d['sensor_tails']['relative_l2']:.6e} | {d['sensor_heads']['relative_l2']:.6e} |")
    lines += ["","report.json contains per-sensor boundary/sign, fixed radial neighborhood, paired solve outcomes, failures, field/ranking/planning and replay metadata."]
    return "\n".join(lines)+"\n"


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--reference-root",type=Path,required=True)
    ap.add_argument("--mode",choices=("smoke","pilot"),required=True); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    device=torch.device("cuda",0); torch.cuda.set_device(0); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    root=args.reference_root.resolve(); out=proto.evaluation_dir(root,args.mode)
    if out.exists(): raise FileExistsError(f"No overwrite: {out}")
    cache,p0,p0_sha=proto.load_reference_root(root)

    cps={}; digests={}; models={}
    for arm in proto.ARMS:
        cp,d=proto.load_checkpoint(proto.output_dir(root,arm,args.mode)/"final.pt")
        proto.assert_checkpoint(cp,p0,p0_sha,cache.identity,require_pilot=args.mode=="pilot")
        cps[arm]=cp;digests[arm]=d
    streams={cps[a]["training_stream_sha256"] for a in proto.ARMS}
    if len(streams)!=1: raise ValueError(f"E0/E1/E2 uniform streams differ: {streams}")

    artifact=Path(p0["args"]["artifact_root"])
    v1,_=old.load_v1(artifact/old.V1_REL,device); models["V1"]=v1
    models["P0"]=proto.p0_model(p0,device).eval().requires_grad_(False)
    if args.mode=="pilot":
        for arm in ("B","C"):
            try:
                cp,_=proto.abc.load_abc_checkpoint(root/"abc_capacity_routing"/arm/"final.pt")
                models["ABC-"+arm]=proto.abc.load_model_from_checkpoint(cp,p0,device).requires_grad_(False)
            except FileNotFoundError: pass
    for arm in proto.ARMS: models[arm]=proto.load_model_from_checkpoint(cps[arm],p0,device).requires_grad_(False)
    names=list(models)

    ev=old.module("evaluate_pair",proto.abc.p2.LEGACY); stats=old.module("evaluate_p2",proto.abc.P2_DIR)
    side=old.module("neighborhood",proto.P3_DIR); api=old.module("train_signed_visibility_cdf_pairwise_replace",old.SCRIPTS)
    dataset=api.VisibilityQ0Dataset(str(artifact/old.DATA_REL),1000,0); cache.verify_dataset(dataset,artifact/old.DATA_REL)
    oracle=old.module("oracle",old.AUDIT).SensorOracle(old.URDF,device,api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES)
    pair=side.PairwiseFOV(oracle); core=old.module("core",old.AUDIT); probeapi=old.module("runtime_probe",old.AUDIT)
    lo,hi=dataset.q_limits(device); probes={k:probeapi.make_probe(ev.SensorView(v),dataset.sensor_masks(device),lo,hi) for k,v in models.items()}
    checks=old.module("audit",old.AUDIT).preflight(dataset,oracle,probes,device); checks["pairwise_neighbor_FOV"]=pair.verify(cache,device)
    out.mkdir(parents=True)
    manifest={"status":"RUNNING","mode":args.mode,"parent":"P0","parent_sha256":p0_sha,
        "uniform_training_streams":"MATCH","training_stream_sha256":next(iter(streams)),"models":digests,
        "evaluated_models":names,"cache_manifest_sha256":cache.identity,"preflight":checks,
        "source_sha256":proto.all_fingerprints(),"field_batches":1 if args.mode=="smoke" else 10,
        "planning_batches":1 if args.mode=="smoke" else 2,"solve_points_per_sensor":2 if args.mode=="smoke" else 32,
        "limitations":"Development held-out FOV-only evaluation; LOS/collision/trajectory/actual-seen NOT_RUN."}
    proto.write_json(out/"manifest.json",manifest); started=time.perf_counter()
    try:
        result={"status":"RUNNING","mode":args.mode,"training":{},"boundary":{},"profiles":{},"solves":{}}
        for arm in proto.ARMS:
            result["training"][arm]={"validation":cps[arm]["validation"],"weight_drift":cps[arm]["weight_drift_from_initial"],
                "replay":cps[arm].get("replay"),"last_mining":cps[arm].get("last_mining")}
        d=cache.arrays["val"]
        preds={n:ev.selected(m,d["x"],d["q"],d["s"],d["normal"],device) for n,m in models.items()}
        for group,ids in enumerate(cache.groups["val"]):
            key=f"S{group//2}/{'bank_refined' if group%2==0 else 'offbank_refined'}"
            result["boundary"][key]={"count":len(ids),"models":{n:stats.boundary_stats(a,ids) for n,a in preds.items()}}
        for offset in (-.05,-.02,-.01,-.005,.005,.01,.02,.05):
            qs=d["q"]+offset*d["normal"]; ids=np.flatnonzero(((qs>=cache.lo)&(qs<=cache.hi)).all(1))
            gs=np.asarray([oracle.value(torch.tensor(d["x"][i],device=device),torch.tensor(qs[i],device=device),int(d["s"][i])) for i in ids])
            for n,m in models.items():
                yp=ev.selected(m,d["x"][ids],qs[ids],d["s"][ids],d["normal"][ids],device)["value"]
                for group in range(16):
                    keep=(2*d["s"][ids]+d["kind"][ids])==group; key=f"S{group//2}/kind{group%2}/offset={offset:+.3f}"
                    row=result["profiles"].setdefault(key,{"in_limits":int(keep.sum()),"models":{}}); row["models"][n]=stats.confusion(yp[keep],gs[keep])
        print("[eval] boundary/sign complete",flush=True)

        radial=side.radii_for_update(0,0,len(d["s"])); accum={n:torch.zeros((32,len(side.COLUMNS)),device=device,dtype=torch.float64) for n in names}
        with torch.no_grad():
            for start in range(0,len(d["s"]),128):
                ids=np.arange(start,min(start+128,len(d["s"]))); nb=side.build_queries(cache.tensors("val",ids,device),radial[ids],pair,lo,hi)
                for n,m in models.items():
                    _,st=side.sign_loss(m,nb,torch.ones(32,device=device)); accum[n]+=st
        result["neighborhood_sentinel"]={n:side.summary(st) for n,st in accum.items()}

        starts=[];excluded=Counter();rng=np.random.default_rng(91283)
        for s in range(8):
            pool=np.unique(d["x_index"][d["s"]==s]);pts=rng.choice(pool,min(manifest["solve_points_per_sensor"],len(pool)),replace=False)
            for xi in pts:
                for kind in (0,1):
                    ii=np.flatnonzero((d["s"]==s)&(d["x_index"]==xi)&(d["kind"]==kind))
                    if not len(ii): excluded[f"S{s}/missing_kind{kind}"]+=1;continue
                    i=ii[0];x=torch.tensor(d["x"][i],device=device)
                    for radius in (.02,.05):
                        qout=torch.tensor(d["q"][i]-radius*d["normal"][i],device=device); qin=torch.tensor(d["q"][i]+radius*d["normal"][i],device=device)
                        group=f"S{s}/local_kind{kind}_r{radius}"
                        if (not core.within(qout,lo,hi) or not core.within(qin,lo,hi) or oracle.value(x,qout,s)>=-1e-5 or oracle.value(x,qin,s)<=1e-5):
                            excluded[group]+=1;continue
                        starts.append((group,int(xi),s,x,qout))
                x=dataset.x_cpu[int(xi)].to(device)
                for _ in range(2):
                    q=torch.tensor(rng.uniform(cache.lo,cache.hi),device=device,dtype=torch.float32)
                    if oracle.value(x,q,s)>=0: excluded[f"S{s}/uniform_initially_inside"]+=1;continue
                    starts.append((f"S{s}/uniform_outside",int(xi),s,x,q))
        groups=defaultdict(list);all_rows=[]
        with (out/"solves.jsonl").open("w") as f:
            for number,(group,xi,s,x,q) in enumerate(starts):
                row={"group":group,"x_index":xi,"sensor":s,"x":x.tolist(),"q_init":q.tolist(),"models":{}}
                order=names[number%len(names):]+names[:number%len(names)]
                for n in order: row["models"][n]=probeapi.run_probe(probes[n],oracle,x,q,s)
                f.write(json.dumps(core.json_safe(row),allow_nan=False)+"\n");f.flush();groups[group].append(row);all_rows.append(row)
                if (number+1)%50==0: print(f"[eval] matched_solves={number+1}/{len(starts)}",flush=True)
        result["solves"]={k:solve_summary(v,names,stats) for k,v in groups.items()}; result["solve_aggregate"]=aggregate(all_rows,names);result["excluded"]=dict(excluded)

        sign_oracle=api.PinocchioFOVOracle(str(old.URDF),api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES,50.,66.,.2,.7,.01)
        result["field_sentinel"]=ev.field_sentinel(models,dataset,sign_oracle,device,manifest["field_batches"])
        planning_models={n:compat.legacy_compatible(m) for n,m in models.items()}
        result["planning_sentinel"]=ev.planning_sentinel(planning_models,dataset,sign_oracle,device,out,manifest["planning_batches"])
        result.update(status="COMPLETE",elapsed_seconds=time.perf_counter()-started);proto.write_json(out/"report.json",result)
        (out/"summary.md").write_text(markdown(result,names),encoding="utf-8");manifest.update(status="COMPLETE",elapsed_seconds=result["elapsed_seconds"]);proto.write_json(out/"manifest.json",manifest)
    except Exception as exc:
        manifest.update(status="FAILED",error=repr(exc));proto.write_json(out/"manifest.json",manifest);raise
    print(f"[done] e012_evaluation_complete mode={args.mode} output={out}",flush=True)


if __name__=="__main__": main()
