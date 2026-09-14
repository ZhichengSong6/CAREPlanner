#!/usr/bin/env python3
"""Matched high-density evaluation for P0 versus routed capacity variants A/B/C."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

import abc_protocol as proto
from gradient_conflict import compute as gradient_conflict

old = proto.old


def fraction(k, n):
    return {"passed": int(k), "count": int(n), "rate": float(k/n) if n else None}


def solve_summary(rows, names, stats):
    result = dict(count=len(rows), models={}, paired={})
    for name in names:
        rr = [r["models"][name] for r in rows]
        result["models"][name] = dict(
            fov_pass=fraction(sum(r["fov_pass"] for r in rr), len(rr)),
            root_002=fraction(sum(r["predicted_root_within_002"] for r in rr), len(rr)),
            solver_ms=stats.finite_dist([r["solver_ms"] for r in rr]),
            failure_stages=dict(Counter(r["failure_stage"] for r in rr)),
            root_sources=dict(Counter(r["root_source"] for r in rr)),
            root_source_by_failure=dict(Counter(r["root_source"]+" / "+r["failure_stage"] for r in rr)),
        )
    for candidate in [n for n in names if n in ("A","B","C")]:
        result["paired"]["P0_vs_"+candidate] = dict(Counter(
            "both_pass" if r["models"]["P0"]["fov_pass"] and r["models"][candidate]["fov_pass"] else
            candidate+"_only" if r["models"][candidate]["fov_pass"] else
            "P0_only" if r["models"]["P0"]["fov_pass"] else "both_fail" for r in rows
        ))
    return result


def aggregate_solve(rows, names):
    result = {}
    for cohort, chosen in (
        ("local", [r for r in rows if "/local_" in r["group"]]),
        ("uniform", [r for r in rows if r["group"].endswith("uniform_outside")]),
        ("all", rows),
    ):
        result[cohort] = {name: fraction(sum(r["models"][name]["fov_pass"] for r in chosen), len(chosen)) for name in names}
    return result


def markdown(report, names):
    def f(v): return "N/A" if v is None else f"{v:.5f}"
    lines = [
        "# H9 A/B/C capacity-routing diagnostic", "",
        "All A/B/C start function-equivalent to the same completed P0 checkpoint.",
        "A trains current sensor decoders only; B adds identity residual adapters; C privatizes the final 512->256 tail.",
        "Shared/union paths are frozen. Training uses only the original per-sensor global objective; no P1/P2/P3 boundary losses.",
        "FOV-only development validation; LOS/collision/trajectory/actual-seen NOT_RUN.", "",
        "## Architecture and initialization", "",
        "| Arm | Trainable params | Total params | init max |dq| error | val routed loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("A","B","C"):
        t = report["training"][arm]
        lines.append(f"| {arm} | {t['architecture']['trainable_parameters']} | {t['architecture']['total_parameters']} | "
                     f"{t['initial_equivalence']['max_abs_q_gradient_error']:.3e} | {t['validation']['routed_sensor']['loss']:.5f} |")
    lines += ["", "## Matched solver aggregate", "",
              "| Cohort | N | "+" | ".join(n+" FOV" for n in names)+" |",
              "|---|---:|"+"---:|"*len(names)]
    for cohort in ("local","uniform","all"):
        row = report["solve_aggregate"][cohort]
        n = next(iter(row.values()))["count"]
        lines.append(f"| {cohort} | {n} | "+" | ".join(f(row[nm]["rate"]) for nm in names)+" |")
    lines += ["", "## Fixed radial neighborhood sign", "",
              "| Model | N | sign accuracy | positive recall | negative recall | FP | FN |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        n = report["neighborhood_sentinel"][name]
        lines.append(f"| {name} | {int(n['valid'])} | {f(n['sign_accuracy'])} | {f(n['positive_recall'])} | "
                     f"{f(n['negative_recall'])} | {int(n['fp'])} | {int(n['fn'])} |")
    lines += ["", "## Sensor-max planning", "",
              "| Model | N | projection abs(g)<.03 | ascent1 g>=.03 | ascent10 g>=.03 |",
              "|---|---:|---:|---:|---:|"]
    for name in names:
        row = report["planning_sentinel"][f"{name}/sensor_max"]
        lines.append(f"| {name} | {row['count']} | {f(row['proj_oracle_boundary_030'])} | "
                     f"{f(row['asc1_g_ge_0p03'])} | {f(row['asc10_g_ge_0p03'])} |")
    gc = report.get("gradient_conflict")
    if gc:
        full = gc["matrices"]["full_original_head_objective"]
        lines += ["", "## P0 shared-backbone gradient conflict", "",
                  f"Full shared: {full['full_shared']['negative_pairs']}/{full['full_shared']['total_pairs']} task pairs have negative cosine.",
                  f"Final shared 512->256: {full['last_shared_512_to_256']['negative_pairs']}/{full['last_shared_512_to_256']['total_pairs']} task pairs have negative cosine.",
                  f"Worst full-shared pair: {full['full_shared']['min_pair']}", ""]
    lines += [
        "report.json also contains boundary geometry, two-sided sign profiles, per-group solver failure/root-source tables,",
        "field/ranking metrics, paired P0-vs-A/B/C outcomes, learning metadata and gradient-conflict matrices.",
        "No automatic promotion; this experiment separates frozen-feature sufficiency, decoder capacity, and private-tail capacity.",
    ]
    return "\n".join(lines)+"\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--mode", choices=("smoke","pilot"), required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires one allocated GPU")
    device = torch.device("cuda",0)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root = args.reference_root.resolve()
    out = proto.evaluation_dir(root,args.mode)
    if out.exists():
        raise FileExistsError(f"No overwrite: {out}")
    cache,p0,p0_sha = proto.load_reference_root(root)

    checkpoints, digests, models = {}, {}, {}
    for arm in proto.ARMS:
        cp,digest = proto.load_abc_checkpoint(proto.output_dir(root,arm,args.mode)/"final.pt")
        proto.assert_checkpoint(cp,p0,p0_sha,cache.identity,require_pilot=args.mode=="pilot")
        checkpoints[arm],digests[arm]=cp,digest
    streams={checkpoints[a]["training_stream_sha256"] for a in proto.ARMS}
    if len(streams)!=1:
        raise ValueError(f"A/B/C did not see identical training streams: {streams}")

    artifact=Path(p0["args"]["artifact_root"])
    if args.mode=="pilot":
        v1,_=old.load_v1(artifact/old.V1_REL,device)
        models["V1"]=v1
    models["P0"]=proto.p0_model(p0,device).eval().requires_grad_(False)
    for arm in proto.ARMS:
        models[arm]=proto.load_model_from_checkpoint(checkpoints[arm],p0,device).requires_grad_(False)
    names=list(models)

    ev=old.module("evaluate_pair",proto.p2.LEGACY)
    stats=old.module("evaluate_p2",proto.P2_DIR)
    side=old.module("neighborhood",proto.P3_DIR)
    api=old.module("train_signed_visibility_cdf_pairwise_replace",old.SCRIPTS)
    dataset=api.VisibilityQ0Dataset(str(artifact/old.DATA_REL),1000,0)
    cache.verify_dataset(dataset,artifact/old.DATA_REL)
    oracle=old.module("oracle",old.AUDIT).SensorOracle(old.URDF,device,api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES)
    pair=side.PairwiseFOV(oracle)
    core=old.module("core",old.AUDIT)
    probeapi=old.module("runtime_probe",old.AUDIT)
    lo,hi=dataset.q_limits(device)
    probes={k:probeapi.make_probe(ev.SensorView(v),dataset.sensor_masks(device),lo,hi) for k,v in models.items()}
    checks=old.module("audit",old.AUDIT).preflight(dataset,oracle,probes,device)
    checks["pairwise_neighbor_FOV"]=pair.verify(cache,device)
    out.mkdir(parents=True)

    manifest=dict(
        status="RUNNING",mode=args.mode,parent="P0",parent_sha256=p0_sha,
        training_streams="MATCH",training_stream_sha256=next(iter(streams)),
        cache_manifest_sha256=cache.identity,
        models={"P0":p0_sha,**digests},evaluated_models=names,preflight=checks,
        source_sha256=old.source_fingerprints(),abc_source_sha256=proto.fingerprints(),
        field_batches=1 if args.mode=="smoke" else 10,
        planning_batches=1 if args.mode=="smoke" else 2,
        solve_points_per_sensor=2 if args.mode=="smoke" else 32,
        limitations="Development held-out validation, FOV only; not safety or final generalization certification.",
    )
    proto.write_json(out/"manifest.json",manifest)
    started=time.perf_counter()
    try:
        result=dict(status="RUNNING",mode=args.mode,boundary={},profiles={},solves={},training={})
        for arm in proto.ARMS:
            cp=checkpoints[arm]
            result["training"][arm]=dict(
                architecture=cp["architecture"],initial_equivalence=cp["initial_equivalence"],
                validation=cp["validation"],training_stream_sha256=cp["training_stream_sha256"],
            )
        d=cache.arrays["val"]
        preds={name:ev.selected(m,d["x"],d["q"],d["s"],d["normal"],device) for name,m in models.items()}
        for group,ids in enumerate(cache.groups["val"]):
            key=f"S{group//2}/{'bank_refined' if group%2==0 else 'offbank_refined'}"
            result["boundary"][key]=dict(count=len(ids),models={n:stats.boundary_stats(a,ids) for n,a in preds.items()})
        for offset in (-.05,-.02,-.01,-.005,.005,.01,.02,.05):
            qs=d["q"]+offset*d["normal"]
            ids=np.flatnonzero(((qs>=cache.lo)&(qs<=cache.hi)).all(1))
            gs=np.asarray([oracle.value(torch.tensor(d["x"][i],device=device),torch.tensor(qs[i],device=device),int(d["s"][i])) for i in ids])
            for name,model in models.items():
                yp=ev.selected(model,d["x"][ids],qs[ids],d["s"][ids],d["normal"][ids],device)["value"]
                for group in range(16):
                    keep=(2*d["s"][ids]+d["kind"][ids])==group
                    key=f"S{group//2}/kind{group%2}/offset={offset:+.3f}"
                    row=result["profiles"].setdefault(key,dict(in_limits=int(keep.sum()),models={}))
                    row["models"][name]=stats.confusion(yp[keep],gs[keep])
        print("[eval] boundary and two-sided sign profiles complete",flush=True)

        radial=side.radii_for_update(0,0,len(d["s"]))
        accum={n:torch.zeros((32,len(side.COLUMNS)),device=device,dtype=torch.float64) for n in names}
        with torch.no_grad():
            for start in range(0,len(d["s"]),128):
                ids=np.arange(start,min(start+128,len(d["s"])))
                nb=side.build_queries(cache.tensors("val",ids,device),radial[ids],pair,lo,hi)
                for name,m in models.items():
                    _,st=side.sign_loss(m,nb,torch.ones(32,device=device))
                    accum[name]+=st
        result["neighborhood_sentinel"]={n:side.summary(st) for n,st in accum.items()}

        starts,excluded=[],Counter(); rng=np.random.default_rng(91283)
        for s in range(8):
            pool=np.unique(d["x_index"][d["s"]==s])
            pts=rng.choice(pool,min(manifest["solve_points_per_sensor"],len(pool)),replace=False)
            for xi in pts:
                for kind in (0,1):
                    inds=np.flatnonzero((d["s"]==s)&(d["x_index"]==xi)&(d["kind"]==kind))
                    if not len(inds): excluded[f"S{s}/missing_kind{kind}"]+=1; continue
                    i=inds[0]; x=torch.tensor(d["x"][i],device=device)
                    for radius in (.02,.05):
                        qout=torch.tensor(d["q"][i]-radius*d["normal"][i],device=device)
                        qin=torch.tensor(d["q"][i]+radius*d["normal"][i],device=device)
                        group=f"S{s}/local_kind{kind}_r{radius}"
                        if (not core.within(qout,lo,hi) or not core.within(qin,lo,hi) or
                            oracle.value(x,qout,s)>=-1e-5 or oracle.value(x,qin,s)<=1e-5):
                            excluded[group]+=1; continue
                        starts.append((group,int(xi),s,x,qout))
                x=dataset.x_cpu[int(xi)].to(device)
                for _ in range(2):
                    q=torch.tensor(rng.uniform(cache.lo,cache.hi),device=device,dtype=torch.float32)
                    if oracle.value(x,q,s)>=0: excluded[f"S{s}/uniform_initially_inside"]+=1; continue
                    starts.append((f"S{s}/uniform_outside",int(xi),s,x,q))
        groups=defaultdict(list); all_rows=[]
        with (out/"solves.jsonl").open("w") as f:
            for number,(group,xi,s,x,q) in enumerate(starts):
                row=dict(group=group,x_index=xi,sensor=s,x=x.tolist(),q_init=q.tolist(),models={})
                order=names[number%len(names):]+names[:number%len(names)]
                for name in order:
                    row["models"][name]=probeapi.run_probe(probes[name],oracle,x,q,s)
                f.write(json.dumps(core.json_safe(row),allow_nan=False)+"\n");f.flush()
                groups[group].append(row);all_rows.append(row)
                if (number+1)%50==0: print(f"[eval] matched_solves={number+1}/{len(starts)}",flush=True)
        result["solves"]={k:solve_summary(rows,names,stats) for k,rows in groups.items()}
        result["solve_aggregate"]=aggregate_solve(all_rows,names)
        result["excluded"]=dict(excluded)

        sign_oracle=api.PinocchioFOVOracle(str(old.URDF),api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES,50.,66.,.2,.7,.01)
        result["field_sentinel"]=ev.field_sentinel(models,dataset,sign_oracle,device,manifest["field_batches"])
        result["planning_sentinel"]=ev.planning_sentinel(models,dataset,sign_oracle,device,out,manifest["planning_batches"])
        result["gradient_conflict"]=gradient_conflict(root,device) if args.mode=="pilot" else None
        result.update(status="COMPLETE",elapsed_seconds=time.perf_counter()-started)
        proto.write_json(out/"report.json",result)
        (out/"summary.md").write_text(markdown(result,names),encoding="utf-8")
        manifest.update(status="COMPLETE",elapsed_seconds=result["elapsed_seconds"])
        proto.write_json(out/"manifest.json",manifest)
    except Exception as exc:
        manifest.update(status="FAILED",error=repr(exc));proto.write_json(out/"manifest.json",manifest);raise
    print(f"[done] abc_evaluation_complete mode={args.mode} output={out}",flush=True)


if __name__=="__main__":
    main()
