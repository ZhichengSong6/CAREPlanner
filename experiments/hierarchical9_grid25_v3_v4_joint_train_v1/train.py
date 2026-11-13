#!/usr/bin/env python3
"""One-GPU H9 R1 V3+V4 joint supervised training, fair 1000-step V3-only comparator.

V3 labels are frozen, sign-qualified local continuous-boundary distances.
V4 local tube pairs ONLY supervise f(x,q*)=0 and f(x,q_t)-f(x,q*)≈t.
These are local constraints, NOT global exact SDF or independently FD-PASS gradient.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "experiments/hierarchical9_grid25_v3_first_train_v1"
sys.path.insert(0, str(BASE))
import train as v3base

FORMAT = "care_grid25_h9_r1_v3_v4_local_tube_joint_train_v1"
DEFAULT_ROOT = Path(
    "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
)


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def load_tubes(path, v3_snapshot):
    path = Path(path).resolve()
    m = json_read(path/"manifest.json")
    require(m.get("format") ==
            "care_grid25_segment0_v4_local_tube_training_v1"
            and m.get("status") == "FROZEN_LOCAL_TUBES",
            "Not the expected verified V4 local-tube snapshot")
    require(Path(m["v3_snapshot"]).resolve() == Path(v3_snapshot).resolve(),
            "Joint training V3 snapshot differs from tube source")
    v3meta=json_read(v3_snapshot/"manifest.json")
    require(m["v3_labels_sha256"] == v3meta["labels_sha256"] and
            sha(v3_snapshot/"labels.npz") == m["v3_labels_sha256"],
            "Underlying V3 labels changed")
    require(sha(path/"tubes.npz")==m["tubes_sha256"],
            "V4 tube snapshot checksum changed")
    with np.load(path/"tubes.npz",allow_pickle=False) as z:
        x=np.asarray(z["x"],np.float32).copy()
        qt=np.asarray(z["q"],np.float32).copy()
        center=np.asarray(z["center"],np.float32).copy()
        offset=np.asarray(z["offset"],np.float32).copy()
        split=np.asarray(z["split"],np.uint8).copy()
        sensor=np.asarray(z["sensor"],np.int64).copy()
        xi=np.asarray(z["x_index"],np.int64).copy()
    n=len(offset)
    require(x.shape==(n,3) and qt.shape==center.shape==(n,7)
            and sensor.shape==split.shape==xi.shape==(n,),
            "Tube snapshot invalid tensor shapes")
    require(np.isfinite(x).all() and np.isfinite(qt).all()
            and np.isfinite(center).all() and np.isfinite(offset).all()
            and np.all((sensor>=0)&(sensor<8)) and np.isin(split,[0,1]).all(),
            "Invalid V4 tube snapshot numeric values")
    require(not (set(xi[split==0]) & set(xi[split==1])),
            "Tube training x overlaps tube validation x")
    require(n==m["counts"]["tubes"] and
            int((split==0).sum())==m["counts"]["train_tubes"] and
            int((split==1).sum())==m["counts"]["val_tubes"],
            "Tube sample counts diverged from frozen metadata")
    return x,qt,center,offset,split,sensor,xi,m


def validate_baseline(root, v3meta):
    out=root/"derived/grid25_v3_first_train_v1/run_000"
    status=json_read(out/"run.json")
    require(status.get("status")=="COMPLETE" and
            status.get("format")==v3base.FORMAT and
            status.get("successful_updates")==1000,
            "Need completed 1000-step V3-only comparator")
    require(status.get("dataset_labels_sha256")==v3meta["labels_sha256"],
            "Baseline labels differ from joint-train labels")
    require(status.get("model_sha256")==v3base.sha(v3base.MODEL_SRC) and
            sha(out/"best.pt")==status["best_sha256"],
            "Baseline H9 model source/checkpoint differs")
    reference=dict(steps=1000,batch=512,seed=0,lr=0.001,
                   union_proxy_weight=0.2)
    for key,val in reference.items():
        require(status["config"][key]==val, f"Baseline differs on {key}")
    return dict(
        run=str(out),
        best_sha256=status["best_sha256"],
        best_val_sensor_rmse=status["best_val_sensor_rmse"],
        config=reference,
    )


def selected_head(model, x, q, sensor):
    pred=model.forward_sensors(torch.cat((x,q),dim=1)).float()
    return pred.gather(1,sensor.reshape(-1,1)).squeeze(1)


def tube_losses(model, x, qt, q0, sensor, offset):
    y0=selected_head(model,x,q0,sensor)
    yt=selected_head(model,x,qt,sensor)
    zero=y0.square().mean()
    # A local slope check is not a globally exact signed-distance target.
    # Unit normalization by 0.02 rad makes the 0.005/0.01/0.02 offsets
    # numerically visible to the optimizer; zero-offset rows have no delta loss.
    active=offset.abs()>1e-7
    if active.any():
        delta=(((yt[active]-y0[active]-offset[active])/0.02).square()).mean()
    else:
        delta=(yt*0.).sum()
    return zero,delta


@torch.no_grad()
def eval_tubes(model, x, qt, q0, sensor, offset, ids, device, batch=1024):
    model.eval()
    zero_sq=delta_sq=0.
    nz=0
    seen=0
    for start in range(0,len(ids),batch):
        sub=ids[start:start+batch]
        xx=torch.as_tensor(x[sub],device=device)
        tq=torch.as_tensor(qt[sub],device=device)
        cq=torch.as_tensor(q0[sub],device=device)
        ss=torch.as_tensor(sensor[sub],device=device)
        t=torch.as_tensor(offset[sub],device=device)
        a=selected_head(model,xx,cq,ss)
        b=selected_head(model,xx,tq,ss)
        seen += len(sub)
        zero_sq+=float((a.square()).sum().cpu())
        active=t.abs()>1e-7
        if active.any():
            delta_sq+=float((((b[active]-a[active])-t[active]).square()).sum().cpu())
            nz+=int(active.sum().item())
    require(seen>0 and nz>0,"Empty V4 validation tube subset")
    return dict(
        val_tube_subset_count=seen,
        val_tube_nonzero_count=nz,
        val_tube_zero_rmse=math.sqrt(zero_sq/seen),
        val_tube_delta_rmse_rad=math.sqrt(delta_sq/nz),
    )


def checkpoint(path, state):
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    try:
        torch.save(state,tmp)
        os.replace(tmp,path)
    finally:
        if tmp.exists():
            tmp.unlink()


def train(args):
    require(torch.cuda.is_available(), "CUDA required for H9 joint training")
    require(args.steps==1000 and args.batch==512 and args.eval_every==100
            and args.seed==0 and args.lr==1e-3
            and args.union_proxy_weight==0.2,
            "Use frozen V3-only comparator schedule for paired training")
    require(32<=args.tube_batch<=4096 and
            math.isfinite(args.tube_zero_weight) and
            math.isfinite(args.tube_delta_weight) and
            0 <= args.tube_zero_weight <= 2 and
            0 <= args.tube_delta_weight <= 2,
            "Invalid tube batch or auxiliary loss weights")
    root=args.root.resolve()
    v3=(root/"derived/grid25_v3_first_train_v1/dataset_segment_000").resolve()
    tubes=(root/"derived/grid25_v3_v4_joint_train_v1/tube_snapshot_000").resolve()
    out=(args.out or root/"derived/grid25_v3_v4_joint_train_v1/run_000").resolve()
    require(not out.exists(), "Joint train output already exists; no accidental overwrite")
    inputs,targets,valid,split,v3meta = v3base.load_snapshot(v3)
    tx,tq,t0,toff,tsplit,tsensor,txi,tubemeta=load_tubes(tubes,v3)
    baseline=validate_baseline(root,v3meta)
    train_rows=np.flatnonzero(split==0)
    val_rows=np.flatnonzero(split==1)
    tube_train=np.flatnonzero(tsplit==0)
    tube_val=np.flatnonzero(tsplit==1)
    require(len(train_rows)==3380 and len(val_rows)==716 and
            len(tube_train)>0 and len(tube_val)>0,
            "Frozen segment_000 split counts changed")
    with np.load(v3/"labels.npz", allow_pickle=False) as source_labels:
        v3_xids = np.asarray(source_labels["x_index"], dtype=np.int64)
    require(set(txi[tsplit==0]).issubset(set(v3_xids[split==0])),
            "V4 training includes a position absent from frozen V3 training")
    require(set(txi[tsplit==1]).issubset(set(v3_xids[split==1])),
            "V4 validation includes a position absent from frozen V3 validation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda:0")
    model=v3base.load_model().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    # V3 sampling MUST keep the previous run's global torch RNG stream.
    # V4 is drawn from an isolated generator and never consumes global draws.
    tube_gen=torch.Generator(device=device)
    tube_gen.manual_seed(20261113)
    x_d=torch.as_tensor(inputs,device=device)
    y_d=torch.as_tensor(targets,device=device)
    m_d=torch.as_tensor(valid,device=device)
    tri_d=torch.as_tensor(train_rows.astype(np.int64),device=device)
    tx_d=torch.as_tensor(tx,device=device)
    tq_d=torch.as_tensor(tq,device=device)
    t0_d=torch.as_tensor(t0,device=device)
    tv_d=torch.as_tensor(toff,device=device)
    ts_d=torch.as_tensor(tsensor,device=device)
    tt_d=torch.as_tensor(tube_train,device=device)
    # Fixed non-training subset for comparable local-boundary diagnostics.
    vrng=np.random.default_rng(260929)
    nval=min(8192,len(tube_val))
    sub=np.sort(vrng.choice(tube_val,size=nval,replace=False))
    out.mkdir(parents=True,exist_ok=False)
    config=dict(steps=args.steps,batch=args.batch,eval_every=args.eval_every,
                lr=args.lr,seed=args.seed,union_proxy_weight=args.union_proxy_weight,
                tube_batch=args.tube_batch,
                tube_zero_weight=args.tube_zero_weight,
                tube_delta_weight=args.tube_delta_weight)
    provenance=dict(
        format=FORMAT,status="RUNNING",
        v3_snapshot=str(v3),v3_labels_sha256=v3meta["labels_sha256"],
        v4_snapshot=str(tubes),v4_tubes_sha256=tubemeta["tubes_sha256"],
        source_model_sha256=v3base.sha(v3base.MODEL_SRC),
        trainer_sha256=sha(__file__),
        architecture=model.architecture(),
        baseline=baseline,
        config=config,
        train_queries=int(len(train_rows)),val_queries=int(len(val_rows)),
        train_tubes=int(len(tube_train)),val_tubes=int(len(tube_val)),
        science_limitations=[
            "V3 is the only distance target: best-found local continuous FOV boundary, not certified global nearest",
            "V4 regular tubes supervise local f(q*)=0 and finite-offset consistency only, not globally exact SDF",
            "V3 q-gradient candidate and tube normal are NOT independent FD-PASS gradient labels",
            "Union head max-of-8 V3 labels is a proxy only",
            "Full-grid25 training ready is false; first-segment joint-training ablation only"
        ],
    )
    (out/"run.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
    best=float("inf")
    start=time.monotonic()
    with (out/"metrics.jsonl").open("x",buffering=1) as metrics:
        for step in range(1,args.steps+1):
            model.train()
            chosen=tri_d[torch.randint(len(tri_d),(args.batch,),device=device)]
            # Separate random stream avoids changing original V3 mini-batches.
            chosen_t=tt_d[torch.randint(len(tt_d),(args.tube_batch,),device=device,
                                         generator=tube_gen)]
            opt.zero_grad(set_to_none=True)
            yp=model(x_d[chosen]).float()
            v3loss,vs,uv,un=v3base.supervised_loss(
                yp,y_d[chosen],m_d[chosen],args.union_proxy_weight)
            z,d=tube_losses(model,tx_d[chosen_t],tq_d[chosen_t],
                             t0_d[chosen_t],ts_d[chosen_t],tv_d[chosen_t])
            loss=(v3loss+args.tube_zero_weight*z+
                  args.tube_delta_weight*d)
            require(torch.isfinite(loss).item(),
                    f"Nonfinite combined loss at training step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.)
            opt.step()
            if step==1 or step%args.eval_every==0 or step==args.steps:
                ev=v3base.evaluate(model,inputs,targets,valid,split,
                                   args.batch,device)
                local=eval_tubes(model,tx,tq,t0,tsensor,toff,sub,device)
                row=dict(
                    step=step,train_batch_total_mse=float(loss.detach().cpu()),
                    train_batch_v3_sensor_mse=float(vs.cpu()),
                    train_batch_v3_union_proxy_mse=float(uv.cpu()),
                    train_batch_tube_zero_mse=float(z.detach().cpu()),
                    train_batch_tube_delta_normalized_mse=float(d.detach().cpu()),
                    train_batch_union_proxy_count=un,
                    **ev,**local,elapsed_s=round(time.monotonic()-start,2))
                require(math.isfinite(row["val_sensor_rmse"])
                        and math.isfinite(row["val_tube_zero_rmse"])
                        and math.isfinite(row["val_tube_delta_rmse_rad"]),
                        "Nonfinite joint validation metric")
                metrics.write(json.dumps(row,sort_keys=True)+"\n")
                print("[JOINT TRAIN]",json.dumps({k:v for k,v in row.items()
                      if k!="by_sensor"},sort_keys=True),flush=True)
                state=dict(format=FORMAT,step=step,model_state=model.state_dict(),
                           optimizer_state=opt.state_dict(),config=config,
                           v3_labels_sha256=v3meta["labels_sha256"],
                           v4_tubes_sha256=tubemeta["tubes_sha256"],
                           baseline_best_val_sensor_rmse=baseline["best_val_sensor_rmse"],
                           val_sensor_rmse=ev["val_sensor_rmse"],
                           val_tube_zero_rmse=local["val_tube_zero_rmse"],
                           val_tube_delta_rmse_rad=local["val_tube_delta_rmse_rad"],
                           original_r1_model_source_sha256=provenance["source_model_sha256"],
                           gradient_labels_fd_verified=False,
                           union_head_is_proxy=True)
                checkpoint(out/"latest.pt",state)
                if ev["val_sensor_rmse"]<best:
                    best=ev["val_sensor_rmse"]
                    checkpoint(out/"best.pt",state)
    os.replace(out/"latest.pt",out/"final.pt")
    provenance.update(
        status="COMPLETE",successful_updates=args.steps,
        best_val_sensor_rmse=best,
        improvement_over_v3_only_rmse=baseline["best_val_sensor_rmse"]-best,
        best_sha256=sha(out/"best.pt"),
        final_sha256=sha(out/"final.pt"),
        metrics_sha256=sha(out/"metrics.jsonl"),
        elapsed_s=round(time.monotonic()-start,2))
    temp=out/"run.json.tmp"
    temp.write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n")
    os.replace(temp,out/"run.json")
    print("[JOINT H9 TRAIN COMPLETE] steps=",args.steps,
          "best_rmse=",round(best,6),
          "baseline_rmse=",round(baseline["best_val_sensor_rmse"],6),
          "improvement=",round(provenance["improvement_over_v3_only_rmse"],6),
          flush=True)
    print("[BEST CHECKPOINT]",out/"best.pt",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--out",type=Path)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--batch",type=int,default=512)
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--union-proxy-weight",type=float,default=0.2)
    p.add_argument("--tube-batch",type=int,default=512)
    p.add_argument("--tube-zero-weight",type=float,default=0.2)
    p.add_argument("--tube-delta-weight",type=float,default=0.05)
    args=p.parse_args()
    train(args)


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,OSError,KeyError,FileNotFoundError) as exc:
        raise SystemExit("[STOP] "+str(exc))
