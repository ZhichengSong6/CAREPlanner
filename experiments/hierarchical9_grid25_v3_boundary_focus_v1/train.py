#!/usr/bin/env python3
"""Controlled second H9 R1 training with boundary-focused V3 distance loss.

The *only* changed optimization choice from the completed 20k full-Grid25
baseline is increased masked MSE weight for |V3 distance| <= 0.10.
Same R1 architecture, frozen train/val split, random seed, Adam, cosine
schedule, union-proxy coefficient, batch, and 20k updates.

No LOS, self-occlusion, FD-gradient, Eikonal or V4 targets are introduced.
Checkpoint selection uses qualified near-boundary validation RMSE; global
RMSE/MAE/sign remain reported for honest baseline comparison.
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

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BASE=REPO/"experiments/hierarchical9_grid25_v3_first_train_v1"
ROOT=Path("/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")
FORMAT="care_grid25_h9_r1_full_124seg_v3_boundary_focus_training_v1"
BOUNDARY_BAND=0.10
BOUNDARY_BOOST=2.0

spec=importlib.util.spec_from_file_location("grid25_frozen_v3_training_helpers",BASE/"train.py")
if spec is None or spec.loader is None: raise ImportError("Original R1 training helper missing")
prior=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=prior
spec.loader.exec_module(prior)


def require(ok,msg):
    if not ok: raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def snapshot_path(root):
    return Path(root).resolve()/"derived/grid25_v3_full_training_v1/dataset_segments_000_123"


def run_path(root):
    return Path(root).resolve()/"derived/grid25_v3_boundary_focus_v1/run_r1_v3_boundary_focus"


def load_data(root):
    dataset=snapshot_path(root)
    meta=read(dataset/"manifest.json")
    require(meta.get("format")=="care_grid25_full_v3only_distance_snapshot_v1"
            and meta.get("status")=="FROZEN"
            and meta.get("ready_for_v3_distance_training") is True
            and meta.get("ready_for_original_fd_h9_training") is False,
            "Expected full 124-segment V3 distance-only dataset")
    require(sha(dataset/"labels.npz")==meta["labels_sha256"],
            "Frozen V3 label data changed")
    with np.load(dataset/"labels.npz",allow_pickle=False) as z:
        x=np.asarray(z["x"],np.float32).copy()
        q=np.asarray(z["q"],np.float32).copy()
        target=np.asarray(z["target"],np.float32).copy()
        valid=np.asarray(z["valid"],bool).copy()
        split=np.asarray(z["split"],np.uint8).copy()
        ix=np.asarray(z["x_index"],np.int64).copy()
    n=len(x)
    require(n==126201*4 and x.shape==(n,3) and q.shape==(n,7)
            and target.shape==valid.shape==(n,8)
            and ix.shape==split.shape==(n,),
            "Wrong full Grid25 V3 distance dataset shape")
    require(np.isfinite(x).all() and np.isfinite(q).all()
            and np.isfinite(target[valid]).all()
            and np.isin(split,[0,1]).all()
            and not (set(ix[split==0]) & set(ix[split==1])),
            "Nonfinite labels or train/val spatial split overlap")
    require(int(valid.sum())==meta["counts"]["valid_sensor_labels"]
            and len(np.unique(ix[split==0]))==meta["counts"]["train_x"]
            and len(np.unique(ix[split==1]))==meta["counts"]["val_x"],
            "Frozen train/val label counts changed")
    return np.concatenate((x,q),axis=1).astype(np.float32),target,valid,split,meta


def boundary_supervised_loss(output,y,mask,union_weight):
    """Baseline masked MSE with a single fixed boost near V3 zero level.

    Targets are the original qualified best-found continuous-FOV distances.
    Invalid labels remain masked. The union proxy is identical to baseline.
    """
    require(output.shape==(len(y),9) and y.shape==mask.shape==(len(y),8),
            "Unexpected H9 boundary training tensor shape")
    pred=output[:,1:].float()
    w=(1.0+BOUNDARY_BOOST*(y.abs()<=BOUNDARY_BAND).float())*mask.float()
    denom=w.sum()
    require(bool(denom>0), "Sampled batch lacks qualified V3 labels")
    sensor=(w*(pred-y).square()).sum()/denom
    both=mask.all(dim=1)
    union=(output[:,0]*0.).sum()
    if union_weight>0 and bool(both.any()):
        pseudo=y[both].max(dim=1).values
        union=(output[both,0].float()-pseudo).square().mean()
    return sensor+union_weight*union,sensor.detach(),union.detach(),int(both.sum().item())


def evaluate_boundary(model,inputs,targets,masks,split,batch,device):
    """Report only true qualified V3 labels in the fixed near-zero band."""
    chosen=np.flatnonzero(split==1)
    sq=ab=correct=count=0.
    model.eval()
    with torch.no_grad():
        for start in range(0,len(chosen),batch):
            ids=chosen[start:start+batch]
            t=torch.as_tensor(inputs[ids],device=device)
            y=torch.as_tensor(targets[ids],device=device)
            m=torch.as_tensor(masks[ids],device=device)
            pred=model(t)[:,1:].float()
            band=m & (y.abs()<=BOUNDARY_BAND)
            if not bool(band.any()):
                continue
            delta=pred-y
            sq+=float((delta.square()*band).sum().item())
            ab+=float((delta.abs()*band).sum().item())
            correct+=float(((((pred>=0)==(y>=0))&band).sum().item()))
            count+=float(band.sum().item())
    require(count>0, "No qualified near-boundary validation V3 labels")
    return dict(
        val_boundary_band=BOUNDARY_BAND,
        val_boundary_sensor_count=int(count),
        val_boundary_sensor_rmse=math.sqrt(sq/count),
        val_boundary_sensor_mae=ab/count,
        val_boundary_sensor_sign_accuracy=correct/count,
    )


def atomic_save(path,checkpoint):
    temp=path.with_name(path.name+f".tmp.{os.getpid()}")
    try:
        torch.save(checkpoint,temp)
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def run(args):
    require(torch.cuda.is_available(),"No GPU available for Grid25 H9 training")
    require(1<=args.steps<=100000 and 32<=args.batch<=4096
            and args.val_every>0 and 0<args.lr<=0.01
            and math.isfinite(args.lr) and 0<=args.union_weight<=1,
            "Invalid full Grid25 V3 training configuration")
    root=args.root.resolve()
    out=run_path(root)
    require(not out.exists(),"Training run already exists; do not overwrite")
    inputs,targets,valid,split,meta=load_data(root)
    train_ids=np.flatnonzero(split==0)
    val_ids=np.flatnonzero(split==1)
    require(len(train_ids)==meta["counts"]["train_queries"]
            and len(val_ids)==meta["counts"]["val_queries"],
            "Train/val query counts changed")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device("cuda:0")
    model=prior.load_model().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=args.steps,eta_min=args.lr*0.01)
    x_d=torch.as_tensor(inputs,device=device)
    y_d=torch.as_tensor(targets,device=device)
    m_d=torch.as_tensor(valid,device=device)
    index_d=torch.as_tensor(train_ids.astype(np.int64),device=device)
    out.mkdir(parents=True,exist_ok=False)
    config=dict(steps=args.steps,batch=args.batch,lr=args.lr,
                seed=args.seed,val_every=args.val_every,
                union_weight=args.union_weight,
                lr_schedule="cosine",lr_min=args.lr*0.01,
                boundary_band=BOUNDARY_BAND,
                boundary_boost=BOUNDARY_BOOST,
                sensor_loss="masked_distance_mse_boundary_weighted",
                checkpoint_selector="val_boundary_sensor_rmse")
    qualified_boundary=(valid & (np.abs(targets)<=BOUNDARY_BAND))
    boundary_train=int(qualified_boundary[split==0].sum())
    boundary_val=int(qualified_boundary[split==1].sum())
    require(boundary_train>0 and boundary_val>0,
            "No qualified V3 zero-boundary labels in frozen train/val")
    print("[BOUNDARY V3 FROZEN COVERAGE]",
          json.dumps(dict(band=BOUNDARY_BAND,
                          train_valid_labels=boundary_train,
                          val_valid_labels=boundary_val)),flush=True)
    log=dict(
        format=FORMAT,status="RUNNING",
        dataset=str(snapshot_path(root)),
        dataset_sha256=meta["labels_sha256"],
        model_path=str(prior.MODEL_SRC.relative_to(REPO)),
        model_source_sha256=sha(prior.MODEL_SRC),
        trainer_sha256=sha(__file__),
        model_architecture=model.architecture(),
        config=config,
        train_x=meta["counts"]["train_x"],val_x=meta["counts"]["val_x"],
        train_queries=len(train_ids),val_queries=len(val_ids),
        total_valid_sensor_v3=meta["counts"]["valid_sensor_labels"],
        no_fd_gradient_supervision=True,union_proxy_not_exact=True,
        v3_distance_full_grid25_training=True,
        boundary_only_change_from_baseline=True,
        boundary_train_valid_labels=boundary_train,
        boundary_val_valid_labels=boundary_val,
        ready_for_original_fd_h9_training=False,
    )
    (out/"run.json").write_text(json.dumps(log,indent=2,sort_keys=True)+"\n")
    best=float("inf")
    best_global_rmse=float("inf")
    best_boundary_sign=float("nan")
    best_step=0
    begin=time.monotonic()
    with (out/"metrics.jsonl").open("x",buffering=1) as metrics:
        for step in range(1,args.steps+1):
            model.train()
            idx=index_d[torch.randint(len(index_d),(args.batch,),device=device)]
            optimizer.zero_grad(set_to_none=True)
            prediction=model(x_d[idx]).float()
            loss,sensor_mse,proxy_mse,nunion=boundary_supervised_loss(
                prediction,y_d[idx],m_d[idx],args.union_weight)
            require(torch.isfinite(loss).item(),
                    "Nonfinite H9 supervised loss at step "+str(step))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            scheduler.step()
            if step==1 or step%args.val_every==0 or step==args.steps:
                ev=prior.evaluate(model,inputs,targets,valid,split,args.batch,device)
                ev.update(evaluate_boundary(
                    model,inputs,targets,valid,split,args.batch,device))
                require(math.isfinite(ev["val_sensor_rmse"])
                        and math.isfinite(ev["val_boundary_sensor_rmse"]),
                        "Nonfinite heldout V3 RMSE")
                row=dict(
                    step=step,
                    train_batch_total_mse=float(loss.detach().cpu()),
                    train_batch_sensor_mse=float(sensor_mse.cpu()),
                    train_batch_union_proxy_mse=float(proxy_mse.cpu()),
                    train_batch_union_proxy_count=nunion,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    elapsed_s=round(time.monotonic()-begin,2),
                    **ev,
                )
                metrics.write(json.dumps(row,sort_keys=True)+"\n")
                print("[H9 FULL V3 TRAIN]",json.dumps({
                    k:v for k,v in row.items() if k!="by_sensor"
                },sort_keys=True),flush=True)
                state=dict(
                    format=FORMAT,step=step,model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    config=config,source_v3_sha256=meta["labels_sha256"],
                    model_source_sha256=log["model_source_sha256"],
                    trainer_sha256=log["trainer_sha256"],
                    val_sensor_rmse=ev["val_sensor_rmse"],
                    val_boundary_sensor_rmse=ev["val_boundary_sensor_rmse"],
                    val_boundary_sensor_sign_accuracy=ev["val_boundary_sensor_sign_accuracy"],
                    no_fd_gradient_supervision=True,union_proxy_not_exact=True,
                )
                atomic_save(out/"latest.pt",state)
                if ev["val_boundary_sensor_rmse"]<best:
                    best=ev["val_boundary_sensor_rmse"]
                    best_global_rmse=ev["val_sensor_rmse"]
                    best_boundary_sign=ev["val_boundary_sensor_sign_accuracy"]
                    best_step=step
                    atomic_save(out/"best.pt",state)
    os.replace(out/"latest.pt",out/"final.pt")
    log.update(
        status="COMPLETE",successful_updates=args.steps,
        best_val_sensor_rmse=best_global_rmse,
        best_val_boundary_sensor_rmse=best,
        best_val_boundary_sign_accuracy=best_boundary_sign,
        best_step=best_step,
        best_sha256=sha(out/"best.pt"),
        final_sha256=sha(out/"final.pt"),
        metrics_sha256=sha(out/"metrics.jsonl"),
        elapsed_s=round(time.monotonic()-begin,2))
    temp=out/"run.json.tmp"
    temp.write_text(json.dumps(log,indent=2,sort_keys=True)+"\n")
    os.replace(temp,out/"run.json")
    print("[H9 FULL V3 BOUNDARY TRAIN COMPLETE]",
          "best_boundary_RMSE=",round(best,6),
          "global_RMSE_at_best=",round(best_global_rmse,6),
          "best_step=",best_step,"out=",out,flush=True)


def status(root):
    out=run_path(root)
    m=read(out/"run.json")
    print(json.dumps({
        "status":m.get("status"),
        "successful_updates":m.get("successful_updates"),
        "best_val_sensor_rmse":m.get("best_val_sensor_rmse"),
        "best_val_boundary_sensor_rmse":m.get("best_val_boundary_sensor_rmse"),
        "best_val_boundary_sign_accuracy":m.get("best_val_boundary_sign_accuracy"),
        "best_step":m.get("best_step"),
        "output":str(out),
        "v3_distance_full_grid25_training":True,
    },indent=2))
    if m.get("status")=="COMPLETE":
        require(sha(out/"best.pt")==m["best_sha256"]
                and sha(out/"final.pt")==m["final_sha256"]
                and sha(out/"metrics.jsonl")==m["metrics_sha256"],
                "H9 checkpoint metrics checksum failed")
        print("[H9 FULL V3 BOUNDARY CHECKPOINTS VERIFIED]")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("train","status"))
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--steps",type=int,default=20000)
    p.add_argument("--batch",type=int,default=1024)
    p.add_argument("--val-every",type=int,default=500)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--union-weight",type=float,default=0.2)
    args=p.parse_args()
    (run(args) if args.mode=="train" else status(args.root))


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,OSError,KeyError,FileNotFoundError) as e:
        raise SystemExit("[STOP] "+str(e))
