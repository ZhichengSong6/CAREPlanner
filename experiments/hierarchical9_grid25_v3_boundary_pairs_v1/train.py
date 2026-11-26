#!/usr/bin/env python3
"""Third-round controlled H9 R1 training with real re-labeled FOV boundary pairs.

Source: exactly the immutable 124-segment V3-only dataset PLUS newly
qualified original-V3 two-sided signed distances near true FOV boundaries.
No gradient loss, LOS, V4, model architecture changes or old data overwrite.
Fixed 20k steps, original full-Val checkpoint selector; added paired-Val
diagnostics remain separate and never enter training.
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
FORMAT="care_grid25_h9_r1_full_124seg_v3_boundary_pair_aug_training_v1"
PAIR_FRACTION=0.25

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
    return Path(root).resolve()/"derived/grid25_v3_boundary_pairs_v1/run_r1_v3_pairs_aug"


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


def pair_root(root):
    return Path(root).resolve()/"derived/grid25_v3_boundary_pairs_v1"


def load_pairs(root):
    folder=pair_root(root)
    info=read(folder/"augmentation_manifest.json")
    require(info.get("format")=="care_grid25_v3_boundary_pairs_relabel_v1"
            and info.get("status")=="FROZEN"
            and info.get("no_los") is True
            and info.get("no_fd_gradient_supervision") is True,
            "Expected frozen original-V3, FOV-only boundary-pair labels")
    original=read(snapshot_path(root)/"manifest.json")
    require(info.get("source_dataset_labels_sha256")==original["labels_sha256"],
            "Augmentation uses a different Grid25 base")
    pairs={}
    for split,name,checksum in ((0,"train_pairs.npz","train_pairs_sha256"),
                                (1,"val_pairs.npz","val_pairs_sha256")):
        p=folder/name
        require(sha(p)==info[checksum],"Augmentation SHA mismatch "+name)
        with np.load(p,allow_pickle=False) as z:
            values={k:np.asarray(z[k]).copy()
                    for k in ("x","q","target","valid","split","x_index","sensor")}
        n=len(values["x"])
        v=values["valid"]
        t=values["target"]
        require(values["x"].shape==(n,3)
                and values["q"].shape==(n,7)
                and t.shape==v.shape==(n,8)
                and len(values["split"])==n
                and len(values["x_index"])==n
                and n==2*info["qualified_"+("train" if split==0 else "val")+"_pairs"]
                and np.all(values["split"]==split)
                and np.all(v.sum(axis=1)==1)
                and np.all(np.isfinite(t[v]))
                and np.all(np.abs(t[v])<=0.20)
                and np.isfinite(values["x"]).all()
                and np.isfinite(values["q"]).all(),
                "Malformed or unqualified two-sided augmentation")
        require(np.all(t[~v]==0),"Invalid sensors must be zero-filled")
        require(np.all(t[0::2][v[0::2]]>0)
                and np.all(t[1::2][v[1::2]]<0),
                "Augmented pairs are not opposite signed FOV boundary samples")
        inp=np.concatenate((values["x"],values["q"]),axis=1).astype(np.float32)
        pairs[split]=dict(inputs=inp,targets=t.astype(np.float32),valid=v,
                          x_index=values["x_index"])
    require(not (set(pairs[0]["x_index"].tolist())&
                 set(pairs[1]["x_index"].tolist())),
            "Augmentation train/val x leakage")
    return pairs,info


def evaluate_pair_val(model,pair,batch,device):
    """Separate new frozen boundary-pair Val; never used for best.pt selection."""
    n=len(pair["inputs"])
    square=absolute=correct=count=0.
    by={f"S{s}":dict(count=0,sum_sq=0.,sign_correct=0) for s in range(8)}
    model.eval()
    with torch.no_grad():
        for a in range(0,n,batch):
            b=min(n,a+batch)
            x=torch.as_tensor(pair["inputs"][a:b],device=device)
            y=torch.as_tensor(pair["targets"][a:b],device=device)
            mask=torch.as_tensor(pair["valid"][a:b],device=device)
            pred=model(x)[:,1:].float()
            diff=pred-y
            square+=float((diff.square()*mask).sum().item())
            absolute+=float((diff.abs()*mask).sum().item())
            correct+=int(((((pred>=0)==(y>=0))&mask).sum().item()))
            count+=int(mask.sum().item())
            for s in range(8):
                n_s=int(mask[:,s].sum().item())
                by[f"S{s}"]["count"]+=n_s
                by[f"S{s}"]["sum_sq"]+=float((diff[:,s].square()*mask[:,s]).sum().item())
                by[f"S{s}"]["sign_correct"]+=int(((((pred[:,s]>=0)==(y[:,s]>=0))&mask[:,s]).sum().item())
    require(count==n,"Each qualified augmented Val row must have one label")
    return dict(
        val_new_boundary_count=count,
        val_new_boundary_rmse=math.sqrt(square/count),
        val_new_boundary_mae=absolute/count,
        val_new_boundary_sign_accuracy=correct/count,
        val_new_by_sensor={
            s:dict(count=v["count"],
                   rmse=(math.sqrt(v["sum_sq"]/v["count"]) if v["count"] else None),
                   sign_accuracy=(v["sign_correct"]/v["count"] if v["count"] else None))
            for s,v in by.items()},
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
    pairs,pair_meta=load_pairs(root)
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
    pair_x_d=torch.as_tensor(pairs[0]["inputs"],device=device)
    pair_y_d=torch.as_tensor(pairs[0]["targets"],device=device)
    pair_m_d=torch.as_tensor(pairs[0]["valid"],device=device)
    require(len(pair_x_d)>0 and len(pair_y_d)==len(pair_m_d)==len(pair_x_d),
            "Missing genuine new two-sided training labels")
    out.mkdir(parents=True,exist_ok=False)
    config=dict(steps=args.steps,batch=args.batch,lr=args.lr,
                seed=args.seed,val_every=args.val_every,
                union_weight=args.union_weight,
                lr_schedule="cosine",lr_min=args.lr*0.01,
                new_boundary_sample_fraction=PAIR_FRACTION,
                augmented_sampling="75pct_old_query_rows_25pct_new_one_sensor_rows",
                checkpoint_selector="original_frozen_val_sensor_rmse",
                no_new_loss_terms=True)
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
        original_124_segment_dataset_unchanged=True,
        original_124_segment_source_sha256=meta["labels_sha256"],
        augmentation_manifest_sha256=sha(pair_root(root)/"augmentation_manifest.json"),
        augmentation_train_sha256=pair_meta["train_pairs_sha256"],
        augmentation_val_sha256=pair_meta["val_pairs_sha256"],
        new_augmented_train_rows=len(pair_x_d),
        new_augmented_val_rows=len(pairs[1]["inputs"]),
        ready_for_original_fd_h9_training=False,
    )
    (out/"run.json").write_text(json.dumps(log,indent=2,sort_keys=True)+"\n")
    best=float("inf")
    best_new_boundary_rmse=float("inf")
    best_new_boundary_sign=float("nan")
    best_step=0
    begin=time.monotonic()
    with (out/"metrics.jsonl").open("x",buffering=1) as metrics:
        for step in range(1,args.steps+1):
            model.train()
            old_batch=int(args.batch*(1.0-PAIR_FRACTION))
            new_batch=args.batch-old_batch
            idx=index_d[torch.randint(len(index_d),(old_batch,),device=device)]
            pidx=torch.randint(len(pair_x_d),(new_batch,),device=device)
            optimizer.zero_grad(set_to_none=True)
            old_pred=model(x_d[idx]).float()
            new_pred=model(pair_x_d[pidx]).float()
            old_total,sensor_mse,proxy_mse,nunion=prior.supervised_loss(
                old_pred,y_d[idx],m_d[idx],args.union_weight)
            pair_mse=((new_pred[:,1:]-pair_y_d[pidx]).square()
                      *pair_m_d[pidx]).sum()/pair_m_d[pidx].sum()
            # Exactly the original supervised distance MSE, but sample the
            # newly labeled two-sided boundary queries for 25% of updates.
            # Original union proxy remains 0.2 on fully valid original rows.
            loss=(1.0-PAIR_FRACTION)*sensor_mse+PAIR_FRACTION*pair_mse
            loss+=args.union_weight*proxy_mse
            require(torch.isfinite(loss).item(),
                    "Nonfinite H9 supervised loss at step "+str(step))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            scheduler.step()
            if step==1 or step%args.val_every==0 or step==args.steps:
                ev=prior.evaluate(model,inputs,targets,valid,split,args.batch,device)
                ev.update(evaluate_pair_val(model,pairs[1],args.batch,device))
                require(math.isfinite(ev["val_sensor_rmse"])
                        and math.isfinite(ev["val_new_boundary_rmse"]),
                        "Nonfinite heldout V3 RMSE")
                row=dict(
                    step=step,
                    train_batch_total_mse=float(loss.detach().cpu()),
                    train_batch_sensor_mse=float(sensor_mse.cpu()),
                    train_batch_new_boundary_sensor_mse=float(pair_mse.detach().cpu()),
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
                    val_new_boundary_rmse=ev["val_new_boundary_rmse"],
                    val_new_boundary_sign_accuracy=ev["val_new_boundary_sign_accuracy"],
                    augmented_train_sha256=pair_meta["train_pairs_sha256"],
                    augmented_val_sha256=pair_meta["val_pairs_sha256"],
                    no_fd_gradient_supervision=True,union_proxy_not_exact=True,
                )
                atomic_save(out/"latest.pt",state)
                if ev["val_sensor_rmse"]<best:
                    best=ev["val_sensor_rmse"]
                    best_new_boundary_rmse=ev["val_new_boundary_rmse"]
                    best_new_boundary_sign=ev["val_new_boundary_sign_accuracy"]
                    best_step=step
                    atomic_save(out/"best.pt",state)
    os.replace(out/"latest.pt",out/"final.pt")
    log.update(
        status="COMPLETE",successful_updates=args.steps,
        best_val_sensor_rmse=best,
        best_val_new_boundary_rmse=best_new_boundary_rmse,
        best_val_new_boundary_sign_accuracy=best_new_boundary_sign,
        best_step=best_step,
        best_sha256=sha(out/"best.pt"),
        final_sha256=sha(out/"final.pt"),
        metrics_sha256=sha(out/"metrics.jsonl"),
        elapsed_s=round(time.monotonic()-begin,2))
    temp=out/"run.json.tmp"
    temp.write_text(json.dumps(log,indent=2,sort_keys=True)+"\n")
    os.replace(temp,out/"run.json")
    print("[H9 V3 BOUNDARY PAIRS TRAIN COMPLETE]",
          "old_global_best_RMSE=",round(best,6),
          "new_pair_val_RMSE_at_best=",round(best_new_boundary_rmse,6),
          "best_step=",best_step,"out=",out,flush=True)


def status(root):
    out=run_path(root)
    m=read(out/"run.json")
    print(json.dumps({
        "status":m.get("status"),
        "successful_updates":m.get("successful_updates"),
        "best_val_sensor_rmse":m.get("best_val_sensor_rmse"),
        "best_val_new_boundary_rmse":m.get("best_val_new_boundary_rmse"),
        "best_val_new_boundary_sign_accuracy":m.get("best_val_new_boundary_sign_accuracy"),
        "best_step":m.get("best_step"),
        "output":str(out),
        "v3_distance_full_grid25_training":True,
    },indent=2))
    if m.get("status")=="COMPLETE":
        require(sha(out/"best.pt")==m["best_sha256"]
                and sha(out/"final.pt")==m["final_sha256"]
                and sha(out/"metrics.jsonl")==m["metrics_sha256"],
                "H9 checkpoint metrics checksum failed")
        print("[H9 V3 BOUNDARY PAIRS CHECKPOINTS VERIFIED]")


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
