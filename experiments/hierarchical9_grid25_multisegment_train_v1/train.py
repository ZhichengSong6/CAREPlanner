#!/usr/bin/env python3
"""Paired H9 R1 experiments on exactly Grid25 production segments 0+1.

Modes:
- v3:   masked per-sensor verified V3 value + optional 8/8 union proxy
- joint: same V3 supervision plus V4 local anchor-zero / offset consistency
Both start from same random R1 weights; same 1000 steps, same V3 batch sampler,
same Grid25 frozen 75mm-buffered train/validation x positions. No FD gradient loss.
"""
from __future__ import annotations
import argparse
import hashlib
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
BASE = REPO/"experiments/hierarchical9_grid25_v3_first_train_v1"
sys.path.insert(0,str(BASE))
import train as original

FORMAT="care_grid25_h9_segment01_v3_v4_paired_train_v1"
ROOT = Path("/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")


def require(ok,message):
    if not ok: raise RuntimeError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dataset_path(root):
    return root/"derived/grid25_multisegment_train_v1/dataset_segments_000_001"


def output_path(root,mode):
    return root/"derived/grid25_multisegment_train_v1"/f"run_{mode}"


def load_data(root,mode):
    dataset=dataset_path(root)
    m=read(dataset/"manifest.json")
    require(m.get("format")=="care_grid25_multisegment_01_v3_v4_labels_v1"
            and m.get("status")=="FROZEN"
            and m.get("original_segments")==[0,1],
            "Expected frozen segments 0+1 training dataset")
    require(digest(dataset/"v3.npz")==m["v3.npz_sha256"],
            "V3 training dataset sha changed")
    if mode=="joint":
        require(digest(dataset/"v4.npz")==m["v4.npz_sha256"],
                "V4 training dataset sha changed")
    with np.load(dataset/"v3.npz",allow_pickle=False) as z:
        x=np.asarray(z["x"],np.float32).copy()
        q=np.asarray(z["q"],np.float32).copy()
        target=np.asarray(z["target"],np.float32).copy()
        valid=np.asarray(z["valid"],bool).copy()
        split=np.asarray(z["split"],np.uint8).copy()
        ix=np.asarray(z["x_index"],np.int64).copy()
    n=len(x)
    require(x.shape==(n,3) and q.shape==(n,7)
            and valid.shape==target.shape==(n,8)
            and split.shape==ix.shape==(n,)
            and n==8192 and int(valid.sum())==48455,
            "V3 training snapshot shape/count error")
    require(np.isfinite(x).all() and np.isfinite(q).all()
            and np.isfinite(target[valid]).all()
            and not (set(ix[split==0]) & set(ix[split==1])),
            "V3 training/val leakage or nonfinite label")
    require(len(np.unique(ix[split==0]))==1701
            and len(np.unique(ix[split==1]))==347,
            "Unexpected frozen spatial counts")
    inp=np.concatenate((x,q),axis=1).astype(np.float32)
    tubes=None
    if mode=="joint":
        with np.load(dataset/"v4.npz",allow_pickle=False) as z:
            xx=np.asarray(z["x"],np.float32).copy()
            tq=np.asarray(z["q"],np.float32).copy()
            center=np.asarray(z["center"],np.float32).copy()
            sensor=np.asarray(z["sensor"],np.int64).copy()
            offset=np.asarray(z["offset"],np.float32).copy()
            tsplit=np.asarray(z["split"],np.uint8).copy()
            txi=np.asarray(z["x_index"],np.int64).copy()
        k=len(xx)
        require(k==1958335 and tq.shape==center.shape==(k,7)
                and sensor.shape==offset.shape==tsplit.shape==txi.shape==(k,)
                and np.isfinite(xx).all() and np.isfinite(tq).all()
                and np.isfinite(center).all() and np.isfinite(offset).all()
                and np.all((sensor>=0)&(sensor<8))
                and not (set(txi[tsplit==0])&set(txi[tsplit==1])),
                "V4 tube snapshot invalid or leaking split")
        require(set(txi[tsplit==0]).issubset(set(ix[split==0]))
                and set(txi[tsplit==1]).issubset(set(ix[split==1])),
                "V4 tube and V3 training positions do not share frozen split")
        tubes=(xx,tq,center,sensor,offset,tsplit,txi)
    return inp,target,valid,split,ix,tubes,m


def chosen_sensor(model,x,q,s):
    pred=model.forward_sensors(torch.cat((x,q),dim=1)).float()
    return pred.gather(1,s.reshape(-1,1)).squeeze(1)


def local_loss(model,x,q,center,sensor,offset):
    y0=chosen_sensor(model,x,center,sensor)
    yt=chosen_sensor(model,x,q,sensor)
    z=y0.square().mean()
    nonzero=offset.abs()>1e-7
    if nonzero.any():
        d=(((yt[nonzero]-y0[nonzero]-offset[nonzero])/0.02).square()).mean()
    else:
        d=(yt*0).sum()
    return z,d


@torch.no_grad()
def local_eval(model,torch_data,selected,batch=1024):
    x,q,center,sensor,offset=torch_data
    model.eval()
    zero=delta=0.0
    count=nonzero_count=0
    for a in range(0,len(selected),batch):
        ids=selected[a:a+batch]
        indices=torch.as_tensor(ids,device=x.device,dtype=torch.long)
        y0=chosen_sensor(model,x[indices],center[indices],sensor[indices])
        y1=chosen_sensor(model,x[indices],q[indices],sensor[indices])
        t=offset[indices]
        zero+=float(y0.square().sum().cpu())
        nz=t.abs()>1e-7
        if nz.any():
            delta+=float(((y1[nz]-y0[nz]-t[nz]).square()).sum().cpu())
            nonzero_count+=int(nz.sum().item())
        count+=len(ids)
    require(count>0 and nonzero_count>0,"Insufficient V4 validation examples")
    return dict(val_tube_subset_count=count,
                val_tube_zero_rmse=math.sqrt(zero/count),
                val_tube_delta_rmse_rad=math.sqrt(delta/nonzero_count))


def atomic_pt(out,state):
    tmp=out.with_name(out.name+f".tmp.{os.getpid()}")
    try:
        torch.save(state,tmp)
        os.replace(tmp,out)
    finally:
        if tmp.exists():tmp.unlink()


def run(args):
    require(args.mode in ("v3","joint"),"Unknown training mode")
    require(torch.cuda.is_available(),"No CUDA GPU visible")
    require(args.steps==1000 and args.batch==512 and args.eval_every==100
            and args.seed==0 and args.lr==1e-3,
            "Frozen experiment requires 1000 steps, batch512, seed0, lr1e-3")
    require(args.tube_batch==512 and args.union_weight==0.2
            and args.zero_weight==0.2 and args.delta_weight==0.05,
            "Frozen joint loss weights or batch changed")
    root=args.root.resolve()
    out=output_path(root,args.mode)
    require(not out.exists(),"Training output exists; do not overwrite")
    inp,target,valid,split,ix,tubes,meta=load_data(root,args.mode)
    tr=np.flatnonzero(split==0)
    va=np.flatnonzero(split==1)
    require(len(tr)==6804 and len(va)==1388,"Train/Val V3 query counts changed")
    if args.mode=="joint":
        xx,tq,t0,ss,toff,tsplit,txi=tubes
        tidx=np.flatnonzero(tsplit==0)
        vidx=np.flatnonzero(tsplit==1)
        require(len(tidx)>0 and len(vidx)>0,"Empty tube train or val split")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda:0")
    model=original.load_model().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    xb=torch.from_numpy(inp).to(device)
    yb=torch.from_numpy(target).to(device)
    mb=torch.from_numpy(valid).to(device)
    trb=torch.from_numpy(tr.astype(np.int64)).to(device)
    if args.mode=="joint":
        tx=torch.from_numpy(xx).to(device)
        tqb=torch.from_numpy(tq).to(device)
        tcb=torch.from_numpy(t0).to(device)
        tsb=torch.from_numpy(ss).to(device)
        to=torch.from_numpy(toff).to(device)
        tb=torch.from_numpy(tidx.astype(np.int64)).to(device)
        aux_rng=torch.Generator(device=device)
        aux_rng.manual_seed(20261113)
        evrng=np.random.default_rng(260929)
        eval_sub=np.sort(evrng.choice(vidx,size=min(8192,len(vidx)),replace=False))
        tubet=(tx,tqb,tcb,tsb,to)
    out.mkdir(parents=True,exist_ok=False)
    cfg=dict(mode=args.mode,steps=args.steps,batch=args.batch,eval_every=args.eval_every,
             seed=args.seed,lr=args.lr,union_weight=args.union_weight,
             tube_batch=args.tube_batch,zero_weight=args.zero_weight,
             delta_weight=args.delta_weight)
    r=dict(format=FORMAT,status="RUNNING",
           config=cfg,model_architecture=model.architecture(),
           model_source_sha256=digest(original.MODEL_SRC),
           training_code_sha256=digest(__file__),
           dataset=str(dataset_path(root)),
           dataset_v3_sha256=meta["v3.npz_sha256"],
           dataset_v4_sha256=meta["v4.npz_sha256"],
           train_x=1701,val_x=347,train_queries=6804,val_queries=1388,
           valid_sensor_labels=48455,
           gradient_target_fd_pass=False,union_target_exact=False,
           training_ready_full_grid25=False,
           qualification="Segment0+1 supervised data expansion / loss ablation, not final full-grid H9")
    (out/"run.json").write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    best=float("inf")
    tstart=time.monotonic()
    with (out/"metrics.jsonl").open("x",buffering=1) as metrics:
        for step in range(1,args.steps+1):
            model.train()
            chosen=trb[torch.randint(len(trb),(args.batch,),device=device)]
            optimizer.zero_grad(set_to_none=True)
            pred=model(xb[chosen]).float()
            v3,sensor,proxy,_=original.supervised_loss(
                pred,yb[chosen],mb[chosen],args.union_weight)
            loss=v3
            loss0=lossd=None
            if args.mode=="joint":
                # Separate generator keeps V3 sampler identical across both arms.
                chosen_t=tb[torch.randint(len(tb),(args.tube_batch,),
                                            generator=aux_rng,device=device)]
                loss0,lossd=local_loss(model,tx[chosen_t],tqb[chosen_t],
                                         tcb[chosen_t],tsb[chosen_t],to[chosen_t])
                loss=v3+args.zero_weight*loss0+args.delta_weight*lossd
            require(bool(torch.isfinite(loss).item()),f"Nonfinite loss at {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            if step==1 or step%args.eval_every==0 or step==args.steps:
                ev=original.evaluate(model,inp,target,valid,split,args.batch,device)
                row=dict(step=step,train_batch_total_mse=float(loss.detach().cpu()),
                         train_batch_v3_sensor_mse=float(sensor.cpu()),
                         train_batch_v3_union_proxy_mse=float(proxy.cpu()),
                         **ev,elapsed_s=round(time.monotonic()-tstart,2))
                if args.mode=="joint":
                    row.update(
                        train_batch_v4_zero_mse=float(loss0.detach().cpu()),
                        train_batch_v4_delta_normalized_mse=float(lossd.detach().cpu()))
                    row.update(local_eval(model,tubet,eval_sub))
                require(math.isfinite(row["val_sensor_rmse"]), "Nonfinite validation RMSE")
                metrics.write(json.dumps(row,sort_keys=True)+"\n")
                print("[SEG01 TRAIN]",args.mode,
                      json.dumps({k:v for k,v in row.items() if k!="by_sensor"},sort_keys=True),
                      flush=True)
                state=dict(format=FORMAT,mode=args.mode,step=step,
                           model_state=model.state_dict(),
                           optimizer_state=optimizer.state_dict(),
                           config=cfg,dataset_v3_sha256=meta["v3.npz_sha256"],
                           dataset_v4_sha256=meta["v4.npz_sha256"],
                           val_sensor_rmse=ev["val_sensor_rmse"],
                           original_r1_model_sha256=r["model_source_sha256"],
                           gradient_fd_verified=False,union_target_is_proxy=True)
                atomic_pt(out/"latest.pt",state)
                if row["val_sensor_rmse"]<best:
                    best=row["val_sensor_rmse"]
                    atomic_pt(out/"best.pt",state)
    os.replace(out/"latest.pt",out/"final.pt")
    r.update(status="COMPLETE",successful_updates=args.steps,
             best_val_sensor_rmse=best,
             best_sha256=digest(out/"best.pt"),
             final_sha256=digest(out/"final.pt"),
             metrics_sha256=digest(out/"metrics.jsonl"),
             elapsed_s=round(time.monotonic()-tstart,2))
    tmp=out/"run.json.tmp"
    tmp.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,out/"run.json")
    print("[SEG01 TRAIN COMPLETE]",args.mode,
          "best_rmse=",round(best,6),flush=True)


def compare(root):
    root=root.resolve()
    output={}
    for mode in ("v3","joint"):
        folder=output_path(root,mode)
        r=read(folder/"run.json")
        require(r.get("format")==FORMAT and r.get("status")=="COMPLETE"
                and r.get("successful_updates")==1000
                and r["config"]["mode"]==mode
                and digest(folder/"best.pt")==r["best_sha256"]
                and digest(folder/"final.pt")==r["final_sha256"]
                and digest(folder/"metrics.jsonl")==r["metrics_sha256"],
                f"Run {mode} incomplete or modified")
        output[mode]=r
    a=output["v3"];b=output["joint"]
    require(a["dataset_v3_sha256"]==b["dataset_v3_sha256"]
            and a["model_source_sha256"]==b["model_source_sha256"]
            and all(a["config"][k]==b["config"][k] for k in (
                "steps","batch","seed","lr","eval_every","union_weight")),
            "V3 vs V3+V4 cannot be directly compared")
    improvement=a["best_val_sensor_rmse"]-b["best_val_sensor_rmse"]
    print("[SEG01 COMPARISON VERIFIED]")
    print(json.dumps(dict(
        v3_only_best_val_rmse=a["best_val_sensor_rmse"],
        v3_v4_joint_best_val_rmse=b["best_val_sensor_rmse"],
        absolute_improvement=improvement,
        relative_improvement_pct=100*improvement/a["best_val_sensor_rmse"],
        better="joint" if improvement>0 else "v3_only" if improvement<0 else "equal",
        training_ready_full_grid25=False,
    ),indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("v3","joint","compare"))
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--batch",type=int,default=512)
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--union-weight",type=float,default=0.2)
    p.add_argument("--tube-batch",type=int,default=512)
    p.add_argument("--zero-weight",type=float,default=0.2)
    p.add_argument("--delta-weight",type=float,default=0.05)
    args=p.parse_args()
    compare(args.root) if args.mode=="compare" else run(args)


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,KeyError,OSError,FileNotFoundError) as e:
        raise SystemExit("[STOP] "+str(e))
