#!/usr/bin/env python3
"""First actual Grid25 supervised H9 GPU training on verified segment-0 V3 values.

Uses original R1 nine-head architecture; sensor heads have V3 distance labels.
Union head only receives a transparently declared max-of-8 pseudo-target for
queries with all eight sensors V3-valid. No unverified q-gradient or tube offsets.
This is a *first supervised experiment*, not the old full original H9 objective.
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
MODEL_SRC = REPO / "experiments/hierarchical9_scratch50k_r012_v1/model.py"
FORMAT = "care_grid25_h9_r1_v3_distance_first_train_v1"


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_model():
    spec = importlib.util.spec_from_file_location("grid25_h9_original_r1", MODEL_SRC)
    require(spec is not None and spec.loader is not None, "Original H9 R1 model missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PrivateTailCDF()


def load_snapshot(dataset):
    dataset = dataset.resolve()
    meta = json.loads((dataset / "manifest.json").read_text())
    require(meta.get("format") == "care_grid25_segment0_v3_sensor_distance_train_v1"
            and meta.get("status") == "FROZEN_DISTANCE_ONLY",
            "Wrong label-adapter snapshot")
    require(sha(dataset / "labels.npz") == meta["labels_sha256"],
            "Training snapshot hash mismatch")
    with np.load(dataset/"labels.npz", allow_pickle=False) as z:
        x, q, split, target, valid, x_index = [
            np.asarray(z[k]).copy() for k in
            ("x","q","split","target","valid","x_index")]
    require(x.shape == (len(x), 3) and q.shape == (len(x), 7) and
            target.shape == valid.shape == (len(x), 8) and
            split.shape == x_index.shape == (len(x),),
            "Training snapshot shape mismatch")
    require(np.isfinite(x).all() and np.isfinite(q).all()
            and np.isfinite(target[valid]).all(),
            "Training inputs/targets nonfinite")
    require(np.isin(split,[0,1]).all()
            and not (set(x_index[split==0]) & set(x_index[split==1])),
            "Train/val spatial split leakage")
    require(int(valid.sum()) == meta["counts"]["valid_sensor_labels"],
            "Label count/provenance mismatch")
    inputs = np.concatenate((x,q), axis=1).astype(np.float32)
    # All invalid label slots are zero-filled in the frozen snapshot.
    targets = np.asarray(target, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    return inputs, targets, valid, split, meta


def evaluate(model, inputs, targets, masks, split, batch, device):
    chosen = np.flatnonzero(split == 1)
    require(len(chosen) > 0, "Empty val split")
    model.eval()
    sum_sq, sum_abs, sign_correct, count = np.zeros((4,8), np.float64)
    union_sq=union_abs=union_count=0.
    with torch.no_grad():
        for start in range(0, len(chosen), batch):
            ids = chosen[start:start+batch]
            x = torch.as_tensor(inputs[ids], device=device)
            m = torch.as_tensor(masks[ids], device=device)
            y = torch.as_tensor(targets[ids], device=device)
            out = model(x).float()
            z = out[:,1:]
            err = z - y
            sum_sq += ((err.square()*m).sum(0)).detach().cpu().numpy()
            sum_abs += ((err.abs()*m).sum(0)).detach().cpu().numpy()
            sign_correct += ((((z>=0)==(y>=0))&m).sum(0)).detach().cpu().numpy()
            count += m.sum(0).detach().cpu().numpy()
            both = m.all(dim=1)
            if both.any():
                pseudo = y[both].max(dim=1).values
                diff = out[both,0] - pseudo
                union_sq += float(diff.square().sum().cpu())
                union_abs += float(diff.abs().sum().cpu())
                union_count += int(both.sum().item())
    require(np.all(count>0), "At least one validation sensor has no valid labels")
    by_sensor = {}
    for s in range(8):
        by_sensor[f"S{s}"] = dict(count=int(count[s]),
                                 rmse=math.sqrt(sum_sq[s]/count[s]),
                                 mae=sum_abs[s]/count[s],
                                 sign_accuracy=sign_correct[s]/count[s])
    total=int(count.sum())
    return {
        "val_sensor_count":total,
        "val_sensor_rmse":math.sqrt(float(sum_sq.sum()/count.sum())),
        "val_sensor_mae":float(sum_abs.sum()/count.sum()),
        "val_sensor_sign_accuracy":float(sign_correct.sum()/count.sum()),
        "val_union_proxy_count":int(union_count),
        "val_union_proxy_rmse":(math.sqrt(union_sq/union_count)
                                if union_count else None),
        "val_union_proxy_mae":(union_abs/union_count if union_count else None),
        "by_sensor":by_sensor,
    }


def supervised_loss(output, y, mask, union_weight):
    require(output.shape == (len(y),9) and y.shape == mask.shape == (len(y),8),
            "Invalid 9-head H9 batch")
    pred=output[:,1:].float()
    n=mask.sum()
    require(n.item()>0, "Sampled training batch contains no valid V3 sensor labels")
    sensor=((pred-y).square()*mask).sum()/n
    both=mask.all(dim=1)
    union=(output[:,0]*0.).sum()
    if union_weight>0 and both.any():
        pseudo=y[both].max(dim=1).values
        union=(output[both,0].float()-pseudo).square().mean()
    return sensor+union_weight*union, sensor.detach(), union.detach(), int(both.sum().item())


def save_atomic(path, data):
    path=Path(path)
    tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
    try:
        torch.save(data,tmp)
        os.replace(tmp,path)
    finally:
        if tmp.exists():
            tmp.unlink()


def train(args):
    require(torch.cuda.is_available(), "First Grid25 H9 trainer needs CUDA GPU")
    require(args.steps>=1 and 16<=args.batch<=8192 and args.eval_every>=1,
            "Invalid training schedule")
    require(math.isfinite(args.lr) and 0<args.lr<=0.01
            and math.isfinite(args.union_proxy_weight)
            and 0<=args.union_proxy_weight<=1,
            "Invalid learning rate/union proxy weight")
    data, out = args.dataset.resolve(), args.out.resolve()
    require(not out.exists(), "Run output exists; do not overwrite/resume implicitly")
    inputs,targets,valid,split,meta=load_snapshot(data)
    train_ids = np.flatnonzero(split==0)
    val_ids = np.flatnonzero(split==1)
    require(len(train_ids)==meta["counts"]["train_queries"]
            and len(val_ids)==meta["counts"]["val_queries"],
            "Train/val row count mismatch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda:0")
    model=load_model().to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=args.lr)
    inputs_d=torch.from_numpy(inputs).to(device)
    targets_d=torch.from_numpy(targets).to(device)
    valid_d=torch.from_numpy(valid).to(device)
    train_idx_d=torch.from_numpy(train_ids.astype(np.int64)).to(device)
    out.mkdir(parents=True, exist_ok=False)
    source = {
        "format":FORMAT, "status":"RUNNING",
        "dataset":str(data), "dataset_labels_sha256":meta["labels_sha256"],
        "model_source":str(MODEL_SRC.relative_to(REPO)), "model_sha256":sha(MODEL_SRC),
        "trainer_source_sha256":sha(__file__),
        "architecture":model.architecture(),
        "config":dict(steps=args.steps,batch=args.batch,lr=args.lr,
                      union_proxy_weight=args.union_proxy_weight,
                      seed=args.seed,eval_every=args.eval_every),
        "supervision":"V3 valid value only; original H9 gradient/Eikonal/tension losses disabled",
        "union_target":"max of eight V3 values when all eight valid; proxy, not certified exact union SDF",
        "train_queries":int(len(train_ids)),
        "val_queries":int(len(val_ids)),
        "training_ready_full_grid25":False,
    }
    (out/"run.json").write_text(json.dumps(source,indent=2)+"\n")
    metrics_path=out/"metrics.jsonl"
    best=float("inf")
    start=time.monotonic()
    torch.backends.cuda.matmul.allow_tf32=True
    with metrics_path.open("x", buffering=1) as metrics:
        for step in range(1,args.steps+1):
            model.train()
            sample=torch.randint(len(train_idx_d),(args.batch,),device=device)
            idx=train_idx_d[sample]
            optimizer.zero_grad(set_to_none=True)
            out_pred=model(inputs_d[idx])
            loss,sensor,union,nunion=supervised_loss(
                out_pred,targets_d[idx],valid_d[idx],args.union_proxy_weight)
            require(torch.isfinite(loss).item(),f"Nonfinite train loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.)
            optimizer.step()
            if step==1 or step%args.eval_every==0 or step==args.steps:
                evals=evaluate(model,inputs,targets,valid,split,
                               args.batch,device)
                row={
                    "step":step, "train_batch_total_mse":float(loss.detach().cpu()),
                    "train_batch_sensor_mse":float(sensor.cpu()),
                    "train_batch_union_proxy_mse":float(union.cpu()),
                    "train_batch_union_proxy_count":nunion,
                    **evals,
                    "elapsed_s":round(time.monotonic()-start,2),
                }
                metrics.write(json.dumps(row,sort_keys=True)+"\n")
                print("[TRAIN]",json.dumps({k:v for k,v in row.items()
                      if k not in ("by_sensor",)},sort_keys=True),flush=True)
                require(math.isfinite(row["val_sensor_rmse"]),
                        "Nonfinite validation RMSE")
                state=dict(
                    format=FORMAT,step=step,model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    config=source["config"],dataset_labels_sha256=meta["labels_sha256"],
                    model_source_sha256=source["model_sha256"],
                    trainer_source_sha256=source["trainer_source_sha256"],
                    val_sensor_rmse=row["val_sensor_rmse"],
                    val_sensor_mae=row["val_sensor_mae"],
                    union_head_label_is_proxy=True,
                    gradients_independently_fd_verified=False,
                )
                save_atomic(out/"latest.pt",state)
                if row["val_sensor_rmse"]<best:
                    best=row["val_sensor_rmse"]
                    save_atomic(out/"best.pt",state)
    # Final checkpoint is the last update; best.pt holds best validation snapshot.
    os.replace(out/"latest.pt",out/"final.pt")
    source.update(
        status="COMPLETE",successful_updates=args.steps,
        best_val_sensor_rmse=best,
        final_sha256=sha(out/"final.pt"),best_sha256=sha(out/"best.pt"),
        metrics_sha256=sha(metrics_path),
        elapsed_s=round(time.monotonic()-start,2))
    tmp=out/"run.json.tmp"
    tmp.write_text(json.dumps(source,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,out/"run.json")
    print("[H9 FIRST GRID25 V3 TRAIN COMPLETE] steps=",args.steps,
          "best_val_sensor_rmse=",round(best,6),flush=True)
    print("[CHECKPOINT]",out/"best.pt",flush=True)
    print("[NOT FULL H9 QUALIFICATION] No verified gradient or exact union labels",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--batch",type=int,default=512)
    ap.add_argument("--eval-every",type=int,default=100)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--union-proxy-weight",type=float,default=0.2)
    args=ap.parse_args()
    train(args)


if __name__=="__main__":
    try:
        main()
    except (RuntimeError,ValueError,OSError,KeyError,FileNotFoundError) as e:
        raise SystemExit("[STOP] "+str(e))
