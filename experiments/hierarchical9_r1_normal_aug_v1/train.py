#!/usr/bin/env python3
"""R1 50k controlled augmentation: retain ORIGINAL R1 loop; add ONE normal-direction loss.

Never load R1 weights as initialization. R012's exact base sampling/objective,
optimizer, plateau schedule, AMP and original 50k train-stream SHA are preserved.
No V3/V4 new distance targets, no boundary-zero / unit-slope / margins.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('R1A_REPO', str(HERE.parents[1]))).resolve()
R012 = REPO/"experiments/hierarchical9_scratch50k_r012_v1"
SCRATCH = REPO/"experiments/hierarchical9_scratch_v1"
V4DATA = REPO/"experiments/hierarchical9_paired_slope_scratch_v1/data.py"
FORMAT = "care_h9_r1_boundary_direction_aug_v1"
ORIGINAL_R1_SHA = "4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002"

from boundary_aug import BoundaryDirection, attach_to_original_r1


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):
            h.update(b)
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def json_atomic(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def protocol():
    d=json_read(HERE/"protocol.json")
    if (d["format"]!=FORMAT or d["steps"]!=50000 or
        not 0<float(d["normal_direction_weight"])<=0.02 or
        d["anchors_per_sensor_per_rank"] !=32 or
        d["baseline"]["global_batch_x"] !=4000 or
        d["baseline"]["batch_q"] !=100 or
        d["baseline"]["amp"]!="fp16"):
        raise ValueError("Unexpected controlled augmentation protocol")
    return d


def dependency_fingerprints():
    paths=list(HERE.glob("*.py"))+list(HERE.glob("*.sh"))+[HERE/"protocol.json",
        R012/"train.py",R012/"model.py",SCRATCH/"train.py",
        SCRATCH/"objective.py",SCRATCH/"model.py",V4DATA]
    return {("experiments/hierarchical9_r1_normal_aug_v1/"+p.name)
                if p.parent.resolve()==HERE.resolve() else str(p.resolve().relative_to(REPO)):
            sha(p) for p in sorted(paths)}


# Freeze the actual source code used by the historical R1 baseline. Git blob
# hashes are content-addressed and independent of checkout directory paths.
R1_ORIGINAL_SOURCE_GIT_BLOBS = {
    "r012/train.py": "8092a791585dff8ac6ab5e562d158450b1095df4",
    "r012/model.py": "0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b",
    "scratch/train.py": "88d365045bb1c53cf71f5b8fbd9f8db74a5c2ae9",
    "scratch/objective.py": "a53539d27fb003d6b06d46da80ca0c7a93b1230c",
    "scratch/model.py": "33580154af2e9cda22f9c64f15cd5b4f5417a86f",
}


def verify_original_r1_code():
    mapping = {
        "r012/train.py": R012/"train.py",
        "r012/model.py": R012/"model.py",
        "scratch/train.py": SCRATCH/"train.py",
        "scratch/objective.py": SCRATCH/"objective.py",
        "scratch/model.py": SCRATCH/"model.py",
    }
    for key, path in mapping.items():
        data=path.read_bytes()
        gitsha=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        if gitsha!=R1_ORIGINAL_SOURCE_GIT_BLOBS[key]:
            raise ValueError(f"Original R1 source changed: {key} ({gitsha})")


def baseline(r012_root):
    root=Path(r012_root)/"formal/R1"
    checkpoint=root/"final.pt"
    run=json_read(root/"run.json")
    if sha(checkpoint)!=ORIGINAL_R1_SHA:
        raise ValueError("Wrong original R1 checkpoint SHA256")
    for key,expected in (("status","COMPLETE"),("arm","R1"),
                         ("successful_updates",50000),
                         ("final_sha256",ORIGINAL_R1_SHA),
                         ("training_stream_updates",50000)):
        if run.get(key)!=expected:
            raise ValueError(f"R1 provenance mismatch for {key}")
    return run


def preflight(args):
    verify_original_r1_code()
    p=protocol()
    base=baseline(args.r012_root)
    if not Path(args.data).is_file() or not Path(args.urdf).is_file():
        raise FileNotFoundError("Original R1 dataset/URDF unavailable")
    mod_path=V4DATA
    spec=importlib.util.spec_from_file_location("aug_preflight_cache_reader",mod_path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cache=mod.Cache(args.cache,verify=True)
    if cache.identity!=p["v4_cache_sha256"]:
        raise ValueError("Augmentation cache identity mismatch")
    # PRE-SLURM REAL-DATA GATE: reproduce R1's exact x/valid-split mapping,
    # inspecting original .npz arrays on CPU rather than trusting fake fixtures.
    with np.load(args.data,allow_pickle=True) as source:
        original_x=np.asarray(source["x"],np.float32)
        valid_any=np.asarray(source["valid_fov"],bool).any(axis=(1,2))
    if original_x.shape!=cache.x.shape or np.max(np.abs(original_x-cache.x))>1e-6:
        raise ValueError("Original R1/V4 x-index mapping mismatch BEFORE GPU submission")
    ids=np.flatnonzero(valid_any)
    rng=np.random.default_rng(int(p["baseline"]["seed"]))
    rng.shuffle(ids)
    nval=min(int(p["baseline"]["val_count"]),max(1,len(ids)//10))
    xtrain,xval=ids[nval:],ids[:nval]
    for split,allowed in (("train",xtrain),("val",xval)):
        a=cache.a[split]["boundary"]
        mask=np.isin(a["x_index"],allowed)
        cell=np.bincount(np.asarray(a["sensor"][mask],np.int64),minlength=8)
        if np.any(cell==0):
            raise ValueError(f"No {split} boundary anchors aligned to R1 x split: {cell.tolist()}")
        print(f"[preflight] R1 {split} x aligned; 8 sensor normal anchors={cell.tolist()}",flush=True)
    print("[preflight] original R1 COMPLETE and checksum verified",flush=True)
    print("[preflight] frozen cache verified; boundary train/val x split checked at launch",flush=True)
    print("[preflight] R1 stream reference:",base["training_stream_sha256"],flush=True)
    print("[preflight] original 400k Cartesian pairs/update + 4*8*32 normals/update",flush=True)
    return p,base,cache


def load_r012_module():
    name="r1_direction_aug_frozen_original_trainer"
    spec=importlib.util.spec_from_file_location(name,R012/"train.py")
    if spec is None or spec.loader is None:
        raise ImportError(R012/"train.py")
    trainer=importlib.util.module_from_spec(spec)
    sys.modules[name]=trainer
    spec.loader.exec_module(trainer)
    return trainer


def train(args):
    cfg,base_run,_=preflight(args)
    out=Path(args.out).resolve()
    expected_stream=base_run["training_stream_sha256"]
    fingerprint=dependency_fingerprints()
    if args.resume:
        checkpoint=Path(args.resume).resolve()
        if checkpoint!=out/"latest.pt":
            raise ValueError("Resume ONLY the same new experiment latest.pt")
        cp=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if cp.get("format")!=FORMAT or cp.get("source_sha256")!=fingerprint:
            raise ValueError("Resume source/format mismatch")
        augmeta=cp.get("metadata",{}).get("boundary_direction_augmentation")
        if augmeta is None or augmeta["protocol"]!=cfg or augmeta["cache_identity"]!=cfg["v4_cache_sha256"]:
            raise ValueError("Resume augmentation protocol/cache mismatch")
        start_step=int(cp.get("step",-1))
        if not 0 < start_step < cfg["steps"]:
            raise ValueError("No resumable updates left")
    else:
        start_step=0
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("New formal output already exists; refuse overwrite")
    # Keep R012.main and all upstream modules unchanged. Only monkeypatch its
    # data-independent train step at a single, explicitly audited boundary.
    trainer=load_r012_module()
    trainer.FORMAT=FORMAT

    def current_fingerprints():
        found=dependency_fingerprints()
        if found!=fingerprint:
            raise RuntimeError("Training dependency/source files changed during run")
        return found
    trainer.source_fingerprints=current_fingerprints

    holder={"aug":None}
    original_prepare=trainer.base.prepare_batch
    def prepared(api,dataset,oracle,original_args,device,split):
        if holder["aug"] is None:
            holder["aug"]=BoundaryDirection(REPO,Path(args.cache),dataset,cfg)
            holder["aug"].step=start_step
            print("[aug] verified R1 x-index alignment, R1 train-only boundary sampling",
                  holder["aug"].counts,flush=True)
        return original_prepare(api,dataset,oracle,original_args,device,split)
    trainer.base.prepare_batch=prepared

    # Wrapper retrieves holder at each call, after the first original batch
    # initialized sampler; model construction/stream reset are untouched.
    original_run=trainer.base.run_prepared_batch
    # Install the augmentation on the first real prepared batch only.
    def ensure_aug(*a,**kw):
        if holder["aug"] is None: raise RuntimeError("Missing augmentation")
        if not holder.get("installed"):
            trainer.base.run_prepared_batch=original_run
            attach_to_original_r1(trainer,holder["aug"],cfg)
            holder["installed"]=True
        return trainer.base.run_prepared_batch(*a,**kw)
    trainer.base.run_prepared_batch=ensure_aug

    old_save=trainer.save_checkpoint
    def save(path,ddp,optimizer,scheduler,scaler,orig_args,step,best_val,stats,stream,metadata):
        if holder["aug"] is None:
            raise RuntimeError("No verified sampler at checkpoint time")
        metadata={**metadata,"boundary_direction_augmentation":{
            "protocol":cfg,"cache_identity":holder["aug"].cache_identity,
            "R1_baseline_final_sha256":ORIGINAL_R1_SHA,
            "R1_baseline_stream_sha256":expected_stream,
            "R1_x_aligned":True,"train_split_excludes_R1_val_x":True,
            "additional_optimizer_steps":0,
            "objective":"original_R1 + normal_direction_weight*(1-cosine)",
            "scheduler_selection":"original_R1_objective_only"}}
        return old_save(path,ddp,optimizer,scheduler,scaler,
                        orig_args,step,best_val,stats,stream,metadata)
    trainer.save_checkpoint=save

    argv=["original_R012_train.py","--arm","R1","--repo",str(REPO),
          "--data",str(args.data),"--urdf",str(args.urdf),
          "--out-dir",str(out)]
    b=cfg["baseline"]
    argv.extend(["--steps",str(cfg["steps"]),"--seed",str(b["seed"]),
                 "--stream-seed",str(b["stream_seed"]),"--lr",str(b["lr"]),
                 "--amp",b["amp"],"--global-batch-x",str(b["global_batch_x"]),
                 "--batch-q",str(b["batch_q"]),"--microbatch-x",str(b["microbatch_x"]),
                 "--val-global-batch-x",str(b["val_global_batch_x"]),
                 "--val-batch-q",str(b["val_batch_q"]),
                 "--val-microbatch-x",str(b["val_microbatch_x"]),
                 "--val-count",str(b["val_count"]),
                 "--decode-x-chunk",str(b["decode_x_chunk"]),
                 "--weight-sdf",str(b["weight_sdf"]),
                 "--weight-grad",str(b["weight_grad"]),
                 "--weight-eikonal",str(b["weight_eikonal"]),
                 "--weight-tension",str(b["weight_tension"]),
                 "--weight-union-objective",str(b["weight_union_objective"]),
                 "--weight-sensor-objective",str(b["weight_sensor_objective"]),
                 "--weight-consistency",str(b["weight_consistency"]),
                 "--log-every","100","--val-every","1000","--save-every","5000"])
    if args.resume:
        argv.extend(["--resume-training",str(args.resume)])
    sys.argv=argv
    trainer.main()
    if not (out/"final.pt").is_file():
        raise RuntimeError("R1 augmented loop returned without final checkpoint")
    run=json_read(out/"run.json")
    run.update(variant=FORMAT,augmentation=cfg,
               original_r1_final_sha256=ORIGINAL_R1_SHA,
               original_r1_training_stream_sha256=expected_stream,
               boundary_cache_sha256=cfg["v4_cache_sha256"],
               boundary_groups=holder["aug"].counts,
               original_loss_scheduler=True)
    if run.get("training_stream_sha256")!=expected_stream:
        run["status"]="INVALID_ORIGINAL_R1_STREAM"
        json_atomic(out/"run.json",run)
        raise RuntimeError("R1 original x/q stream changed: comparison INVALID")
    run["baseline_stream_exact_match"]=True
    json_atomic(out/"run.json",run)
    print("[verified] 50k exact original R1 training stream, auxiliary-only change",flush=True)


def verify(args):
    cfg, old, _=preflight(args)
    out=Path(args.out)
    run=json_read(out/"run.json")
    if (run.get("status")!="COMPLETE" or run.get("successful_updates")!=50000 or
        run.get("variant")!=FORMAT or not run.get("baseline_stream_exact_match") or
        run["training_stream_sha256"]!=old["training_stream_sha256"]):
        raise ValueError("Augmented run not formally comparable")
    digest=sha(out/"final.pt")
    if digest!=run.get("final_sha256"):
        raise ValueError("Final augmented checkpoint checksum mismatch")
    cp=torch.load(out/"final.pt",map_location="cpu",weights_only=False)
    if (cp.get("format")!=FORMAT or cp.get("step")!=50000 or
        cp.get("initialization")!="random_from_scratch" or
        cp.get("source_sha256")!=dependency_fingerprints() or
        cp.get("metadata",{}).get("boundary_direction_augmentation",{}).get("protocol")!=cfg):
        raise ValueError("Augmented checkpoint provenance mismatch")
    print(f"[verified] R1+boundary formal COMPLETE @50k; exact base stream; final={digest}",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("command",choices=("preflight","train","verify"))
    ap.add_argument("--r012-root",type=Path,required=True)
    ap.add_argument("--cache",type=Path,required=True)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--urdf",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--resume",type=Path,default=None)
    args=ap.parse_args()
    if args.command=="preflight":preflight(args)
    elif args.command=="train":train(args)
    else:verify(args)

if __name__=="__main__":main()
