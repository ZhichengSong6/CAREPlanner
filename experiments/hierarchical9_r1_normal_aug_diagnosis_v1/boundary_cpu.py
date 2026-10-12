"""CPU-only same-point boundary comparison of frozen original R1 and R1+Normal.

The V4 "val" subset is intersected with ORIGINAL R1 val x indices.
No sampling from R1 training x, no geometry regeneration, no optimizer updates.
The forward score is intentionally NOT forced to zero: original R1's
distance scale differs from V4 local normal offsets.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from diagnose import EXPECTED_R1, VARIANT, sha, read_json, numeric_stats, finite


def load_r1_architecture(repo):
    path=repo/"experiments/hierarchical9_scratch50k_r012_v1/model.py"
    spec=importlib.util.spec_from_file_location("frozen_r1_normal_diagnosis_model",path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def model_pair(repo,r012,aug):
    base=r012/"formal/R1/final.pt"
    new=aug/"formal/final.pt"
    baserun=read_json(r012/"formal/R1/run.json")
    newrun=read_json(aug/"formal/run.json")
    if (sha(base)!=EXPECTED_R1 or
        baserun.get("final_sha256")!=EXPECTED_R1 or
        sha(new)!=newrun.get("final_sha256") or
        newrun.get("variant")!=VARIANT or
        newrun.get("baseline_stream_exact_match") is not True or
        newrun.get("original_r1_final_sha256")!=EXPECTED_R1 or
        baserun.get("training_stream_sha256")!=newrun.get("training_stream_sha256")):
        raise ValueError("Checkpoint validation failed for boundary CPU probe")
    arch=load_r1_architecture(repo)
    models={}
    digests={}
    # Only load trusted local checkpoints that have passed the exact SHA guard.
    for name,path in (("R1",base),("R1_AUG",new)):
        cp=torch.load(path,map_location="cpu",weights_only=False)
        if not isinstance(cp.get("model_state"),dict) or cp.get("step")!=50000:
            raise ValueError(f"Invalid frozen checkpoint {name}")
        model=arch.build_model("R1").to("cpu",dtype=torch.float32)
        model.load_state_dict(cp["model_state"],strict=True)
        model.eval().requires_grad_(False)
        models[name]=model
        digests[name]=sha(path)
    return models,digests,baserun,newrun


def original_val_x(r012):
    run=read_json(r012/"formal/R1/run.json")
    ckpt=r012/"formal/R1/final.pt"
    cp=torch.load(ckpt,map_location="cpu",weights_only=False)
    if cp.get("arm")!="R1" or cp.get("step")!=50000:
        raise ValueError("Wrong R1 baseline")
    args=cp["args"]
    path=Path(args["data"])
    with np.load(path,allow_pickle=True) as bank:
        x=np.asarray(bank["x"],dtype=np.float32)
        valid=np.asarray(bank["valid_fov"],dtype=bool).any(axis=(1,2))
    ids=np.flatnonzero(valid)
    np.random.default_rng(int(args["seed"])).shuffle(ids)
    nval=min(int(args["val_count"]),max(1,len(ids)//10))
    val=ids[:nval].astype(np.int64)
    if not len(val): raise ValueError("Empty original R1 val")
    return x,val,{"data":str(path),"val_count":nval,"seed":int(args["seed"])}


def choose(cache,r012,per_sensor,seed):
    manifest=read_json(cache/"manifest.json")
    if (manifest.get("format")!="care_h9_v4_training_cache_v1" or
        manifest.get("status")!="COMPLETE" or
        manifest.get("training_ready") is not True):
        raise ValueError("Bad V4 training cache")
    if sha(cache/"manifest.json")!="060eef93a0030800c8ef2265258c7dff850944ab9c5df52382adb98cb506f7cc":
        raise ValueError("Unexpected frozen V4 manifest SHA")
    x_orig,val,origin=original_val_x(r012)
    x=np.load(cache/"x.npy",mmap_mode="r",allow_pickle=False)
    if x.shape!=x_orig.shape or np.max(np.abs(x_orig-x))>1e-6:
        raise ValueError("R1/V4 x-index geometry mismatch")
    root=cache/"val"
    x_ids=np.load(root/"boundary_x_index.npy",mmap_mode="r",allow_pickle=False)
    sensors=np.load(root/"boundary_sensor.npy",mmap_mode="r",allow_pickle=False)
    qs=np.load(root/"boundary_q.npy",mmap_mode="r",allow_pickle=False)
    normals=np.load(root/"boundary_grad.npy",mmap_mode="r",allow_pickle=False)
    if not (len(x_ids)==len(sensors)==len(qs)==len(normals)):
        raise ValueError("Boundary val arrays have incompatible lengths")
    accepted=np.isin(x_ids,val)
    ids=[]
    for sensor in range(8):
        pool=np.flatnonzero(accepted&(sensors==sensor))
        if len(pool)<per_sensor:
            raise ValueError(f"Insufficient R1-val V4 boundary anchors: S{sensor}={len(pool)}")
        rng=np.random.default_rng(np.random.SeedSequence([seed,1066,sensor]))
        select=rng.choice(pool,size=per_sensor,replace=False)
        ids.extend(sorted(int(i) for i in select))
    idx=np.asarray(ids,dtype=np.int64)
    chosen_xindex=np.asarray(x_ids[idx],dtype=np.int64).copy()
    chosen_sensor=np.asarray(sensors[idx],dtype=np.int64).copy()
    chosen_q=np.asarray(qs[idx],dtype=np.float32).copy()
    chosen_n=np.asarray(normals[idx],dtype=np.float32).copy()
    chosen_x=np.asarray(x[chosen_xindex],dtype=np.float32).copy()
    if not (np.isfinite(chosen_q).all() and np.isfinite(chosen_n).all() and
            np.max(np.abs(np.linalg.norm(chosen_n,axis=1)-1))<5e-3):
        raise ValueError("Invalid frozen boundary normals")
    if not np.isin(chosen_xindex,val).all():
        raise ValueError("Selected a non-validation R1 x")
    digest=hashlib.sha256()
    for arr in (idx,chosen_xindex,chosen_sensor,chosen_q,chosen_n,chosen_x):
        digest.update(np.ascontiguousarray(arr).tobytes())
    return {"ids":idx,"x_index":chosen_xindex,"sensor":chosen_sensor,
            "q":chosen_q,"normal":chosen_n,"x":chosen_x,
            "selection_hash":digest.hexdigest(),"origin":origin,
            "selected_per_sensor":per_sensor,"seed":seed}


def evaluate_sensor(model, x, q, normal, sensor):
    n=len(q)
    if not n:raise ValueError("No same-point sample")
    reports=[]
    ever_active=np.zeros(128,dtype=bool)
    for lo in range(0,n,64):
        hi=min(n,lo+64)
        input_x=torch.as_tensor(x[lo:hi],dtype=torch.float32)
        input_q=torch.as_tensor(q[lo:hi],dtype=torch.float32).requires_grad_(True)
        truth=torch.as_tensor(normal[lo:hi],dtype=torch.float32)
        holder={}
        def activation(_module, _inputs, out):
            holder["relu"]=out.detach()
        hook=model.sensor_heads[sensor][1].register_forward_hook(activation)
        try:
            with torch.enable_grad():
                yy=model.forward_sensor(torch.cat([input_x,input_q],1),sensor).float()
                grad=torch.autograd.grad(yy.sum(),input_q,
                                         create_graph=False,retain_graph=False)[0].float()
        finally:
            hook.remove()
        if "relu" not in holder:
            raise RuntimeError("No sensor-specific ReLU activation observed")
        relu=holder["relu"]
        if relu.shape!=(hi-lo,128):
            raise ValueError("Unexpected sensor first-head ReLU dimensions")
        arr=relu.numpy()>0
        ever_active|=arr.any(axis=0)
        y=yy.detach().numpy().reshape(-1)
        g=grad.detach().numpy()
        gnorm=np.linalg.norm(g,axis=1)
        ncos=np.sum(g*normal[lo:hi],axis=1)/np.maximum(gnorm*np.linalg.norm(normal[lo:hi],axis=1),1e-12)
        if not np.isfinite(gnorm).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite frozen model gradients/scores")
        for j in range(hi-lo):
            reports.append({"score":float(y[j]),
                            "grad_norm":float(gnorm[j]),
                            "normal_cosine":None if gnorm[j]<1e-8 else float(ncos[j]),
                            "relu_frac":float(arr[j].mean()),
                            "head_relu_active_units":int(arr[j].sum()),
                            "predicted_near_zero":bool(abs(y[j])<=.03)})
    return reports,int(ever_active.sum())


def inspect_boundary(repo:Path,cache:Path,r012:Path,aug:Path,*,per_sensor=128,seed=261013):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    selected=choose(cache,r012,per_sensor,seed)
    models, hashes, oldrun,newrun=model_pair(repo,r012,aug)
    digests_pre={}
    for name,model in models.items():
        digests_pre[name]=hashlib.sha256(b"".join(
            v.detach().contiguous().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    rows=[]
    per_sensor_results={}
    for sensor in range(8):
        ix=np.flatnonzero(selected["sensor"]==sensor)
        x,q,n=(selected[z][ix] for z in ("x","q","normal"))
        predictions={}
        ever={}
        for name in ("R1","R1_AUG"):
            predictions[name],ever[name]=evaluate_sensor(models[name],x,q,n,sensor)
        per_sensor_results[f"S{sensor}"]={}
        for name in ("R1","R1_AUG"):
            rp=predictions[name]
            per_sensor_results[f"S{sensor}"][name]={
                "n":len(rp),"zero_abs_rad":numeric_stats([abs(r["score"]) for r in rp]),
                "normal_cosine":numeric_stats([r["normal_cosine"] for r in rp]),
                "q_grad_norm":numeric_stats([r["grad_norm"] for r in rp]),
                "relu_activation_fraction":numeric_stats([r["relu_frac"] for r in rp]),
                "ever_active_first_relu_units":ever[name],
                "near_zero_score_fraction":sum(r["predicted_near_zero"] for r in rp)/len(rp),
                "zero_gradient_count":sum(r["normal_cosine"] is None for r in rp),
            }
        for j,ii in enumerate(ix):
            row={"cache_val_boundary_row":int(selected["ids"][ii]),
                 "x_index":int(selected["x_index"][ii]),"sensor":sensor}
            for name,prefix in (("R1","r1"),("R1_AUG","aug")):
                p=predictions[name][j]
                for key in ("score","normal_cosine","grad_norm","relu_frac",
                            "head_relu_active_units","predicted_near_zero"):
                    row[prefix+"_"+key]=p[key]
            rows.append(row)
    digests_post={}
    for name,model in models.items():
        digests_post[name]=hashlib.sha256(b"".join(
            v.detach().contiguous().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        if digests_pre[name]!=digests_post[name]:
            raise RuntimeError(f"Frozen checkpoint changed in boundary evaluation: {name}")
    return {"summary":{
          "status":"COMPLETE","device":"cpu","models_sha256":hashes,
          "model_parameter_digests_before":digests_pre,"after":digests_post,
          "selection_sha256":selected["selection_hash"],
          "sample_count":len(rows),"per_sensor_requested":per_sensor,
          "selection_seed":seed,"original_R1_validation_split":selected["origin"],
          "per_sensor":per_sensor_results,
          "semantics":"Same sampled R1-validation x, V4-val analytic normals, no new labels. Zero MAE is relative to V4 anchor (not R1 historical distance).",
          "no_optimizer_updates":True,"no_solver_calls":True},
          "rows":rows}
