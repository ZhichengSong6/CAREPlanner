#!/usr/bin/env python3
"""Value/gradient/timing qualification for V1 and R1 runtime sensor adapters."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, statistics, sys, time
from pathlib import Path

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
VIS=REPO/"src/care_visibility_cdf/scripts"
if str(VIS) not in sys.path: sys.path.insert(0,str(VIS))
from evaluate_direct_vs_projection_ascent import torch_load_checkpoint
from per_sensor_visibility_runtime import build_per_sensor_model

V1_SHA="979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199"
R1_SHA="4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002"
OLD8_SHA="43f962729adcd17aa114edb9fc410facbbb97ebe7343f0ad3309fe50d273acdb"

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise ImportError(path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def ref_model(kind,ckpt,device):
    if kind=="V1":
        m=load_module("runtime_qual_v1_training_model",REPO/"experiments/hierarchical9_scratch_v1/model.py")
        model=m.HierarchicalVisibilityCDF()
    elif kind=="R1":
        m=load_module("runtime_qual_r1_training_model",REPO/"experiments/hierarchical9_scratch50k_r012_v1/model.py")
        model=m.build_model("R1")
    else: raise ValueError(kind)
    model.load_state_dict(ckpt["model_state"],strict=True)
    return model.to(device=device,dtype=torch.float32).eval().requires_grad_(False)

def sensor_min_value_grad(model,points,q,sensor,full9):
    qv=q.detach().clone().reshape(1,7).requires_grad_(True)
    qb=qv.expand(points.shape[0],-1)
    pred=model(torch.cat([points,qb],dim=-1))
    y=pred[:,1+sensor] if full9 else pred[:,sensor]
    value=torch.min(y)
    grad=torch.autograd.grad(value,qv,create_graph=False)[0]
    return float(value.detach()),grad.detach()

def latency(adapter,points,q,device,repeats):
    for _ in range(10):
        sensor_min_value_grad(adapter,points,q,7,False)
    if device.type=="cuda": torch.cuda.synchronize(device)
    xs=[]
    for i in range(repeats):
        s=i%8
        if device.type=="cuda": torch.cuda.synchronize(device)
        t=time.perf_counter()
        sensor_min_value_grad(adapter,points,q,s,False)
        if device.type=="cuda": torch.cuda.synchronize(device)
        xs.append(1000*(time.perf_counter()-t))
    xs=sorted(xs)
    def pct(p):
        z=p*(len(xs)-1); i=int(z); j=min(i+1,len(xs)-1); w=z-i
        return xs[i]*(1-w)+xs[j]*w
    return {"count":len(xs),"median_ms":statistics.median(xs),"mean_ms":statistics.fmean(xs),"p95_ms":pct(.95),"max_ms":max(xs)}

def check_one(kind,path,expected_sha,device,repeats):
    p=Path(path).expanduser().resolve()
    if not p.is_file(): raise FileNotFoundError(p)
    digest=sha256(p)
    if digest!=expected_sha: raise RuntimeError(f"{kind} SHA mismatch {digest} != {expected_sha}")
    ckpt=torch_load_checkpoint(str(p),device)
    adapter,loaded=build_per_sensor_model(str(p),device)
    direct=ref_model(kind,ckpt,device)
    if loaded.get("model_state").keys()!=ckpt.get("model_state").keys(): raise RuntimeError("checkpoint object mismatch")
    gen=torch.Generator(device="cpu"); gen.manual_seed(260921 if kind=="V1" else 260922)
    x=(torch.rand((9,3),generator=gen)*torch.tensor([.6,.6,.6])+torch.tensor([-.3,-.3,.1])).to(device)
    q=(torch.rand((1,7),generator=gen)*2-1).to(device)
    q=q*torch.tensor([[2.0,1.5,2.0,1.8,2.0,2.0,.8]],device=device)
    with torch.no_grad():
        inp=torch.cat([x,q.expand(len(x),-1)],dim=-1)
        full=direct(inp)
        got=adapter(inp)
        if full.shape!=(9,9) or got.shape!=(9,8): raise RuntimeError((full.shape,got.shape))
        value_err=float(torch.max(torch.abs(full[:,1:]-got)).item())
    grad_err=0.0; rows={}
    for s in (0,4,7):
        vr,gr=sensor_min_value_grad(direct,x,q,s,True)
        va,ga=sensor_min_value_grad(adapter,x,q,s,False)
        grad_err=max(grad_err,float(torch.max(torch.abs(gr-ga)).item()))
        rows[f"S{s}"]={"ref_value":vr,"adapter_value":va,"value_abs_error":abs(vr-va),
                         "grad_max_abs_error":float(torch.max(torch.abs(gr-ga)).item()),
                         "grad_norm":float(torch.linalg.vector_norm(ga).item())}
    if value_err>2e-6 or grad_err>3e-6:
        raise RuntimeError(f"{kind} adapter equivalence failed value={value_err} grad={grad_err}")
    # Confirm a q change recomputes the field rather than reusing cached features.
    with torch.no_grad():
        q2=q.clone(); q2[0,0]+=0.07
        a=adapter(torch.cat([x,q.expand(len(x),-1)],dim=-1))
        b=adapter(torch.cat([x,q2.expand(len(x),-1)],dim=-1))
        recompute_delta=float(torch.max(torch.abs(a-b)).item())
    if recompute_delta<=1e-8: raise RuntimeError(f"{kind} q-change produced no output change")
    return {"checkpoint":str(p),"sha256":digest,"format":ckpt.get("format"),"arm":ckpt.get("arm"),
            "value_max_abs_error":value_err,"gradient_max_abs_error":grad_err,
            "q_recompute_max_delta":recompute_delta,"sensors":rows,
            "latency":latency(adapter,x,q,device,repeats)}

def check_old8(path,device):
    p=Path(path).expanduser().resolve()
    if not p.is_file():
        return {"status":"NOT_PRESENT","checkpoint":str(p)}
    digest=sha256(p)
    if digest!=OLD8_SHA: raise RuntimeError(f"old8 SHA mismatch {digest} != {OLD8_SHA}")
    model,ckpt=build_per_sensor_model(str(p),device)
    x=torch.tensor([[.1,.05,.15]],device=device,dtype=torch.float32)
    q=torch.zeros((1,7),device=device,dtype=torch.float32,requires_grad=True)
    pred=model(torch.cat([x,q],dim=-1))
    if pred.shape!=(1,8): raise RuntimeError(f"old8 shape {tuple(pred.shape)}")
    g=torch.autograd.grad(pred[0,0],q)[0]
    if not torch.isfinite(g).all(): raise RuntimeError("old8 q gradient non-finite")
    return {"status":"PASS","checkpoint":str(p),"sha256":digest,
            "format":ckpt.get("format"),"shape":list(pred.shape),
            "grad_norm":float(torch.linalg.vector_norm(g).item())}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v1-checkpoint",default=str(REPO/"src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt"))
    ap.add_argument("--r1-checkpoint",required=True)
    ap.add_argument("--old8-checkpoint",default=str(REPO/"src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt"))
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--repeats",type=int,default=80)
    ap.add_argument("--output",default="")
    a=ap.parse_args()
    if a.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    device=torch.device(a.device)
    if device.type=="cuda": torch.cuda.set_device(0)
    report={"device":str(device),"torch":torch.__version__,
            "V1":check_one("V1",a.v1_checkpoint,V1_SHA,device,a.repeats),
            "R1":check_one("R1",a.r1_checkpoint,R1_SHA,device,a.repeats),
            "old8":check_old8(a.old8_checkpoint,device),
            "status":"PASS"}
    out=Path(a.output).resolve() if a.output else REPO/"outputs/phase_e_r1_runtime_qualification/runtime_adapter_test.json"
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2)); print("[done] runtime_adapter_test PASS",out)

if __name__=="__main__": main()
