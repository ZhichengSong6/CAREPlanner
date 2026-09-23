"""Shared I/O, loss, and evaluation for the R1 paired-label micro-training study."""
from __future__ import annotations
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
R012=REPO/"experiments/hierarchical9_scratch50k_r012_v1"
FORMAT="care_h9_paired_training_v1"
R1_SHA="4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002"
R1_FORMAT="care_h9_scratch50k_r012_v1"
ARMS=("old_value","new_value","old_value_grad","new_value_grad")

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def write_json(path:Path,obj):
    path=Path(path); tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

rmodel=_load("paired_r012_model",R012/"model.py")

def load_r1(r012_root:Path,device,trainable=True):
    root=Path(r012_root).resolve(); final=root/"formal/R1/final.pt"; runp=root/"formal/R1/run.json"
    if not final.is_file() or not runp.is_file(): raise FileNotFoundError(root/"formal/R1")
    digest=sha256(final)
    if digest!=R1_SHA: raise ValueError(f"Wrong R1 SHA256: {digest}")
    run=json.loads(runp.read_text())
    for k,v in {"status":"COMPLETE","arm":"R1","successful_updates":50000,"final_sha256":digest}.items():
        if run.get(k)!=v: raise ValueError(f"R1 run mismatch {k}: {run.get(k)!r}")
    cp=torch.load(final,map_location="cpu",weights_only=False)
    for k,v in {"format":R1_FORMAT,"arm":"R1","completed":True,"step":50000,"initialization":"random_from_scratch","frozen_parameters":0}.items():
        if cp.get(k)!=v: raise ValueError(f"R1 checkpoint mismatch {k}: {cp.get(k)!r}")
    model=rmodel.build_model("R1"); model.load_state_dict(cp["model_state"],strict=True)
    return model.to(device=device,dtype=torch.float32).train(trainable).requires_grad_(trainable),cp,digest

class PairedCache:
    """Small immutable v3 paired cache loaded fully into RAM (320 rows in the pilot)."""
    def __init__(self,root:Path):
        self.root=Path(root).resolve()
        idx=json.loads((self.root/"dataset_index.json").read_text())
        if idx.get("format")!="careplanner_paired_offline_labels_v3" or not idx.get("complete") or not idx.get("audit_complete"):
            raise ValueError("Need complete audited v3 paired_cache")
        if idx.get("gradient_policy")!="FD_PASS_ONLY": raise ValueError("Unexpected gradient policy")
        if sha256(self.root/"summary.json")!=idx["summary_sha256"]: raise ValueError("Corrupt paired summary")
        rows=[]
        for e in idx["shards"]:
            p=(self.root/e["path"]).resolve()
            if not p.is_relative_to(self.root) or sha256(p)!=e["sha256"]: raise ValueError(f"Corrupt shard {p}")
            with np.load(p,allow_pickle=False) as z: a={k:z[k] for k in z.files}
            if np.any(a["paired_grad_mask"] & (a["gradient_status"]!="PASS")):
                raise ValueError("Unverified gradient exposed by paired mask")
            rows.append(a)
        self.a={k:np.concatenate([r[k] for r in rows],axis=0) for k in rows[0]}
        if len(self.a["query_id"])!=idx["query_count"]: raise ValueError("Query count mismatch")
        self.identity=sha256(self.root/"dataset_index.json")
        self.train=np.flatnonzero(self.a["split"]==0); self.val=np.flatnonzero(self.a["split"]==1)

    def batch(self,ids,device,label_set):
        if label_set not in ("old","new"): raise ValueError(label_set)
        a=self.a; ids=np.asarray(ids,dtype=np.int64); m=label_set
        return {
            "inputs":torch.as_tensor(np.concatenate((a["x"][ids],a["q_query"][ids]),1),device=device,dtype=torch.float32),
            "sensor_value":torch.as_tensor(a[m+"_value"][ids],device=device,dtype=torch.float32),
            "sensor_grad":torch.as_tensor(a[m+"_grad"][ids],device=device,dtype=torch.float32),
            "sensor_value_mask":torch.as_tensor(a["paired_value_mask"][ids],device=device,dtype=torch.bool),
            "sensor_grad_mask":torch.as_tensor(a["paired_grad_mask"][ids],device=device,dtype=torch.bool),
            "union_value":torch.as_tensor(a["union_"+m+"_value"][ids],device=device,dtype=torch.float32),
            "union_grad":torch.as_tensor(a["union_"+m+"_grad"][ids],device=device,dtype=torch.float32),
            "union_value_mask":torch.as_tensor(a["paired_union_value_mask"][ids],device=device,dtype=torch.bool),
            "union_grad_mask":torch.as_tensor(a["paired_union_grad_mask"][ids],device=device,dtype=torch.bool),
            "reference_g":torch.as_tensor(a["reference_g_m"][ids],device=device,dtype=torch.float32),
            "support":torch.as_tensor(a["support"][ids],device=device,dtype=torch.bool),
        }

def arm_spec(arm):
    if arm not in ARMS: raise ValueError(arm)
    return ("old" if arm.startswith("old") else "new", arm.endswith("_grad"))

def stream_indices(n,seed,step,batch_size):
    rng=np.random.default_rng(np.random.SeedSequence([seed,step,481516]))
    p=rng.permutation(n)
    return p if batch_size>=n else p[:batch_size]

def _head_grad(y,q,training):
    return torch.autograd.grad(y.sum(),q,create_graph=training,retain_graph=True)[0].float()

def paired_loss(model,b,label_set,use_grad,training=True,value_weight=5.0,grad_weight=0.1):
    q=b["inputs"][:,3:].detach().float().clone().requires_grad_(True)
    x=b["inputs"][:,:3].detach().float(); pred=model(torch.cat((x,q),1)).float()
    total=(pred*0).sum(); parts={"value_sum":0.0,"value_count":0,"grad_sum":0.0,"grad_count":0}
    sensor_losses=[]; sensor_grad_losses=[]
    for s in range(8):
        vm=b["sensor_value_mask"][:,s]
        if vm.any():
            err=pred[vm,s+1]-b["sensor_value"][vm,s]; sensor_losses.append(err.square().mean())
            parts["value_sum"]+=float(err.detach().abs().sum());parts["value_count"]+=int(vm.sum())
        if use_grad:
            gm=b["sensor_grad_mask"][:,s]
            if gm.any():
                g=_head_grad(pred[:,s+1],q,training)
                cos=F.cosine_similarity(g[gm],b["sensor_grad"][gm,s],dim=-1,eps=1e-6)
                sensor_grad_losses.append((1-cos).mean())
                parts["grad_sum"]+=float((1-cos).detach().sum());parts["grad_count"]+=int(gm.sum())
    if sensor_losses: total=total+value_weight*torch.stack(sensor_losses).mean()
    uvm=b["union_value_mask"]
    if uvm.any():
        ue=pred[uvm,0]-b["union_value"][uvm]; total=total+value_weight*ue.square().mean()
        parts["value_sum"]+=float(ue.detach().abs().sum());parts["value_count"]+=int(uvm.sum())
    if use_grad:
        if sensor_grad_losses: total=total+grad_weight*torch.stack(sensor_grad_losses).mean()
        ugm=b["union_grad_mask"]
        if ugm.any():
            ug=_head_grad(pred[:,0],q,training)
            ucos=F.cosine_similarity(ug[ugm],b["union_grad"][ugm],dim=-1,eps=1e-6)
            total=total+grad_weight*(1-ucos).mean()
            parts["grad_sum"]+=float((1-ucos).detach().sum());parts["grad_count"]+=int(ugm.sum())
    if parts["value_count"]==0: raise RuntimeError("No paired value supervision")
    return total,parts

def evaluate(model,cache,ids,device,parent=None):
    ids=np.asarray(ids,dtype=np.int64)
    inputs=torch.as_tensor(np.concatenate((cache.a["x"][ids],cache.a["q_query"][ids]),1),device=device,dtype=torch.float32)
    q=inputs[:,3:].detach().clone().requires_grad_(True); x=inputs[:,:3].detach(); model.eval()
    with torch.enable_grad(),torch.autocast(device_type=device.type,enabled=False):
        pred=model(torch.cat((x,q),1)).float()
        grads=[_head_grad(pred[:,h],q,False).detach() for h in range(9)]
    result={"rows":len(ids),"targets":{}}
    for label in ("old","new"):
        b=cache.batch(ids,device,label); sensor=[]
        for s in range(8):
            vm=b["sensor_value_mask"][:,s];gm=b["sensor_grad_mask"][:,s];row={"value_n":int(vm.sum()),"grad_n":int(gm.sum())}
            if vm.any():
                e=pred[vm,s+1]-b["sensor_value"][vm,s]
                row.update(mae=float(e.abs().mean()),rmse=float(e.square().mean().sqrt()),
                           sign_accuracy=float(((pred[vm,s+1]>=0)==(b["sensor_value"][vm,s]>=0)).float().mean()))
            if gm.any(): row["grad_cosine"]=float(F.cosine_similarity(grads[s+1][gm],b["sensor_grad"][gm,s],dim=-1,eps=1e-6).mean())
            sensor.append(row)
        uv=b["union_value_mask"];ug=b["union_grad_mask"];union={"value_n":int(uv.sum()),"grad_n":int(ug.sum())}
        if uv.any():
            e=pred[uv,0]-b["union_value"][uv];union.update(mae=float(e.abs().mean()),rmse=float(e.square().mean().sqrt()),
                sign_accuracy=float(((pred[uv,0]>=0)==(b["union_value"][uv]>=0)).float().mean()))
        if ug.any(): union["grad_cosine"]=float(F.cosine_similarity(grads[0][ug],b["union_grad"][ug],dim=-1,eps=1e-6).mean())
        vals=[r["mae"] for r in sensor if "mae" in r]
        result["targets"][label]={"sensor":sensor,"sensor_mae_mean":float(np.mean(vals)) if vals else None,"union":union}
    support=torch.as_tensor(cache.a["support"][ids],device=device,dtype=torch.bool)
    ref=torch.as_tensor(cache.a["reference_g_m"][ids],device=device,dtype=torch.float32);ok=support&torch.isfinite(ref)
    result["analytic_sensor_sign_accuracy"]=float((((pred[:,1:]>=0)==(ref>=0))&ok).sum()/ok.sum()) if ok.any() else None
    if parent is not None:
        with torch.no_grad(): p0=parent(inputs).float()
        result["parent_value_drift_mae"]=float((pred.detach()-p0).abs().mean())
    return result
