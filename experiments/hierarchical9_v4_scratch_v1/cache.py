#!/usr/bin/env python3
"""Build/read a mmap training cache from the completed full-scale V4 production run."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,os,shutil,sys,tempfile,time
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PROD=REPO/"experiments/hierarchical9_full_dataset_v4_v1"
FORMAT="care_h9_v4_training_cache_v1"

def _load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
prod=_load("v4train_prod_common",PROD/"prod_common.py")

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    p=Path(path);tmp=p.with_name(p.name+f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n");os.replace(tmp,p)

def _memmap(path,dtype,shape):
    return np.lib.format.open_memmap(path,mode="w+",dtype=dtype,shape=shape)

FILES={
 "global":{"x_index":(np.int32,()),"q":(np.float32,(7,)),"sensor_sign":(np.int8,(8,)),"union_sign":(np.int8,())},
 "v3":{"x_index":(np.int32,()),"q":(np.float32,(7,)),"sensor_sign":(np.int8,(8,)),"union_sign":(np.int8,()),
       "sensor_value":(np.float32,(8,)),"sensor_value_mask":(np.bool_,(8,)),
       "union_value":(np.float32,()),"union_value_mask":(np.bool_,())},
 "v4":{"x_index":(np.int32,()),"q":(np.float32,(7,)),"sensor":(np.uint8,()),"value":(np.float32,())},
 "boundary":{"x_index":(np.int32,()),"q":(np.float32,(7,)),"sensor":(np.uint8,()),"grad":(np.float32,(7,))},
}

def _allocate(root,split,kind,n):
    d=Path(root)/split;d.mkdir(parents=True,exist_ok=True);out={}
    for key,(dtype,tail) in FILES[kind].items():
        out[key]=_memmap(d/f"{kind}_{key}.npy",dtype,(int(n),)+tail)
    return out

def _put(dst,offset,data):
    n=len(next(iter(data.values())))
    for k,v in data.items():dst[k][offset:offset+n]=v
    return offset+n

def _sign(g):return np.where(np.asarray(g)>=0,1,-1).astype(np.int8)

def group_v3(global_arrays,v3_arrays):
    """Return one row per selected (x,q_slot), with 8-head value masks."""
    gx=np.asarray(global_arrays["x_index"]);qpool=np.asarray(global_arrays["q_pool"]);gpool=np.asarray(global_arrays["g_pool"])
    selected=np.asarray(global_arrays["v3_selected_index"]);spl=np.asarray(global_arrays["split"])
    pos={int(x):i for i,x in enumerate(gx)}
    n=len(gx);k=selected.shape[1]
    q=qpool[np.arange(n)[:,None],selected]
    gg=gpool[np.arange(n)[:,None],selected]
    value=np.zeros((n,k,8),np.float32);mask=np.zeros((n,k,8),bool)
    vx=np.asarray(v3_arrays["x_index"]);vs=np.asarray(v3_arrays["sensor"]);slot=np.asarray(v3_arrays["q_slot"])
    valid=np.asarray(v3_arrays["value_valid"]);nv=np.asarray(v3_arrays["new_value"],np.float32)
    for r in range(len(vx)):
        if not valid[r]:continue
        i=pos[int(vx[r])];j=int(slot[r]);s=int(vs[r])
        if mask[i,j,s]:raise ValueError("duplicate V3 task")
        value[i,j,s]=nv[r];mask[i,j,s]=True
    um=mask.all(axis=2);uv=np.zeros((n,k),np.float32)
    if um.any():uv[um]=value[um].max(axis=1)
    return dict(
      x_index=np.repeat(gx[:,None],k,axis=1).reshape(-1).astype(np.int32),
      split=np.repeat(spl[:,None],k,axis=1).reshape(-1).astype(np.uint8),
      q=q.reshape(-1,7).astype(np.float32),
      sensor_sign=_sign(gg).reshape(-1,8),
      union_sign=np.where((gg>=0).any(axis=2),1,-1).astype(np.int8).reshape(-1),
      sensor_value=value.reshape(-1,8),sensor_value_mask=mask.reshape(-1,8),
      union_value=uv.reshape(-1),union_value_mask=um.reshape(-1))

def build(source,out):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    for name in ("global_pool_summary.json","v3_summary.json","v4_summary.json","manifest.json"):
        if not (source/name).is_file():raise FileNotFoundError(source/name)
    summaries={n:json.loads((source/n).read_text()) for n in ("global_pool_summary.json","v3_summary.json","v4_summary.json")}
    if any(v.get("status")!="COMPLETE" for v in summaries.values()):raise ValueError("production stages incomplete")
    run,manifest,bank,plan=prod.open_run(source)
    cfg=prod.Config(**manifest["config"]);nsh=prod.shard_count(len(plan["x_index"]),cfg.x_shard_size)
    for stage in ("global_pool","v3_labels","v4_tubes"):prod.verify_stage(source,stage,nsh)
    counts={sp:{k:0 for k in FILES} for sp in ("train","val")}
    for sp,idv in (("train",0),("val",1)):
        nx=int((plan["split"]==idv).sum())
        counts[sp]["global"]=nx*cfg.candidate_q_per_x
        counts[sp]["v3"]=nx*cfg.v3_q_per_x
    for sh in range(nsh):
        p,_=prod.stage_paths(source,"v4_tubes",sh)
        with np.load(p,allow_pickle=False) as z:
            for sp,idv in (("train",0),("val",1)):
                m=z["split"]==idv
                counts[sp]["v4"]+=int(m.sum())
                counts[sp]["boundary"]+=int((m&(z["value"]==0)).sum())
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix="."+out.name+".building.",dir=out.parent))
    try:
        np.save(tmp/"x.npy",np.asarray(bank.x,np.float32),allow_pickle=False)
        dst={sp:{k:_allocate(tmp,sp,k,counts[sp][k]) for k in FILES} for sp in counts}
        off={sp:{k:0 for k in FILES} for sp in counts}
        t0=time.perf_counter()
        for sh in range(nsh):
            gp,_=prod.stage_paths(source,"global_pool",sh);vp,_=prod.stage_paths(source,"v3_labels",sh);tp,_=prod.stage_paths(source,"v4_tubes",sh)
            with np.load(gp,allow_pickle=False) as z:g={k:z[k] for k in z.files}
            with np.load(vp,allow_pickle=False) as z:v={k:z[k] for k in z.files}
            grouped=group_v3(g,v)
            # Global rows.
            Q=cfg.candidate_q_per_x
            flat=dict(x_index=np.repeat(g["x_index"],Q).astype(np.int32),
                      split=np.repeat(g["split"],Q).astype(np.uint8),
                      q=g["q_pool"].reshape(-1,7).astype(np.float32),
                      sensor_sign=_sign(g["g_pool"]).reshape(-1,8),
                      union_sign=np.where((g["g_pool"]>=0).any(axis=2),1,-1).astype(np.int8).reshape(-1))
            for sp,idv in (("train",0),("val",1)):
                m=flat["split"]==idv
                data={k:flat[k][m] for k in FILES["global"]}
                off[sp]["global"]=_put(dst[sp]["global"],off[sp]["global"],data)
                m=grouped["split"]==idv
                data={k:grouped[k][m] for k in FILES["v3"]}
                off[sp]["v3"]=_put(dst[sp]["v3"],off[sp]["v3"],data)
            with np.load(tp,allow_pickle=False) as z:
                for sp,idv in (("train",0),("val",1)):
                    m=z["split"]==idv
                    data=dict(x_index=z["x_index"][m].astype(np.int32),q=z["q"][m].astype(np.float32),
                              sensor=z["sensor"][m].astype(np.uint8),value=z["value"][m].astype(np.float32))
                    off[sp]["v4"]=_put(dst[sp]["v4"],off[sp]["v4"],data)
                    b=m&(z["value"]==0)
                    data=dict(x_index=z["x_index"][b].astype(np.int32),q=z["q"][b].astype(np.float32),
                              sensor=z["sensor"][b].astype(np.uint8),grad=z["grad"][b].astype(np.float32))
                    off[sp]["boundary"]=_put(dst[sp]["boundary"],off[sp]["boundary"],data)
            if (sh+1)%10==0 or sh+1==nsh:print(f"[cache] shard={sh+1}/{nsh} elapsed={time.perf_counter()-t0:.1f}s",flush=True)
        for sp in off:
            for k in off[sp]:
                if off[sp][k]!=counts[sp][k]:raise RuntimeError(f"count mismatch {sp}/{k}: {off[sp][k]} != {counts[sp][k]}")
        # Flush before hashing.
        for sp in dst:
            for kind in dst[sp]:
                for a in dst[sp][kind].values():a.flush()
        del dst
        files={}
        for p in sorted(tmp.rglob("*.npy")):
            rel=str(p.relative_to(tmp));files[rel]=dict(bytes=p.stat().st_size,sha256=sha256_file(p))
        cache_manifest=dict(format=FORMAT,status="COMPLETE",source=str(source),
          source_manifest_sha256=sha256_file(source/"manifest.json"),
          source_summaries_sha256={n:sha256_file(source/n) for n in summaries},
          counts=counts,files=files,
          semantics=dict(global="all-8 analytic signs + exact union sign",
            v3="grouped selected arbitrary q; per-sensor continuous values only where V3 value_valid; union value only when all 8 values valid",
            v4="sensor-specific local signed normal-offset values",
            boundary="V4 zero-offset rows only; analytic positive-side normal"),
          training_ready=True)
        write_json(tmp/"manifest.json",cache_manifest)
        out.parent.mkdir(parents=True,exist_ok=True);os.rename(tmp,out)
    finally:
        if tmp.exists():shutil.rmtree(tmp,ignore_errors=True)
    print(json.dumps(cache_manifest["counts"],indent=2),flush=True);print("[complete]",out,flush=True)

class TrainingCache:
    def __init__(self,root,verify_hashes=False):
        self.root=Path(root).resolve();self.manifest=json.loads((self.root/"manifest.json").read_text())
        if self.manifest.get("format")!=FORMAT or not self.manifest.get("training_ready") or self.manifest.get("status")!="COMPLETE":
            raise ValueError("not a complete V4 training cache")
        if verify_hashes:
            for rel,m in self.manifest["files"].items():
                p=self.root/rel
                if p.stat().st_size!=m["bytes"] or sha256_file(p)!=m["sha256"]:raise ValueError(f"corrupt cache file {rel}")
        self.x=np.load(self.root/"x.npy",mmap_mode="r",allow_pickle=False)
        self.a={}
        for sp in ("train","val"):
            self.a[sp]={}
            for kind,fields in FILES.items():
                self.a[sp][kind]={k:np.load(self.root/sp/f"{kind}_{k}.npy",mmap_mode="r",allow_pickle=False) for k in fields}
    @property
    def identity(self):return sha256_file(self.root/"manifest.json")
    def size(self,split,kind):return len(next(iter(self.a[split][kind].values())))
    def indices(self,split,kind,n,seed,step,rank):
        total=self.size(split,kind)
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(step),int(rank),{"global":11,"v3":23,"v4":37,"boundary":41}[kind],0 if split=="train" else 999]))
        return rng.integers(0,total,size=int(n),endpoint=False,dtype=np.int64)
    def batch(self,split,kind,ids,device):
        a=self.a[split][kind];ids=np.asarray(ids,np.int64)
        out={k:torch.as_tensor(np.asarray(v[ids]).copy(),device=device) for k,v in a.items()}
        xidx=np.asarray(a["x_index"][ids],np.int64)
        out["inputs"]=torch.cat((torch.as_tensor(np.asarray(self.x[xidx]).copy(),device=device,dtype=torch.float32),out["q"].float()),dim=1)
        return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    build(a.source,a.out)
if __name__=="__main__":main()
