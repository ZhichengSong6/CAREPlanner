"""Shared plumbing for full-scale V4 dataset production."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,json,math,os,sys,tempfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V1=ROOT/"experiments/hierarchical9_offline_relabel_v1"
V3=ROOT/"experiments/hierarchical9_offline_relabel_v3"
FAST=ROOT/"experiments/hierarchical9_offline_relabel_v3_fast_v1"
for p in (V1,V3,FAST):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import cache_io
import common as v3
from analytic_geometry import AnalyticGeometry

core=v3.core
RepoOracle=v3.RepoOracle
FORMAT="careplanner_full_v4_dataset_v1"
TUBE_RADII=(0.005,0.01,0.02)

@dataclass(frozen=True)
class Config:
    candidate_q_per_x:int=128
    v3_q_per_x:int=4
    anchors_per_sensor:int=32
    x_shard_size:int=128
    seed:int=260927
    def validate(self):
        if self.candidate_q_per_x<16 or not 1<=self.v3_q_per_x<=self.candidate_q_per_x:
            raise ValueError("bad q budgets")
        if self.anchors_per_sensor<1 or self.x_shard_size<1 or self.seed<0:
            raise ValueError("bad production config")

def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,x):cache_io.write_json(Path(p),x)
def write_npz(p,x):cache_io.write_npz(Path(p),x)
def sha256_file(p):return v3.sha256_file(Path(p))

def fresh_x_indices(path):
    ids=set()
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                row=json.loads(line)
                if "x_index" not in row:raise ValueError("fresh start lacks x_index")
                ids.add(int(row["x_index"]))
    return np.asarray(sorted(ids),np.int64)

def task_seed(seed,xi):
    return np.random.SeedSequence([int(seed),int(xi)])

def code_identity():
    files=list(HERE.glob("*.py"))
    deps=[FAST/"analytic_geometry.py",V3/"common.py",V3.parent/"hierarchical9_offline_relabel_v2"/"verified_core.py",
          V1/"cache_io.py",V1/"repo_oracle.py",V1/"label_core.py"]
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(files+deps) if p.is_file()}

def open_run(out,check_code=True):
    out=Path(out).resolve();m=read_json(out/"manifest.json")
    if m.get("format")!=FORMAT:raise ValueError("wrong full-v4 format")
    if sha256_file(out/"x_plan.npz")!=m["x_plan_sha256"]:raise ValueError("x_plan changed")
    if check_code and m["code_identity"]!=code_identity():raise ValueError("production code changed; use a new OUT")
    source,sm,bank,_=cache_io.open_job(m["source"])
    with np.load(out/"x_plan.npz",allow_pickle=False) as z:plan={k:z[k] for k in z.files}
    return out,m,bank,plan

def shard_bounds(n,shard,size):
    a=shard*size;b=min(n,a+size)
    if a>=n:raise IndexError(shard)
    return a,b

def shard_count(n,size):return (n+size-1)//size

def stage_paths(out,stage,shard):
    root=Path(out)/stage;return root/f"shard_{shard:04d}.npz",root/f"shard_{shard:04d}.json"

def install_shard(out,stage,shard,arrays,meta):
    p,j=stage_paths(out,stage,shard);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() or j.exists():
        if not(p.exists() and j.exists()):raise RuntimeError(f"partial existing shard {stage}/{shard}")
        old=read_json(j)
        if sha256_file(p)!=old["sha256"]:raise RuntimeError(f"corrupt existing shard {p}")
        return False
    write_npz(p,arrays);meta=dict(meta,sha256=sha256_file(p),file=p.name)
    write_json(j,meta);return True

def verify_stage(out,stage,nshards):
    entries=[]
    for sh in range(nshards):
        p,j=stage_paths(out,stage,sh)
        if not p.is_file() or not j.is_file():raise FileNotFoundError(f"missing {stage} shard {sh}")
        m=read_json(j)
        if sha256_file(p)!=m["sha256"]:raise RuntimeError(f"bad checksum {p}")
        entries.append(m)
    return entries

def select_v3_indices(q,g,support,k,lo,hi):
    """Greedy cover available (sensor, sign) states; tie-break by q diversity and |FOV margin|."""
    q=np.asarray(q,float);g=np.asarray(g,float);support=np.asarray(support,bool)
    n=len(q);state=(g>=0).astype(np.int8)
    available=np.zeros((8,2),bool)
    for s in np.flatnonzero(support):
        available[s,0]=np.any(state[:,s]==0);available[s,1]=np.any(state[:,s]==1)
    uncovered=available.copy();selected=[]
    span=np.maximum(np.asarray(hi)-np.asarray(lo),1e-12)
    qn=(q-np.asarray(lo))/span
    margin=np.mean(np.abs(g[:,support]),axis=1) if support.any() else np.zeros(n)
    for _ in range(k):
        best=None;bestkey=None
        for i in range(n):
            if i in selected:continue
            gain=sum(bool(uncovered[s,state[i,s]]) for s in np.flatnonzero(support))
            diversity=min((float(np.linalg.norm(qn[i]-qn[j])) for j in selected),default=0.0)
            key=(gain,diversity,float(margin[i]),-i)
            if bestkey is None or key>bestkey:bestkey,best=key,i
        selected.append(int(best))
        for s in np.flatnonzero(support):uncovered[s,state[best,s]]=False
    covered=available & ~uncovered
    return np.asarray(selected,np.int16),available,covered

def select_anchor_slots(bank,xi,s,k):
    slots=np.flatnonzero(np.asarray(bank.valid[int(xi),:,int(s)],bool))
    if len(slots)<=k:return slots.astype(np.int32)
    pts=np.asarray(bank.q[int(xi),slots,:,int(s)],float)
    active=np.flatnonzero(bank.masks[int(s)])
    z=pts[:,active]
    chosen=[0];d2=np.sum((z-z[0])**2,axis=1)
    for _ in range(1,k):
        j=int(np.argmax(d2));chosen.append(j)
        d2=np.minimum(d2,np.sum((z-z[j])**2,axis=1))
    return slots[np.asarray(chosen,int)].astype(np.int32)

def refine_anchor(q0,mask,lo,hi,geometry,tol=1e-6,maxiter=16,max_step=.10,backtracks=12):
    q=np.asarray(q0,np.float64).copy();mask=np.asarray(mask,np.float64);calls=0
    if np.any(q<lo) or np.any(q>hi) or not np.isfinite(q).all():
        return dict(success=False,reason="BAD_Q0",q_star=q,g=np.nan,iterations=0,calls=0)
    reason="MAXITER"
    for it in range(maxiter+1):
        h,j=geometry(q);calls+=1;face=int(np.argmin(h));g=float(h[face])
        if not np.isfinite(g) or not np.isfinite(j).all():reason="NONFINITE";break
        if abs(g)<=tol:return dict(success=True,reason="BOUNDARY",q_star=q.copy(),g=g,iterations=it,calls=calls)
        if it==maxiter:break
        n=np.asarray(j[face])*mask;nn=float(n@n)
        if not np.isfinite(nn) or nn<1e-14:reason="DEGENERATE_NORMAL";break
        step=g*n/nn;sn=float(np.linalg.norm(step))
        if sn>max_step:step*=max_step/sn
        accepted=False
        for b in range(backtracks):
            cand=q-(.5**b)*step
            if np.any(cand<lo) or np.any(cand>hi):continue
            hc,_=geometry(cand);calls+=1;gc=float(np.min(hc))
            if np.isfinite(gc) and (abs(gc)<abs(g) or abs(gc)<=tol):
                q=cand;accepted=True;break
        if not accepted:reason="LINE_SEARCH_STALLED";break
    h,_=geometry(q);calls+=1
    return dict(success=False,reason=reason,q_star=q.copy(),g=float(np.min(h)),iterations=it,calls=calls)
