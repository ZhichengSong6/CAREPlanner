"""Accelerated V3 base solver with face-aware bank seeds and fail-closed fallback.

Scientific label definition is unchanged: best-found per-sensor continuous FOV-boundary
projection, screened by the existing V2/V3 gates. Speed comes from avoiding obviously
mismatched bank-seed/face combinations. Hard or uncertain cases fall back to the
original seed set. This remains an approximate/local nearest-boundary label, never a
global certificate.
"""
from __future__ import annotations
from collections import OrderedDict
import math,time
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
import torch

HERE=Path(__file__).resolve().parent
V3=HERE.parent/"hierarchical9_offline_relabel_v3"
V2=HERE.parent/"hierarchical9_offline_relabel_v2"
V1=HERE.parent/"hierarchical9_offline_relabel_v1"
for p in (V3,V2,V1):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import common as v3common
core=v3common.core
from repo_oracle import FOV


class ExactMemo:
    """Task-local exact-value memo; unlike legacy last-value memo, reuses repeated seeds across faces."""
    def __init__(self,geometry,base,active,cap=8192):
        self.geometry=geometry;self.base=np.asarray(base,float).copy();self.active=np.asarray(active,int)
        self.cap=int(cap);self.cache=OrderedDict();self.calls=0;self.hits=0
    def full_q(self,z):
        q=self.base.copy();q[self.active]=z;return q
    def evaluate(self,z):
        z=np.asarray(z,np.float64);key=z.tobytes()
        if key in self.cache:
            self.hits+=1;h,j=self.cache.pop(key);self.cache[key]=(h,j);return h,j
        h,j=self.geometry(self.full_q(z));h=np.asarray(h,np.float64);j=np.asarray(j,np.float64)[:,self.active]
        if h.ndim!=1 or j.shape!=(len(h),len(self.active)) or not np.isfinite(h).all() or not np.isfinite(j).all():
            raise FloatingPointError("bad geometry")
        self.calls+=1;self.cache[key]=(h,j)
        if len(self.cache)>self.cap:self.cache.popitem(last=False)
        return h,j


class FaceBankCache:
    """Cheap batched FP32 active-face classification for the existing q0 bank."""
    def __init__(self,oracle,max_pairs=64):
        self.oracle=oracle;self.cap=max_pairs;self.cache=OrderedDict()
    def faces(self,xi,s,x,bank_rows):
        key=(int(xi),int(s))
        if key in self.cache:
            v=self.cache.pop(key);self.cache[key]=v;return v
        q=np.asarray(bank_rows,np.float32)
        with torch.no_grad():
            point=torch.as_tensor(np.asarray(x),dtype=torch.float32,device=self.oracle.device)
            qt=torch.as_tensor(q,dtype=torch.float32,device=self.oracle.device)
            _,_,_,active=self.oracle.upstream.visibility_g_batch(
                point,qt,self.oracle.specs,FOV["horizontal_fov_deg"],FOV["vertical_fov_deg"],
                FOV["z_min"],FOV["z_max"],FOV["delta"])
            v=active[:,int(s)].detach().cpu().numpy().astype(np.int8)
        self.cache[key]=v
        while len(self.cache)>self.cap:self.cache.popitem(last=False)
        return v


def _attempt(plane,seed,seed_id,seed_kind,zq,memo,lo,hi,cfg):
    h0,_=memo.evaluate(zq);other=np.arange(len(h0))!=plane
    cons=[dict(type="eq",fun=lambda z,p=plane:memo.evaluate(z)[0][p],
               jac=lambda z,p=plane:memo.evaluate(z)[1][p])]
    if other.any():
        cons.append(dict(type="ineq",fun=lambda z,k=other:memo.evaluate(z)[0][k],
                         jac=lambda z,k=other:memo.evaluate(z)[1][k]))
    t0=time.perf_counter();calls=memo.calls
    try:
        r=minimize(lambda z:.5*np.sum((z-zq)**2),np.asarray(seed,float),
          jac=lambda z:z-zq,method="SLSQP",constraints=cons,bounds=list(zip(lo,hi)),
          options=dict(ftol=cfg.ftol,maxiter=cfg.maxiter,disp=False))
        row=dict(plane=int(plane),seed_id=int(seed_id),seed_kind=str(seed_kind),
          q_seed=memo.full_q(seed).tolist(),q_star=memo.full_q(r.x).tolist(),
          success=bool(r.success),optimizer_status=int(r.status),message=str(r.message),
          iterations=int(r.nit))
    except (FloatingPointError,np.linalg.LinAlgError) as exc:
        row=dict(plane=int(plane),seed_id=int(seed_id),seed_kind=str(seed_kind),
                 success=False,message=repr(exc))
    row.update(elapsed_ms=1000*(time.perf_counter()-t0),geometry_calls=memo.calls-calls)
    return row


def solve_fast(q,bank_rows,mask,lo_full,hi_full,geometry,face_ids,cfg=None,vc=None,
               face_starts=2,competitive_ratio=.25,competitive_abs=.01):
    """Face-aware first pass, then targeted/full fallback when evidence is weak."""
    cfg=cfg or core.SolverConfig();vc=vc or core.VerifyConfig()
    q=np.asarray(q,float);mask=np.asarray(mask,float);lo_full=np.asarray(lo_full,float);hi_full=np.asarray(hi_full,float)
    active=np.flatnonzero(mask);zq=q[active];lo=lo_full[active];hi=hi_full[active]
    bank=np.asarray(bank_rows,float);faces=np.asarray(face_ids,int)
    if len(bank)!=len(faces):raise ValueError("bank/face length mismatch")
    memo=ExactMemo(geometry,q,active);hq,_=memo.evaluate(zq);nface=len(hq)
    d=np.linalg.norm(bank[:,active]-zq[None],axis=1)
    attempts=[];used={p:[] for p in range(nface)}
    # First pass: nearest face-matched boundary seeds. Missing face gets query + nearest global seed.
    for p in range(nface):
        ids=np.flatnonzero(faces==p)
        if len(ids):
            ids=ids[np.argsort(d[ids],kind="stable")[:face_starts]]
            seeds=[(bank[i,active],f"face_bank:{int(i)}") for i in ids]
        else:
            nearest=int(np.argmin(d))
            seeds=[(zq,"query_missing_face"),(bank[nearest,active],f"global_bank:{nearest}")]
        for sid,(seed,kind) in enumerate(seeds):
            attempts.append(_attempt(p,seed,sid,kind,zq,memo,lo,hi,cfg));used[p].append(np.asarray(seed,float))
    screened=core.screen_candidates(q,mask,lo_full,hi_full,geometry,attempts,cfg,vc)
    qual={}
    for c in screened["checked_candidates"]:
        if c["qualified"]:qual.setdefault(int(c["plane"]),[]).append(c)
    best_d=min((c["distance_rad"] for cc in qual.values() for c in cc),default=math.inf)
    fallback_planes=set()
    for p in range(nface):
        cc=sorted(qual.get(p,[]),key=lambda x:x["distance_rad"])
        if not cc:
            fallback_planes.add(p)
        elif cc[0]["distance_rad"]<=best_d*(1+competitive_ratio)+competitive_abs and len(cc)<2:
            fallback_planes.add(p)
    if (not screened["value_valid"]) or screened["ambiguity"] or screened["uncertain_competitor"]:
        fallback_planes=set(range(nface))
    # Original seed policy, but only for planes that need challenge/fallback; skip already-used exact starts.
    original=core.legacy._seeds(q,bank,active,lo,hi,cfg)
    added=0
    for p in sorted(fallback_planes):
        for seed in original:
            if any(np.linalg.norm(seed-u)<1e-9 for u in used[p]):continue
            attempts.append(_attempt(p,seed,1000+added,"original_fallback",zq,memo,lo,hi,cfg))
            used[p].append(np.asarray(seed,float));added+=1
    final=core.screen_candidates(q,mask,lo_full,hi_full,geometry,attempts,cfg,vc)
    final["fast_diagnostics"]=dict(
      first_pass_attempts=sum(min(face_starts,int((faces==p).sum())) if np.any(faces==p) else 2 for p in range(nface)),
      fallback_planes=sorted(fallback_planes),fallback_attempts=added,
      total_attempts=len(attempts),memo_geometry_calls=memo.calls,memo_hits=memo.hits,
      face_bank_counts=[int((faces==p).sum()) for p in range(nface)])
    return attempts,final
