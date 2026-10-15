"""Independent bounded geometry search; no neural model or checkpoint is loaded.

V4 anchors are INITIAL GUESSES only. Every witness and local +/- pair is
rechecked at the NEW x with the exact upstream FP32 FOV used in evaluation.
Failure of this bounded search is UNRESOLVED, never an infeasibility claim.
"""
from __future__ import annotations
from pathlib import Path
import os
import tempfile
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from common import require, load, read, sha, write, write_lines, verify_sources


class SeedBank:
    def __init__(self,cache,cfg):
        root=Path(cache);self.cfg=cfg
        self.x=np.load(root/'x.npy',mmap_mode='r',allow_pickle=False)
        self.a={};self.index={};self.trees={};self.xids={}
        for sp in ('train','val'):
            a={k:np.load(root/sp/f'boundary_{k}.npy',mmap_mode='r',allow_pickle=False)
               for k in ('x_index','sensor','q')}
            n=len(a['sensor']);require(a['q'].shape==(n,7) and len(a['x_index'])==n,'Bad seed shape')
            codes=a['x_index'].astype(np.int64)*8+a['sensor']
            order=np.argsort(codes,kind='stable');sorted_codes=codes[order]
            u,start,count=np.unique(sorted_codes,return_index=True,return_counts=True)
            for code,first,num in zip(u,start,count):
                require(int(code) not in self.index,'Train/val seed x-sensor overlap')
                self.index[int(code)]=(sp,order[first:first+num])
            self.a[sp]=a
        for s in range(8):
            ids=np.array(sorted({code//8 for code in self.index if code%8==s}),np.int64)
            require(len(ids)>=cfg['nearest_seed_x'],f'Too few seed x: S{s}')
            self.xids[s]=ids;self.trees[s]=cKDTree(np.asarray(self.x[ids],np.float64))

    def seeds(self,x,s):
        _,local=self.trees[s].query(x,k=self.cfg['nearest_seed_x'])
        chosen=[];provenance=[]
        for pos in np.atleast_1d(local):
            xi=int(self.xids[s][pos]);sp,ids=self.index[8*xi+s]
            k=min(self.cfg['seeds_per_nearest_x'],len(ids))
            for row in ids[np.linspace(0,len(ids)-1,k,dtype=int)]:
                q=np.asarray(self.a[sp]['q'][row],np.float64)
                if np.isfinite(q).all():chosen.append(q);provenance.append((sp,xi,int(row)))
        return np.asarray(chosen).reshape(-1,7),provenance


class ExactGeometry:
    def __init__(self,job):
        import torch
        self.torch=torch;self.cfg=job['cfg'];repo=Path(job['repo'])
        ro=load('offgrid_source_oracle',repo/'experiments/hierarchical9_offline_relabel_v1/repo_oracle.py')
        self.oracle=ro.RepoOracle(repo,job['urdf'],'cpu')
        ag=load('offgrid_analytic',repo/'experiments/hierarchical9_offline_relabel_v3_fast_v1/analytic_geometry.py')
        self.factory=ag.AnalyticGeometry;self.fov=ro.FOV
        require(np.array_equal(self.oracle.masks,np.asarray(job['masks'])), 'URDF/data sensor masks disagree')
        require(np.allclose(self.oracle.urdf_lo,job['lo'],rtol=0,atol=1e-6) and
                np.allclose(self.oracle.urdf_hi,job['hi'],rtol=0,atol=1e-6),'URDF/data limits disagree')

    def geometry(self,x,s):return self.factory(self.oracle,x,s)

    def canonical(self,x,qs,s):
        """Original FP32 margins, not the double optimization surrogate."""
        t=self.torch;qs=np.asarray(qs,np.float32).reshape(-1,7)
        with t.no_grad():
            _,m,_,_=self.oracle.upstream.visibility_g_batch(t.tensor(np.asarray(x,np.float32)),
                t.tensor(qs),self.oracle.specs,self.fov['horizontal_fov_deg'],self.fov['vertical_fov_deg'],
                self.fov['z_min'],self.fov['z_max'],self.fov['delta'])
        return (m[:,s]-self.fov['delta']).cpu().numpy().astype(float)

    def check_parity(self,x,q,s):
        fn=self.geometry(x,s);h,j=fn(q);v=float(self.canonical(x,[q],s)[0])
        err=abs(v-float(h.min()))
        require(err<=self.cfg['geometry_parity_m'],f'FP32/double FOV parity failed S{s}: {err}')
        eps=1e-6;fd=np.stack([(fn(q+eps*np.eye(7)[i])[0]-fn(q-eps*np.eye(7)[i])[0])/(2*eps) for i in range(7)],1)
        je=float(np.abs(fd-j).max())
        require(je<1e-6,f'Analytic joint Jacobian check failed S{s}: {je}')
        return {'value_max_abs_m':err,'jacobian_max_abs':je}


def witness(fn,seeds,lo,hi,mask,cfg):
    """Maximize minimum plane margin with fixed finite budget; retain real feasible q."""
    seeds=np.maximum(np.minimum(np.asarray(seeds,float),hi),lo)
    scores=np.asarray([fn(q)[0].min() for q in seeds])
    require(np.isfinite(scores).all(),'Nonfinite geometry during witness search')
    order=np.argsort(-scores,kind='stable');best=seeds[order[0]].copy();bg=float(scores[order[0]])
    attempts=[];active=np.flatnonzero(mask)
    if bg>=cfg['witness_margin_m']:return best,bg,attempts
    # q inactive dimensions stay at the selected seed. No q0 distance minimization.
    for i in order[:cfg['witness_optimizer_starts']]:
        fixed=seeds[i].copy()
        def unpack(z):
            q=fixed.copy();q[active]=z[:-1];return q
        def cons(z):return fn(unpack(z))[0]-z[-1]
        def jac(z):return np.c_[fn(unpack(z))[1][:,active],-np.ones(6)]
        z=np.r_[fixed[active],float(scores[i])]
        res=minimize(lambda v:-v[-1],z,jac=lambda v:np.r_[np.zeros(len(active)),-1.],
                     method='SLSQP',bounds=list(zip(lo[active],hi[active]))+[(-10.,.25)],
                     constraints={'type':'ineq','fun':cons,'jac':jac},
                     options={'maxiter':cfg['witness_maxiter'],'ftol':1e-9,'disp':False})
        q=unpack(res.x);hv=fn(q)[0]
        require(np.isfinite(hv).all() and np.isfinite(q).all(),'Nonfinite optimizer output')
        g=float(hv.min());attempts.append({'status':int(res.status),'success':bool(res.success),
                           'iterations':int(res.nit),'verified_min_margin_m':g})
        if g>bg and in_limits(q,lo,hi):best,bg=q,g
        if bg>=cfg['witness_margin_m']:break
    return best,bg,attempts


def in_limits(q,lo,hi):
    q=np.asarray(q);return bool(np.isfinite(q).all() and np.all(q>=lo) and np.all(q<=hi))


def boundary(fn,inside,outside,mask,cfg):
    """Bracket a continuous FOV sign change; no claim about nearest boundary."""
    a=np.asarray(inside,float).copy();b=np.asarray(outside,float).copy()
    require(fn(a)[0].min()>0 and fn(b)[0].min()<0,'Invalid boundary bracket')
    q=(a+b)/2
    for _ in range(cfg['root_steps']):
        q=(a+b)/2;g=float(fn(q)[0].min())
        if abs(g)<=cfg['root_tolerance_m']:break
        if g>0:a=q
        else:b=q
    h,j=fn(q);active=int(h.argmin());normal=j[active]*mask;norm=np.linalg.norm(normal)
    if abs(float(h.min()))>1e-6 or norm<1e-8:return None
    return q,normal/norm,active,float(h.min())


def generate_pair(x,pid,s,seeds,exact,lo,hi,mask,cfg):
    fn=exact.geometry(x,s)
    # Witness search RNG and uniform evaluation-start RNG are independent.
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],pid,s,101]))
    random=rng.uniform(lo,hi,(cfg['witness_random_q'],7))
    search=np.vstack([seeds,random,(lo+hi)[None,:]/2])
    w,g,attempts=witness(fn,search,lo,hi,mask,cfg)
    w=w.astype(np.float32);wg=float(exact.canonical(x,[w],s)[0])
    info={'point_id':pid,'sensor':s,'status':'UNRESOLVED_WITNESS','witness_margin_m':wg,
          'optimizer_attempts':attempts,'uniform_count':0,'local_count':0,'local_exclusions':[]}
    cases=[]
    if not in_limits(w,lo,hi) or wg<cfg['witness_margin_m']:return info,cases
    info.update(status='WITNESS_CONFIRMED',witness_q=w.tolist())
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],pid,s,202]))
    candidates=rng.uniform(lo,hi,(cfg['uniform_attempts'],7)).astype(np.float32)
    margins=exact.canonical(x,candidates,s)
    require(np.isfinite(margins).all(),'Nonfinite uniform initial margins')
    chosen=np.flatnonzero(margins < -cfg['side_margin_m'])[:cfg['uniform_per_pair']]
    for k,i in enumerate(chosen):
        cases.append({'point_id':pid,'sensor':s,'cohort':'uniform','group':f'S{s}/uniform_outside',
           'x':x.tolist(),'q_init':candidates[i].tolist(),'initial_g_m':float(margins[i]),
           'witness_q':w.tolist(),'witness_g_m':wg,'sample_slot':int(k)})
    info['uniform_count']=len(chosen)
    if not len(chosen):info['local_exclusions']=['NO_OUTSIDE_START_FOR_BRACKET'];return info,cases
    br=boundary(fn,w,candidates[chosen[0]],mask,cfg)
    if br is None:info['local_exclusions']=['BOUNDARY_REFINEMENT_NOT_QUALIFIED'];return info,cases
    qstar,n,face,resid=br
    for radius in cfg['local_radii_rad']:
        minus=(qstar-radius*n).astype(np.float32);plus=(qstar+radius*n).astype(np.float32)
        if not in_limits(minus,lo,hi) or not in_limits(plus,lo,hi):
            info['local_exclusions'].append({'radius':radius,'reason':'JOINT_LIMIT'});continue
        gm,gp=exact.canonical(x,[minus,plus],s)
        if not (gm < -cfg['side_margin_m'] and gp > cfg['side_margin_m']):
            info['local_exclusions'].append({'radius':radius,'reason':'TWO_SIDED_FOV_FAIL','gm':gm,'gp':gp});continue
        cases.append({'point_id':pid,'sensor':s,'cohort':'local','group':f'S{s}/local_r{radius}',
           'x':x.tolist(),'q_init':minus.tolist(),'initial_g_m':float(gm),
           'witness_q':plus.tolist(),'witness_g_m':float(gp),'radius_rad':radius,
           'q_boundary':qstar.tolist(),'normal':n.tolist(),'active_plane':face,'boundary_residual_m':resid})
        info['local_count']+=1
    return info,cases


def generate(job,rank):
    out=Path(job['out']);verify_sources(job)
    require(sha(out/'points.json')==job['points_sha256'],'Point set changed')
    cfg=job['cfg'];points=np.asarray(read(out/'points.json')['points'],np.float32)
    exact=ExactGeometry(job);seedbank=SeedBank(job['cache'],cfg)
    lo=np.asarray(job['lo'],float);hi=np.asarray(job['hi'],float);masks=np.asarray(job['masks'],float)
    output=[];coverage=[];parity=[]
    for pid in range(rank,len(points),cfg['world']):
        x=points[pid]
        for s in range(8):
            seeds,prov=seedbank.seeds(x,s)
            if pid==rank:parity.append(exact.check_parity(x,(lo+hi)/2,s))
            info,cases=generate_pair(x,pid,s,seeds,exact,lo,hi,masks[s],cfg)
            info['seed_sources']=[list(v) for v in prov];coverage.append(info);output.extend(cases)
        if ((pid-rank)//cfg['world']+1)%8==0:
            print(f'[geometry rank{rank}] points={(pid-rank)//cfg["world"]+1}/{len(points)//cfg["world"]} starts={len(output)}',flush=True)
    base=out/'geometry';base.mkdir(exist_ok=True)
    final=base/f'rank{rank}'
    require(not final.exists(),f'Existing geometry shard: {final}')
    folder=Path(tempfile.mkdtemp(prefix=f'.rank{rank}.building.',dir=base))
    f=folder/'starts.jsonl';c=folder/'coverage.json'
    write_lines(f,output);write(c,{'rows':coverage,'parity':parity})
    write(folder/'complete.json',{'status':'COMPLETE','rank':rank,'files':{f.name:sha(f),c.name:sha(c)}})
    os.rename(folder,final)
    print(f'[geometry rank{rank}] COMPLETE starts={len(output)}',flush=True)
