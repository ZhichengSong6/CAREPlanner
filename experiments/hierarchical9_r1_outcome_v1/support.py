"""Pinned repository loading, source checks and frozen-pair sampling. Read-only inputs."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = 'experiments/hierarchical9_scratch50k_r012_v1'
CACHE_SHA = '060eef93a0030800c8ef2265258c7dff850944ab9c5df52382adb98cb506f7cc'
PAIR_SHA = '2ad60ec0c6b109ba25b0d77eb0714c96cf0f0d5df3ca07122f72efe7e3265dd1'
R1_SHA = '4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002'
STREAM = '1e1c4c0186b487e8e5d647ba4c9b8afa7b220771bb85bdce88ae2b71dcd4886e'
FORMAT = 'care_h9_r1_outcome_v1'
PINNED = {
 BASE+'/train.py':'8092a791585dff8ac6ab5e562d158450b1095df4',
 BASE+'/model.py':'0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b',
 'experiments/hierarchical9_scratch_v1/train.py':'88d365045bb1c53cf71f5b8fbd9f8db74a5c2ae9',
 'experiments/hierarchical9_scratch_v1/objective.py':'a53539d27fb003d6b06d46da80ca0c7a93b1230c',
 'experiments/hierarchical9_scratch_v1/model.py':'33580154af2e9cda22f9c64f15cd5b4f5417a86f',
 'experiments/hierarchical9_boundary_audit_v1/oracle.py':'de6d93c02753da9f347a314643c2feca0748629e',
 'experiments/hierarchical9_boundary_audit_v1/runtime_probe.py':'ea8c0875f6e996bcd1159dbd14fccf4555f9cbb6',
 'src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py':'c1de7ef95b4cdab8046882bfaf9ffdf62a0c1141',
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def load(name, path):
    spec=importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None: raise ImportError(path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod);return mod

def fingerprint(repo):
    for rel,digest in PINNED.items():
        b=(repo/rel).read_bytes()
        got=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if got!=digest: raise ValueError(f'Pinned original source changed: {rel}')
    files=list(PINNED)+[
      'src/care_visibility_cdf/scripts/extract_visibility_zero_level_sets.py',
      'src/care_visibility_cdf/scripts/train_signed_visibility_cdf_pairwise_replace.py',
      'src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py',
    ]
    result={rel:sha(repo/rel) for rel in files}
    result.update({'outcome/'+n:sha(HERE/n) for n in
                   ('driver.py','engine.py','rollout.py','support.py','protocol.json')})
    return result

def original_split(x, valid, seed=0, count=1000):
    pool=np.flatnonzero(np.asarray(valid,bool).any(axis=(1,2)))
    np.random.default_rng(seed).shuffle(pool)
    n=min(count,max(1,len(pool)//10))
    return {'train':pool[n:], 'val':pool[:n]}

class Pairs:
    """Verified existing PS pair cache, restricted to ORIGINAL R1 spatial split.

    This reuses pair DATA only, never PS weights or PS losses. Sampler RNG is
    local/stateless, so it cannot perturb R1's global x/q stream.
    """
    def __init__(self, paths, allowed=None, *, verify=True):
        self.cache=Path(paths['cache']); self.root=Path(paths['pairs'])
        cm=read(self.cache/'manifest.json');pm=read(self.root/'manifest.json')
        if sha(self.cache/'manifest.json')!=CACHE_SHA or pm.get('cache_identity')!=CACHE_SHA:
            raise ValueError('Unexpected V4 cache identity')
        if (pm.get('format')!='care_h9_v4_verified_pairs_v1' or pm.get('status')!='COMPLETE'
            or cm.get('status')!='COMPLETE' or not cm.get('training_ready')):
            raise ValueError('Incomplete pair/cache source')
        self.identity=sha(self.root/'manifest.json')
        if self.identity!=PAIR_SHA:raise ValueError('Unexpected original pair manifest SHA')
        self.source_hashes={str(self.root/'manifest.json'):self.identity,
                            str(self.cache/'manifest.json'):CACHE_SHA}
        if verify:
            specs={str(self.root/rel):dg for rel,dg in pm['fields'].items()}
            if not all(Path(p).resolve().is_relative_to(self.root.resolve()) for p in specs):
                raise ValueError('Pair manifest path escapes source')
            xspec=cm['files']['x.npy'];specs[str(self.cache/'x.npy')]=xspec['sha256']
            for p,dg in specs.items():
                if sha(p)!=dg:raise ValueError('Corrupt pair/cache file: '+p)
            self.source_hashes.update(specs)
        self.x=np.load(self.cache/'x.npy',mmap_mode='r',allow_pickle=False)
        if allowed is None:
            with np.load(paths['data'],allow_pickle=True) as z:
                xx=np.asarray(z['x'],np.float32)
                allowed=original_split(xx,z['valid_fov'])
            if xx.shape!=self.x.shape or not np.allclose(xx,self.x,rtol=0,atol=1e-6):
                raise ValueError('R1/V4 x-index mapping differs')
        self.arrays={};self.groups={};self.counts={}
        for sp in ('train','val'):
            a={k:np.load(self.root/sp/(k+'.npy'),mmap_mode='r',allow_pickle=False)
               for k in ('x_index','sensor','radius_id','q_plus','q_minus')}
            n=pm['total_by_split'][sp]
            if any(len(v)!=n for v in a.values()) or a['q_plus'].shape!=(n,7) or a['q_minus'].shape!=(n,7):
                raise ValueError('Bad pair array layout')
            valid=np.isin(a['x_index'],allowed[sp])
            groups=[np.flatnonzero(valid & (a['sensor']==s) & (a['radius_id']==r))
                    for s in range(8) for r in range(3)]
            if any(len(g)==0 for g in groups):raise ValueError('Missing R1-split sensor/radius pair cell')
            self.arrays[sp]=a;self.groups[sp]=groups;self.counts[sp]=[len(g) for g in groups]

    def batch(self, split, step, rank, device, per_sensor=4):
        rng=np.random.default_rng(np.random.SeedSequence([261014,step,rank,split=='val']))
        ids=[]
        for s in range(8):
            for j in range(per_sensor):
                radius=(step+rank+j)%3
                g=self.groups[split][s*3+radius]
                ids.append(int(g[rng.integers(len(g))]))
        ids=np.asarray(ids,np.int64);a=self.arrays[split]
        def t(v,dtype):return torch.as_tensor(np.asarray(v).copy(),device=device,dtype=dtype)
        return (t(self.x[a['x_index'][ids]],torch.float32),t(a['q_minus'][ids],torch.float32),
                t(a['q_plus'][ids],torch.float32),t(a['sensor'][ids],torch.long))

class Geometry:
    def __init__(self, repo, urdf, device, joint_names, sensor_frames):
        scripts=repo/'src/care_visibility_cdf/scripts'
        if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
        mod=load('outcome_original_oracle',repo/'experiments/hierarchical9_boundary_audit_v1/oracle.py')
        self.oracle=mod.SensorOracle(urdf,device,joint_names,sensor_frames)

    def __call__(self,x,q,sensor):
        import math
        values=q[:,0]*0
        o=self.oracle
        for s in range(8):
            keep=(sensor==s);idx=torch.nonzero(keep).reshape(-1)
            if not len(idx):continue
            transform=o.fk(o.specs[s],q[idx])
            p=torch.einsum('nji,nj->ni',transform[:,:3,:3],x[idx]-transform[:,:3,3])
            px,py,pz=p.unbind(-1)
            ax,ay=math.tan(math.radians(o.hfov)/2),math.tan(math.radians(o.vfov)/2)
            nx,ny=math.sqrt(1+ax*ax),math.sqrt(1+ay*ay)
            planes=torch.stack(((px+pz*ax)/nx,(-px+pz*ax)/nx,
                                (py+pz*ay)/ny,(-py+pz*ay)/ny,pz-o.zmin,o.zmax-pz),-1)-o.delta
            values=values.index_copy(0,idx,planes.min(-1).values)
        return values

    def parity(self,batch):
        x,qm,qp,sensors=batch
        with torch.no_grad():
            a=self(x,qm,sensors)
            err=max(abs(float(a[i])-self.oracle.value(x[i],qm[i],int(sensors[i]))) for i in range(len(x)))
        if err>2e-6:raise ValueError(f'Paired geometry differs from original oracle: {err}')
        # Independent upstream FK/FOV check, for one point per sensor.
        for s in range(8):
            i=int(torch.nonzero(sensors==s)[0])
            self.oracle.verify_against_upstream(x[i],torch.stack((qm[i],qp[i])))
        return err

def reference_class(repo):
    """Compile ONLY the actual frozen runtime's five pure solver methods.

    Avoid ROS/mesh imports in CPU regression without copying their algorithms.
    """
    import ast, math
    path=repo/'src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PerSensorVisibilityRuntime')
    names={'_clamp','_projection_step','_ascent_step','_refine_branch_root','_optimize_branch'}
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(methods)!=len(names):raise ValueError('Runtime solver methods missing')
    # Postponed annotations allow loading the original signatures independently.
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
          ast.ClassDef(name='Reference',bases=[],keywords=[],body=methods,decorator_list=[])]
    module=ast.fix_missing_locations(ast.Module(body=body,type_ignores=[]))
    env={'torch':torch,'math':math,'np':np,'DEFAULT_SENSOR_FRAMES':[f'S{s}' for s in range(8)]}
    exec(compile(module,str(path),'exec'),env)
    return env['Reference']

def check_solver_parity(repo, model, batch, masks, lo, hi, training=False):
    from rollout import SETTINGS, solve
    from dataclasses import asdict
    Reference=reference_class(repo)
    x,qm,qp,s=batch
    xx,ss=torch.cat([x,x]),torch.cat([s,s]);start=torch.cat([qm,qp])
    def score(q):return model(torch.cat([xx,q],1)).gather(1,(ss+1)[:,None])[:,0]
    predicted,_=solve(score,start,masks[ss],lo,hi,training=training)
    expected=[]
    for i in range(len(start)):
        probe=Reference()
        for k,v in asdict(SETTINGS).items():setattr(probe,k,v)
        probe.q_min,probe.q_max=lo,hi
        def val(points,q,sid):
            with torch.no_grad():return float(model.forward_sensor(torch.cat([points,q],1),sid)[0])
        def grad(points,q,sid):
            with torch.enable_grad():
                q=q.detach().clone().requires_grad_(True)
                f=model.forward_sensor(torch.cat([points,q],1),sid)
                g=torch.autograd.grad(f.sum(),q)[0]*masks[sid]
                return float(f.detach()[0]),g.detach()
        probe.branch_score,probe.branch_value_and_grad=val,grad
        r=probe._optimize_branch(xx[i:i+1],start[i:i+1],int(ss[i]))
        expected.append(r['q_candidate'])
    expected=torch.as_tensor(expected,device=start.device,dtype=start.dtype)
    error=float((predicted.detach()-expected).abs().max())
    if error>1e-4:raise ValueError(f'Training rollout != original runtime: max q error={error}')
    return error
