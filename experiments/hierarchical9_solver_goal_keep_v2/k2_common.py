"""Isolated, content-addressed experiment; never modify upstream or parent weights."""
from __future__ import annotations
import hashlib, importlib.util, json, os, shutil, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FORMAT = 'care_solver_goal_keep_v2'
TEACHER_SHA = '5507b1d16c5d34f4a3e8c01526c7bf3f278a51c07df9f4d878aaee476cee48f5'
R1_SHA = '4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002'
CACHE_SHA = '060eef93a0030800c8ef2265258c7dff850944ab9c5df52382adb98cb506f7cc'
ARMS = ('A_continue', 'B_keep')
DEPDIRS = ('hierarchical9_solver_goal_v1', 'hierarchical9_scratch50k_r012_v1',
 'hierarchical9_scratch_v1', 'hierarchical9_paired_slope_scratch_v1',
 'hierarchical9_boundary_audit_v1', 'hierarchical9_offline_relabel_v1',
 'hierarchical9_offline_relabel_v3_fast_v1', 'hierarchical9_solver_goal_offgrid_v1')

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

def safe(v):
    if isinstance(v, dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v, (list, tuple)): return [safe(x) for x in v]
    if isinstance(v, np.ndarray): return safe(v.tolist())
    if isinstance(v, np.generic): return safe(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v

def write(path, value, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+f'.tmp.{os.getpid()}')
    with tmp.open('x') as f:
        json.dump(safe(value), f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    try:
        if replace: os.replace(tmp,path)
        else: os.link(tmp,path)
    finally: tmp.unlink(missing_ok=True)

def lines(path):
    with Path(path).open() as f:
        for text in f:
            if text.strip(): yield json.loads(text)

def write_lines(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+f'.tmp.{os.getpid()}')
    with tmp.open('x') as f:
        for row in rows: f.write(json.dumps(safe(row),sort_keys=True,allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())
    try: os.link(tmp,path)
    finally: tmp.unlink(missing_ok=True)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,f'Cannot import {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod);return mod

@contextmanager
def aliases(**modules):
    previous={k:sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    try: yield
    finally:
        for k,v in previous.items():
            if v is None: sys.modules.pop(k,None)
            else: sys.modules[k]=v

def modules(job):
    root=Path(job['deps']); sg=root/'experiments/hierarchical9_solver_goal_v1'
    scripts=str(root/'src/care_visibility_cdf/scripts')
    if scripts not in sys.path: sys.path.insert(0,scripts)
    goal=load('k2_upstream_goal',sg/'goal.py')
    with aliases(goal=goal): st=load('k2_upstream_sg_train',sg/'train.py')
    core=load('k2_original_r012',root/'experiments/hierarchical9_scratch50k_r012_v1/train.py')
    return core,goal,st

def runtime(job,device):
    core,goal,st=modules(job)
    oracle=st.oracle_for(Path(job['deps']),Path(job['urdf']),device)
    probe=load('k2_runtime_probe',Path(job['deps'])/'experiments/hierarchical9_boundary_audit_v1/runtime_probe.py')
    from hierarchical_visibility_cdf_model import HierarchicalSensorView
    lo,hi,masks=(torch.tensor(job[k],device=device,dtype=torch.float32) for k in ('lo','hi','masks'))
    def make(model): return probe.make_probe(HierarchicalSensorView(model),masks,lo,hi)
    return core,goal,st,oracle,probe,make,lo,hi,masks

def model_digest(model):
    h=hashlib.sha256()
    for key,v in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def load_model(core,path,digest,device):
    require(sha(path)==digest,'Checkpoint SHA mismatch')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    model=core.models.build_model('R1'); model.load_state_dict(cp['model_state'],strict=True)
    return model.to(device),cp

def stage(folder):
    folder=Path(folder);m=read(folder/'complete.json')
    require(m.get('status')=='COMPLETE',f'Incomplete {folder}')
    for name,digest in m['files'].items():
        f=(folder/name).resolve();require(f.is_relative_to(folder.resolve()),'Unsafe stage file')
        require(sha(f)==digest,f'Corrupt stage {f}')
    return m

def finish(folder,**metadata):
    folder=Path(folder)
    files={str(f.relative_to(folder)):sha(f) for f in folder.rglob('*') if f.is_file() and not f.name.startswith('.') and f.name!='complete.json'}
    write(folder/'complete.json',dict(status='COMPLETE',files=files,**metadata))

def verify(job, full=False):
    require(job['format']==FORMAT and job['cfg']==read(HERE/'protocol.json'),'Experiment protocol changed')
    require(sha(Path(job['out'])/'splits.npz')==job['splits_sha256'],'Frozen train/val split changed')
    require(sha(Path(job['out'])/'confirmation_points.json')==job['confirmation_points_sha256'],'Frozen new test points changed')
    require(job['code_hashes']=={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh','.json')},'Frozen new source changed')
    for rel,digest in job['dep_hashes'].items(): require(sha(Path(job['deps'])/rel)==digest,f'Frozen dependency changed {rel}')
    for name,spec in job['inputs'].items():
        p=Path(name);stat=p.stat()
        require(stat.st_size==spec['bytes'] and stat.st_mtime_ns==spec['mtime_ns'],f'Input changed {p}')
        if full or p.name.endswith('.json'): require(sha(p)==spec['sha256'],f'Input hash changed {p}')

def prepare(repo,sgroot,out):
    repo,sgroot,out=map(lambda x:Path(x).resolve(),(repo,sgroot,out))
    cfg=read(HERE/'protocol.json');require(cfg['format']==FORMAT and cfg['updates']==20000,'Bad protocol')
    require(not out.exists(),'Existing experiment: use resume, never overwrite')
    parent=read(sgroot/'job.json');complete=read(sgroot/'formal/solver_goal_complete.json')
    teacher=sgroot/'formal/final.pt'
    require(complete.get('status')=='COMPLETE' and complete.get('base_stream_match') is True and sha(teacher)==TEACHER_SHA==complete.get('final_sha256'),'Wrong/incomplete parent')
    cache,pairs,data=map(Path,(parent['cache'],parent['pairs'],parent['data']))
    for source in (repo,sgroot,cache,pairs,data):
        require(not(out==source or out.is_relative_to(source) or source.is_relative_to(out)),'Output overlaps protected input')
    require(sha(cache/'manifest.json')==CACHE_SHA and sha(pairs/'manifest.json')==parent['pair_sha'],'Cache identity changed')
    sourcefiles={}
    for d in DEPDIRS:
        for f in (repo/'experiments'/d).glob('*'):
            if f.is_file() and f.suffix in ('.py','.json'): sourcefiles[str(f.relative_to(repo))]=f
    for f in (repo/'src/care_visibility_cdf/scripts').glob('*.py'): sourcefiles[str(f.relative_to(repo))]=f
    rel_urdf='src/arm_description/urdf/Arm.urdf';sourcefiles[rel_urdf]=repo/rel_urdf
    # Match parent training code, never infer SHA-256 from a Git blob SHA-1.
    for rel,digest in parent['source'].items():
        if rel.startswith('solver_goal/'): path=repo/'experiments/hierarchical9_solver_goal_v1'/rel.split('/',1)[1]
        else: path=repo/rel
        require(sha(path)==digest,f'Parent training source changed {rel}')
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.prepare.',dir=out.parent))
    try:
        deps=tmp/'dependencies'
        for rel,f in sourcefiles.items():
            target=deps/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
        tempjob={'deps':str(deps)};core,goal,st=modules(tempjob)
        x,tr,va,masks,lo,hi=st.splits(data,parent['cfg'])
        cc=st.open_cache(deps,cache)
        require(np.array_equal(x,cc.x),'R1 and pair x-index differ')
        st.checked_files(cache,cc.manifest['files'],True)
        pm=read(pairs/'manifest.json');st.checked_files(pairs,pm['fields'],False)
        pool=goal.WitnessPool(cc,pairs,tr,va,parent['cfg'])
        require(sha(data)==parent['data_sha'],'Original data hash mismatch')
        cp=torch.load(teacher,map_location='cpu',weights_only=False)
        require(cp['step']==50000 and cp.get('metadata',{}).get('solver_goal_v1')==parent,'Teacher lineage mismatch')
        require(sha(deps/rel_urdf)==cp['metadata']['urdf_sha256'],'URDF differs from teacher training')
        require(cp.get('training_stream_sha256')==parent['base_stream'],'Teacher base stream mismatch')
        inputs={}
        paths=[teacher,sgroot/'job.json',sgroot/'formal/solver_goal_complete.json',data,cache/'manifest.json',pairs/'manifest.json']
        oldcp=Path(parent['r012'])/'formal/R1/final.pt'
        require(sha(oldcp)==R1_SHA,'Original R1 weight changed')
        previous=sgroot.parent/'h9_solver_goal_offgrid_v1/formal/points.json'
        require(previous.is_file(),'Previous off-grid points needed for exclusion; no performance reports read')
        paths += [oldcp,previous]
        paths += [pairs/name for name in pm['fields']]
        paths += [cache/name for name in cc.manifest['files']]
        for p in dict.fromkeys(paths):
            stat=p.stat();inputs[str(p.resolve())]={'sha256':sha(p),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        np.savez_compressed(tmp/'splits.npz',x=x,train=tr,val=va)
        oc=load('k2_offgrid_common',deps/'experiments/hierarchical9_solver_goal_offgrid_v1/common.py')
        ocfg=read(deps/'experiments/hierarchical9_solver_goal_offgrid_v1/protocol.json')
        ocfg.update(seed=cfg['confirmation_seed'],points=cfg['confirmation_points'])
        with np.load(data,allow_pickle=False) as z:
            bounds=np.asarray([[float(z[a]),float(z[b])] for a,b in
                 (('x_min','x_max'),('y_min','y_max'),('z_min_bound','z_max_bound'))])
        points,draws=oc.select_points(np.vstack((x,np.asarray(read(previous)['points']))),bounds,ocfg)
        write(tmp/'confirmation_points.json',dict(points=points.tolist(),seed=cfg['confirmation_seed'],
                   exclusion_source_sha256=sha(previous),draws=draws))
        job=dict(format=FORMAT,cfg=cfg,out=str(out),deps=str(out/'dependencies'),repo=str(repo),
          teacher=str(teacher),teacher_sha256=TEACHER_SHA,r1=str(Path(parent['r012'])/'formal/R1/final.pt'),
          cache=str(cache),pairs=str(pairs),data=str(data),urdf=str(out/'dependencies'/rel_urdf),
          lo=lo.tolist(),hi=hi.tolist(),masks=masks.tolist(),parent=parent,
          dep_hashes={rel:sha(deps/rel) for rel in sourcefiles},
          code_hashes={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh','.json')},
          inputs=inputs,confirmation_cfg=ocfg,confirmation_points_sha256=sha(tmp/'confirmation_points.json'),splits_sha256=sha(tmp/'splits.npz'),initial_model_digest=model_digest(core.models.build_model('R1')))
        # Last two identities refer to weights before any update, not random-init lineage.
        tm=core.models.build_model('R1');tm.load_state_dict(cp['model_state'],strict=True)
        job['initial_model_digest']=model_digest(tm)
        torch.set_num_threads(2)
        oracle=st.oracle_for(deps,deps/rel_urdf,torch.device('cpu'))
        example=pool.draw('train',1,0,torch.device('cpu'))
        for sensor,(xx,qm,qp) in enumerate(example):
            require((goal.paired_planes(oracle,xx,qm,sensor).min(-1).values<0).all(),'Real negative witness failed before submission')
            require((goal.paired_planes(oracle,xx,qp,sensor).min(-1).values>0).all(),'Real positive witness failed before submission')
        parity=oracle.verify_against_upstream(example[0][0][0],example[0][2][:3])
        loss,_=goal.geometry_objective(tm,example,oracle,torch.tensor(masks),torch.tensor(lo),torch.tensor(hi),parent['cfg'])
        gs=torch.autograd.grad(loss,tuple(tm.parameters()),allow_unused=True)
        require(any(g is not None for g in gs) and all(g is None or torch.isfinite(g).all() for g in gs),'Nonfinite real FK-to-parameter gradient')
        job['preflight']=dict(real_witnesses='PASS',upstream_fov_parity=parity,real_model_geometry_backward='PASS')
        del loss,gs,tm
        write(tmp/'job.json',job);os.rename(tmp,out)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    print('[prepared] parent Solver-Goal verified; A/B both start from fixed weights; all upstream dependencies copied',flush=True)
    return job
