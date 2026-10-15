"""Identity, file safety, deterministic point selection. No model deserialization."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
FORMAT = 'care_r1_solver_goal_offgrid_v1'
R1_SHA = '4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002'
NEW_SHA = '5507b1d16c5d34f4a3e8c01526c7bf3f278a51c07df9f4d878aaee476cee48f5'
CACHE_SHA = '060eef93a0030800c8ef2265258c7dff850944ab9c5df52382adb98cb506f7cc'
DATA_SHA = '47a55543870ab86e01424ba3d0d3acadf1d86d7bf9b3781b4beab8d29b8f9172'
STREAM_SHA = '1e1c4c0186b487e8e5d647ba4c9b8afa7b220771bb85bdce88ae2b71dcd4886e'
ARMS = ('R1', 'SOLVER_GOAL')
PINS = {
 'experiments/hierarchical9_scratch50k_r012_v1/model.py':'0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b',
 'experiments/hierarchical9_boundary_audit_v1/runtime_probe.py':'ea8c0875f6e996bcd1159dbd14fccf4555f9cbb6',
 'experiments/hierarchical9_boundary_audit_v1/oracle.py':'de6d93c02753da9f347a314643c2feca0748629e',
 'experiments/hierarchical9_offline_relabel_v3_fast_v1/analytic_geometry.py':'3f3f5fe0167b29d05d80a0493fb5135adbebb844',
 'experiments/hierarchical9_offline_relabel_v1/repo_oracle.py':'265fb09b5e198d6e13d74a837e59941dee01abb8',
}

def require(condition, message):
    if not condition: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def git_blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def safe(x):
    if isinstance(x,dict): return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [safe(v) for v in x]
    if isinstance(x,np.ndarray): return safe(x.tolist())
    if isinstance(x,np.generic): return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def dumps(x):
    return json.dumps(safe(x),sort_keys=True,separators=(',',':'),allow_nan=False)

def write(path,x):
    """Atomic and exclusive publication; a repeat never overwrites previous data."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+f'.tmp.{os.getpid()}')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(safe(x),f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
    try: os.link(tmp,path)
    finally: tmp.unlink(missing_ok=True)

def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: yield json.loads(line)
                except ValueError as e: raise ValueError(f'{path}:{i}: invalid JSON') from e

def write_lines(path,rows):
    path=Path(path);tmp=path.with_name('.'+path.name+f'.tmp.{os.getpid()}')
    with tmp.open('x',encoding='utf-8') as f:
        for r in rows:f.write(dumps(r)+'\n')
        f.flush();os.fsync(f.fileno())
    try:os.link(tmp,path)
    finally:tmp.unlink(missing_ok=True)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,f'Cannot import {path}')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

def config():
    p=read(HERE/'protocol.json')
    require(p['format']==FORMAT and p['points']==512 and p['world']==4,'Formal protocol changed')
    require(p['local_radii_rad']==[.01,.02,.05] and p['uniform_per_pair']==4,'Cohort protocol changed')
    return p

def code_files(repo):
    repo=Path(repo).resolve()
    for rel,digest in PINS.items():
        require(git_blob(repo/rel)==digest,f'Original source changed: {rel}')
    # All runtime script imports are frozen to avoid a dependency changing mid-job.
    deps=list(PINS)+[str(p.relative_to(repo)) for p in sorted((repo/'src/care_visibility_cdf/scripts').glob('*.py'))]
    return {**{rel:sha(repo/rel) for rel in sorted(set(deps))},
      **{'offgrid/'+p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.sh','.json')}}

def verify_sources(job):
    require(job.get('format')==FORMAT and job.get('cfg')==config(),'Job protocol differs from pinned source')
    require(job.get('checkpoint_sha256')=={'R1':R1_SHA,'SOLVER_GOAL':NEW_SHA},'Unexpected model identities')
    require(code_files(job['repo'])==job['code'],'Evaluation source changed since submission')
    for name,item in job['inputs'].items():
        p=Path(name); st=p.stat()
        require(st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns'],f'Input changed: {p}')

def select_points(old_x,bounds,cfg):
    """Fixed box-uniform samples, only geometric noncoincidence exclusions."""
    old_x=np.asarray(old_x,np.float64);bounds=np.asarray(bounds,np.float64)
    require(old_x.ndim==2 and old_x.shape[1]==3 and np.isfinite(old_x).all(),'Invalid old x')
    require(bounds.shape==(3,2) and np.all(bounds[:,1]>bounds[:,0]),'Bad workspace bounds')
    tree=cKDTree(old_x);rng=np.random.default_rng(cfg['seed']);points=[]
    for attempt in range(100000):
        # Check after float32 conversion: these exact bytes will be used by both models.
        x=rng.uniform(bounds[:,0],bounds[:,1]).astype(np.float32)
        if tree.query(x)[0]<cfg['min_distance_to_old_x_m']:continue
        if points and np.linalg.norm(np.asarray(points)-x,axis=1).min()<cfg['min_between_new_x_m']:continue
        points.append(x)
        if len(points)==cfg['points']:break
    require(len(points)==cfg['points'],'Cannot select prescribed off-grid points')
    return np.asarray(points,np.float32),attempt+1

def stage_complete(root,marker):
    root=Path(root);m=read(root/marker)
    require(m.get('status')=='COMPLETE',f'{marker} incomplete')
    for rel,dig in m['files'].items():
        p=(root/rel).resolve();require(p.is_relative_to(root.resolve()),'Unsafe stage manifest')
        require(sha(p)==dig,f'Stage file corrupted: {p}')
    return m

def preflight(repo,sgroot,cache,out):
    """No torch.load: hashes plus sidecars only, before any new test generation."""
    repo,sgroot,cache,out=(Path(p).resolve() for p in (repo,sgroot,cache,out))
    p=config();code=code_files(repo)
    source=read(sgroot/'job.json');done=read(sgroot/'formal/solver_goal_complete.json')
    oldroot=Path(source['r012']).resolve();oldrun=read(oldroot/'formal/R1/run.json')
    newrun=read(sgroot/'formal/run.json');data=Path(source['data']).resolve();urdf=Path(source['urdf']).resolve()
    oldcp=oldroot/'formal/R1/final.pt';newcp=sgroot/'formal/final.pt'
    require(done.get('status')=='COMPLETE' and done.get('steps')==50000 and done.get('base_stream_match') is True,'Solver-Goal incomplete')
    require(done.get('final_sha256')==NEW_SHA and done.get('original_r1_sha')==R1_SHA,'Wrong candidate identity')
    require(oldrun.get('status')=='COMPLETE' and oldrun.get('successful_updates')==50000,'R1 incomplete')
    require(newrun.get('status')=='COMPLETE' and newrun.get('successful_updates')==50000,'Candidate run incomplete')
    require(oldrun.get('training_stream_sha256')==newrun.get('training_stream_sha256')==source['base_stream']==STREAM_SHA,'Training stream mismatch')
    for a in (oldroot,sgroot,cache,data,urdf,repo):
        require(not (out==a or out.is_relative_to(a) or a.is_relative_to(out)),f'Output overlaps protected input {a}')
    manifest=read(cache/'manifest.json')
    require(sha(cache/'manifest.json')==CACHE_SHA and manifest.get('training_ready') is True,'Wrong frozen cache')
    files=[oldcp,newcp,oldroot/'formal/R1/run.json',sgroot/'formal/run.json',sgroot/'job.json',sgroot/'formal/solver_goal_complete.json',data,urdf,cache/'manifest.json',cache/'x.npy']
    for sp in ('train','val'):
        files.extend(cache/sp/f'boundary_{k}.npy' for k in ('x_index','sensor','q','grad'))
    inputs={}
    print('[preflight] hashing frozen checkpoints, original dataset and geometry seed files',flush=True)
    for f in files:
        require(f.is_file() and not f.is_symlink(),f'Missing or symbolic input {f}')
        st=f.stat();inputs[str(f)]={'sha256':sha(f),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    require(inputs[str(oldcp)]['sha256']==R1_SHA and inputs[str(newcp)]['sha256']==NEW_SHA,'Checkpoint SHA failed')
    require(inputs[str(data)]['sha256']==source['data_sha']==DATA_SHA,'Original dataset hash failed')
    # Check manifest hashes of actual geometry seed files, not merely existence.
    for f in files:
        if f.is_relative_to(cache) and f!=cache/'manifest.json':
            rel=str(f.relative_to(cache));expected=manifest['files'][rel]
            require(inputs[str(f)]['sha256']==expected['sha256'] and inputs[str(f)]['bytes']==expected['bytes'],f'Corrupt seed file: {rel}')
    # Runtime and geometry imports must match the model's frozen training job when present.
    for rel,dig in source['source'].items():
        if not rel.startswith('solver_goal/') and rel in code:require(code[rel]==dig,f'Training dependency changed: {rel}')
    with np.load(data,allow_pickle=False) as bank:
        x=np.asarray(bank['x'],np.float32)
        lo=np.asarray(bank['q_min'],np.float32);hi=np.asarray(bank['q_max'],np.float32)
        masks=np.asarray(bank['sensor_chain_masks'],np.float32)
        keys=(('x_min','x_max'),('y_min','y_max'),('z_min_bound','z_max_bound'))
        bounds=np.array([[float(bank[a]),float(bank[b])] for a,b in keys])
    require(np.array_equal(x,np.load(cache/'x.npy',allow_pickle=False)),'Seed x-index mismatch')
    require(lo.shape==hi.shape==(7,) and masks.shape==(8,7) and np.all(hi>lo),'Joint metadata invalid')
    require(np.isin(masks,[0.,1.]).all(),'Invalid sensor mask')
    points,attempts=select_points(x,bounds,p)
    require(not out.exists(),f'Output already exists: {out}')
    out.mkdir(parents=True)
    write(out/'points.json',{'points':points.tolist(),'seed':p['seed'],'bounds_m':bounds.tolist(),'draws':attempts})
    job={'format':FORMAT,'status':'PREPARED','cfg':p,'repo':str(repo),'out':str(out),'sgroot':str(sgroot),'r012':str(oldroot),'cache':str(cache),'data':str(data),'urdf':str(urdf),'lo':lo.tolist(),'hi':hi.tolist(),'masks':masks.tolist(),'inputs':inputs,'code':code,'points_sha256':sha(out/'points.json'),'checkpoints':{'R1':str(oldcp),'SOLVER_GOAL':str(newcp)},'checkpoint_sha256':{'R1':R1_SHA,'SOLVER_GOAL':NEW_SHA}}
    write(out/'job.json',job)
    print(f'[preflight] 512 off-grid x frozen; no model loaded; points_sha={job["points_sha256"]}',flush=True)
    return job
