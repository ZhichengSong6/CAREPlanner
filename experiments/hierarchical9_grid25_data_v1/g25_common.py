"""Frozen inputs and shared storage. No model, task case, or failure log is read."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE_DIR = 'src/care_visibility_cdf/scripts'
PINNED_BLOBS = {
 'data_generator.py':'0195b7016a2668b4d247e75f25ef2a7845e719b0',
 'extract_visibility_zero_level_sets.py':'451e4a31279b9dd9ee89ec59e51f0915e08f07d3',
 'validate_visibility_oracle.py':'d1d9437a3dee67cdb73209fd41fda693590b78bb',
 'check_visibility_self_occlusion.py':'541bb6b286fcbf694a69269c461c1d3c87d87901'}
URDF_REL = 'src/arm_description/urdf/Arm.urdf'
URDF_BLOB = '64eebec0642512cb17a277089d92c6431e40cbab'
SAMPLER_KEYS = ('q_init_per_p','optimize_top_k_per_sensor','q0_per_sensor',
                'zero_level_optimizer','max_iter','lr','lbfgs_lr','epsilon','seed')
# Store canonical arrays, not redundant transposed debugging copies.
ROW_KEYS = ('x','q','k','valid_fov','g','sensor_margins','active_planes',
            'num_sensor_q0','num_sensor_q0_before_downsample','valid_point_fov')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value, replace=False):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name('.'+p.name+f'.{os.getpid()}.tmp')
    with tmp.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:
        if replace: os.replace(tmp,p)
        else: os.link(tmp,p)
    finally: tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def lock(path, blocking=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        acquired=True
        try: fcntl.flock(f,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError: acquired=False
        try: yield acquired
        finally:
            if acquired: fcntl.flock(f,fcntl.LOCK_UN)


def grid(cfg):
    axes=[np.arange(a,b+1,cfg['spacing_mm'],dtype=np.int64) for a,b in cfg['workspace_mm']]
    require([len(a) for a in axes]==cfg['grid_shape'],'Grid shape mismatch')
    mm=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    require(len(mm)==cfg['grid_points'],'Grid count mismatch')
    return (mm.astype(np.float64)/1000).astype(np.float32)


def point_ids(cfg, sid):
    n=cfg['grid_points'];size=cfg['points_per_shard']
    require(0<=sid<(n+size-1)//size,'Invalid shard ID')
    return np.arange(sid*size,min((sid+1)*size,n),dtype=np.int64)


def nshards(cfg): return (cfg['grid_points']+cfg['points_per_shard']-1)//cfg['points_per_shard']


def reference_metadata(path,cfg):
    """Read metadata only: never decompress old q banks or transfer their labels."""
    with np.load(path,allow_pickle=False) as z:
        keys=SAMPLER_KEYS+('joint_names','sensor_frames','q_min','q_max','sensor_chain_masks','has_occlusion_labels')+tuple(cfg['fov'])
        missing=set(keys)-set(z.files); require(not missing,f'Reference metadata missing: {sorted(missing)}; do not guess sampling defaults')
        out={k:z[k].tolist() for k in keys}
        out['old_grid_shape']=z['grid_shape'].tolist()
    for name in ('joint_names','sensor_frames'): require(out[name]==cfg[name],f'Reference {name} differs')
    for name in ('q_min','q_max'):
        require(np.allclose(out[name],cfg[name],rtol=0,atol=1e-6),f'Reference {name} differs')
    for name,value in cfg['fov'].items():
        require(abs(float(out[name])-value)<1e-6,f'Reference FOV {name} differs')
    require(out['has_occlusion_labels'] is False,'Reference must be FOV-only')
    for name in ('q_init_per_p','optimize_top_k_per_sensor','q0_per_sensor','max_iter'):
        require(isinstance(out[name],int) and out[name]>0,f'Invalid sampler {name}')
    require(out['optimize_top_k_per_sensor']<=out['q_init_per_p'],'Invalid optimizer candidate count')
    require(out['zero_level_optimizer'] in ('adam','lbfgs'),'Invalid optimizer')
    for name in ('epsilon','lr','lbfgs_lr'): require(np.isfinite(out[name]) and out[name]>0,f'Invalid {name}')
    require(isinstance(out['seed'],int) and out['seed']>=0,'Invalid seed')
    require(out['sensor_chain_masks']==[[int(i<n) for i in range(7)] for n in (2,2,3,3,4,4,7,7)],'Reference masks differ')
    return out


def check_urdf(path,cfg,expected_masks):
    root=ET.parse(path).getroot(); joints={j.attrib['name']:j for j in root.findall('joint')}
    parent={j.find('child').attrib['link']:j for j in joints.values()}
    for i,name in enumerate(cfg['joint_names']):
        j=joints[name];lim=j.find('limit');require(j.attrib['type']=='revolute',f'Unexpected joint type: {name}')
        require(abs(float(lim.attrib['lower'])-cfg['q_min'][i])<1e-6 and abs(float(lim.attrib['upper'])-cfg['q_max'][i])<1e-6,f'URDF limits differ: {name}')
    masks=[]
    for frame in cfg['sensor_frames']:
        seen=set();link=frame;names=set()
        while link!=cfg['frame']:
            require(link not in seen and link in parent,f'Cannot trace {frame} to {cfg["frame"]}')
            seen.add(link);j=parent[link];names.add(j.attrib['name']);link=j.find('parent').attrib['link']
        masks.append([int(n in names) for n in cfg['joint_names']])
    require(masks==expected_masks,'URDF sensor ancestor masks differ')


def load_generator(job):
    directory=Path(job['deps'])/SOURCE_DIR
    sys.path.insert(0,str(directory))
    for name in ('validate_visibility_oracle','check_visibility_self_occlusion','extract_visibility_zero_level_sets','data_generator'):
        if name in sys.modules:
            f=getattr(sys.modules[name],'__file__',None)
            require(f and Path(f).resolve().parent==directory.resolve(),f'Conflicting imported module: {name}')
    return importlib.import_module('data_generator')


def verify_job(job):
    require(job['cfg']==read(HERE/'protocol.json'),'Protocol changed')
    require(job['code_hashes']=={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh','.json')},'Frozen collector code changed')
    for rel,digest in job['deps_sha256'].items(): require(sha(Path(job['deps'])/rel)==digest,f'Dependency changed: {rel}')
    out=Path(job['out']);require(sha(out/'x.npy')==job['x_sha256'],'Frozen grid changed')


def args_for(job):
    from types import SimpleNamespace
    c=job['cfg'];r=job['reference'];a={k:r[k] for k in SAMPLER_KEYS}
    a.update(c['fov']);a.update(urdf=str(Path(job['deps'])/URDF_REL),base_frame=c['frame'],
        joint_names=c['joint_names'],sensor_frames=c['sensor_frames'],occlusion_urdf='',
        occlusion_backend='none',grid=c['grid_shape'],x_min=-.8,x_max=.8,y_min=-.8,y_max=.8,
        z_min_bound=0.,z_max_bound=1.,device='cuda')
    return SimpleNamespace(**a)


def metadata(job):
    c=job['cfg'];r=job['reference'];m={k:np.asarray(r[k]) for k in SAMPLER_KEYS}
    for k in ('joint_names','sensor_frames'):m[k]=np.asarray(c[k])
    for k in ('q_min','q_max'):m[k]=np.asarray(c[k],np.float32)
    m['sensor_chain_masks']=np.asarray(r['sensor_chain_masks'],np.float32)
    for k,v in c['fov'].items():m[k]=np.asarray(v,np.float32)
    for k,v in dict(x_min=-.8,x_max=.8,y_min=-.8,y_max=.8,z_min_bound=0.,z_max_bound=1.).items():m[k]=np.asarray(v,np.float32)
    m.update(grid_shape=np.asarray(c['grid_shape'],np.int32),spacing_m=np.full(3,.025,np.float32),
       base_frame=np.asarray(c['frame']),format=np.asarray(c['format']),
       has_occlusion_labels=np.asarray(False),occlusion_backend=np.asarray('none'),occlusion_urdf=np.asarray(''),
       point_rng=np.asarray(c['point_rng']),p_num_shards=np.asarray(nshards(c),np.int32),p_shard_id=np.asarray(-1,np.int32),
       p_seed=np.asarray(0,np.int32))
    return m


def stage(job,sid,verify_hash=True):
    p=Path(job['out'])/'shards'/f'{sid:05d}';m=read(p/'complete.json')
    require(m.get('status')=='COMPLETE' and m.get('job_sha256')==sha(Path(job['out'])/'job.json') and m.get('shard_id')==sid,'Wrong shard identity')
    require(m['point_ids']==point_ids(job['cfg'],sid).tolist(),'Shard coverage mismatch')
    if verify_hash: require(sha(p/'data.npz')==m['data_sha256'],f'Corrupt shard {sid}')
    return m
