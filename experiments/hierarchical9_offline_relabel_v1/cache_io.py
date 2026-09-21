"""Numeric-only immutable bank/query caches with atomic writes and checksums."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import numpy as np
from repo_oracle import DEFAULT_JOINTS, DEFAULT_SENSORS, FOV, sha256_file

FORMAT = 'careplanner_offline_continuous_fov_labels_v1'
REQUIRED = ['x','q','k','valid_fov','sensor_chain_masks','q_min','q_max',
            'joint_names','sensor_frames',*FOV.keys()]
OPTIONAL = ['grid_shape','x_min','x_max','y_min','y_max','z_min_bound','z_max_bound',
            'q0_per_sensor','epsilon','seed']


def clean_json(x):
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean_json(v) for v in x]
    return x


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    try:
        with temp.open('w',encoding='utf-8') as f:
            json.dump(clean_json(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def write_npz(path, arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    try:
        with temp.open('wb') as f:
            np.savez_compressed(f,**arrays); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def code_hashes():
    here = Path(__file__).resolve().parent
    return {p.name:sha256_file(p) for p in sorted(here.glob('*.py'))}


def import_bank(source, destination):
    """Extract selected numeric .npy members ONCE; workers memory-map large banks.

    No np.load(large_compressed_npz) per worker, no pickle, no modifications to
    the original archive. Source archive and extracted arrays are hashed once.
    """
    source, destination = Path(source).resolve(),Path(destination).resolve()
    stat=source.stat()
    if destination.exists():
        manifest=read_json(destination/'bank_manifest.json')
        expected=dict(path=str(source),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        if manifest.get('status')!='COMPLETE' or manifest.get('source_stat')!=expected:
            raise RuntimeError('Existing bank cache is incomplete/different; choose a NEW cache directory')
        BankCache(destination)  # Header, shapes and immutable-file stat checks.
        return manifest
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=destination.name+'.building.',dir=destination.parent))
    try:
        hashes, member_stats = {}, {}
        with zipfile.ZipFile(source) as archive:
            names=archive.namelist()
            for key in REQUIRED:
                if names.count(key+'.npy') != 1:
                    raise ValueError(f'Required numeric NPZ member missing/duplicated: {key}')
            keys=REQUIRED+[k for k in OPTIONAL if k+'.npy' in names]
            total=sum(archive.getinfo(k+'.npy').file_size for k in keys)
            if shutil.disk_usage(destination.parent).free < total+(128<<20):
                raise OSError(f'Insufficient disk: extraction needs at least {total} bytes plus headroom')
            for key in keys:
                dst=temp/(key+'.npy'); h=hashlib.sha256()
                with archive.open(key+'.npy') as src, dst.open('wb') as out:
                    for block in iter(lambda:src.read(8<<20),b''):
                        out.write(block);h.update(block)
                a=np.load(dst,mmap_mode='r',allow_pickle=False)
                if a.dtype.hasobject: raise ValueError('Object arrays are not accepted')
                hashes[key]=h.hexdigest()
                st=dst.stat();member_stats[key]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
                del a
        manifest=dict(format=FORMAT,status='COMPLETE',source_stat=dict(path=str(source),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),
                      source_sha256=sha256_file(source),array_sha256=hashes,array_stat=member_stats)
        write_json(temp/'bank_manifest.json',manifest)
        BankCache(temp)
        os.replace(temp,destination)
        return manifest
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True)
        raise


class BankCache:
    def __init__(self, root):
        self.root=Path(root).resolve()
        self.manifest=read_json(self.root/'bank_manifest.json')
        if self.manifest.get('status')!='COMPLETE': raise ValueError('Incomplete bank cache')
        self.arrays={}
        for key in REQUIRED:
            path=self.root/(key+'.npy');st=path.stat()
            if dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)!=self.manifest['array_stat'][key]:
                raise ValueError(f'Bank cache modified: {key}; never edit a frozen cache')
            self.arrays[key]=np.load(path,mmap_mode='r',allow_pickle=False)
        self.x,self.q,self.valid=self.arrays['x'],self.arrays['q'],self.arrays['valid_fov']
        self.masks=np.array(self.arrays['sensor_chain_masks'],dtype=np.float32)
        self.lo=np.array(self.arrays['q_min'],dtype=np.float64)
        self.hi=np.array(self.arrays['q_max'],dtype=np.float64)
        self.joints=self.arrays['joint_names'].astype(str).tolist()
        self.sensors=self.arrays['sensor_frames'].astype(str).tolist()
        if self.x.dtype!=np.float32 or self.q.dtype!=np.float32 or self.valid.dtype!=np.bool_:
            raise ValueError('Expected original numeric x/q=float32 and valid_fov=bool; do not silently convert a different dataset')
        p=len(self.x)
        if self.x.shape!=(p,3) or self.q.ndim!=4 or self.q.shape[0]!=p or self.q.shape[2:]!=(7,8):
            raise ValueError('Expected x[P,3], q[P,K,7,8]')
        if self.valid.shape!=(p,self.q.shape[1],8) or self.masks.shape!=(8,7):
            raise ValueError('valid_fov/mask shape mismatch')
        if self.arrays['k'].shape!=(p,) or self.lo.shape!=(7,) or self.hi.shape!=(7,):
            raise ValueError('k/limits shape mismatch')
        if not np.isfinite(self.x).all() or not np.isfinite(self.lo).all() or not np.isfinite(self.hi).all() or not (self.lo<self.hi).all():
            raise ValueError('Nonfinite coordinates or invalid limits')
        if not np.isin(self.masks,[0.,1.]).all(): raise ValueError('Nonbinary masks')
        if self.joints!=DEFAULT_JOINTS or self.sensors!=DEFAULT_SENSORS:
            raise ValueError('Unexpected joint/sensor ordering')
        for key,value in FOV.items():
            if abs(float(self.arrays[key])-value)>1e-6:
                raise ValueError(f'Frozen FOV mismatch: {key}')

    def sensor_bank(self, xi, s):
        keep=np.asarray(self.valid[int(xi),:,int(s)],dtype=bool)
        rows=np.asarray(self.q[int(xi),:,:,int(s)],dtype=np.float32)[keep]
        if not np.isfinite(rows).all():
            raise ValueError(f'valid_fov contains nonfinite q0: x_index={xi}, sensor={s}')
        return rows

    def original_split(self,val_count=1000,seed=0):
        # Preserve original spatial split EXACTLY. Do not re-split after label failures.
        valid_x=np.any(self.valid,axis=(1,2))
        ids=np.flatnonzero(valid_x)
        rng=np.random.default_rng(seed);rng.shuffle(ids)
        n=min(val_count,max(1,len(ids)//10))
        if n<1 or len(ids)<=n: raise ValueError('Not enough valid x for the original split')
        return ids[n:].copy(),ids[:n].copy()


def prepare_queries(bank:BankCache,out,train_x=8,val_x=2,uniform_per_x=2,near_per_x=1,
                    near_std=.05,seed=260921,split_seed=0,val_count=1000,shard_size=4):
    out=Path(out).resolve()
    if out.exists(): raise FileExistsError(f'No overwrite: {out}')
    if min(train_x,val_x,uniform_per_x,shard_size)<1 or near_per_x<0 or near_std<=0:
        raise ValueError('Invalid query sampling counts')
    train_pool,val_pool=bank.original_split(val_count,split_seed)
    if train_x>len(train_pool) or val_x>len(val_pool):
        raise ValueError('Requested more unique x than the preserved split contains')
    rng=np.random.default_rng(seed)
    ids=[];qs=[];splits=[];groups=[];seed_sensor=[];seed_slot=[];skips=[]
    for split,pool,nx in [(0,train_pool,train_x),(1,val_pool,val_x)]:
        selected=np.sort(rng.choice(pool,nx,replace=False))
        for xi in selected:
            for _ in range(uniform_per_x):
                ids.append(int(xi));qs.append(rng.uniform(bank.lo,bank.hi).astype(np.float32))
                splits.append(split);groups.append(0);seed_sensor.append(-1);seed_slot.append(-1)
            slots=np.argwhere(np.asarray(bank.valid[xi],bool))
            for near_i in range(near_per_x):
                kk,s=slots[rng.integers(len(slots))]
                anchor=np.array(bank.q[xi,kk,:,s],dtype=np.float64)
                if not np.isfinite(anchor).all() or not ((anchor>=bank.lo)&(anchor<=bank.hi)).all():
                    skips.append(dict(x_index=int(xi),near_index=near_i,reason='invalid_or_out_of_bounds_anchor'));continue
                q=None
                for _ in range(32):
                    candidate=anchor+rng.normal(0,near_std,7)*bank.masks[s]
                    if ((candidate>=bank.lo)&(candidate<=bank.hi)).all():
                        q=candidate.astype(np.float32);break
                if q is None:
                    skips.append(dict(x_index=int(xi),near_index=near_i,reason='out_of_bounds_near_query'));continue
                ids.append(int(xi));qs.append(q);splits.append(split);groups.append(1)
                seed_sensor.append(int(s));seed_slot.append(int(kk))
    if not ids: raise ValueError('No queries')
    # These are near-bank samples, NOT verified local-boundary benchmark starts.
    arrays=dict(query_id=np.arange(len(ids),dtype=np.int64),x_index=np.asarray(ids,dtype=np.int64),
                q_query=np.asarray(qs,dtype=np.float32),split=np.asarray(splits,dtype=np.uint8),
                query_group=np.asarray(groups,dtype=np.uint8),source_sensor=np.asarray(seed_sensor,dtype=np.int8),
                source_bank_slot=np.asarray(seed_slot,dtype=np.int32))
    out.mkdir(parents=True)
    write_npz(out/'queries.npz',arrays)
    write_npz(out/'spatial_splits.npz',dict(train_x_indices=train_pool,val_x_indices=val_pool))
    manifest=dict(format=FORMAT,status='QUERIES_READY',bank_cache=str(bank.root),
                  bank_manifest_sha256=sha256_file(bank.root/'bank_manifest.json'),
                  original_data_sha256=bank.manifest['source_sha256'],queries_sha256=sha256_file(out/'queries.npz'),
                  spatial_splits_sha256=sha256_file(out/'spatial_splits.npz'),query_count=len(ids),
                  shard_size=shard_size,shard_count=(len(ids)+shard_size-1)//shard_size,
                  sampling=dict(train_x=train_x,val_x=val_x,uniform_per_x=uniform_per_x,near_per_x=near_per_x,
                                near_std_rad=near_std,seed=seed,split_seed=split_seed,val_count=val_count),
                  query_groups={'0':'uniform_in_joint_box_inside_and_outside','1':'near_bank_not_verified_local_boundary'},
                  split_names={'0':'train','1':'development_val_not_fresh_final_holdout'},
                  exclusions=skips,code_sha256=code_hashes(),
                  limitations=['FOV only; no LOS/collision/trajectory/actual-seen',
                               'Finite frozen query pool; not the historical endless Cartesian stream',
                               'No changes to weights, runtime, training, FOV or safety thresholds'])
    write_json(out/'manifest.json',manifest)
    return manifest


def open_job(out):
    out=Path(out).resolve();m=read_json(out/'manifest.json')
    if m.get('format')!=FORMAT: raise ValueError('Unknown cache format')
    for name,key in [('queries.npz','queries_sha256'),('spatial_splits.npz','spatial_splits_sha256')]:
        if sha256_file(out/name)!=m[key]: raise ValueError(f'Changed frozen file: {name}')
    if code_hashes()!=m['code_sha256']:
        raise ValueError('Label source changed after prepare; use a new output directory')
    bank=BankCache(m['bank_cache'])
    if sha256_file(bank.root/'bank_manifest.json')!=m['bank_manifest_sha256']:
        raise ValueError('Bank identity mismatch')
    with np.load(out/'queries.npz',allow_pickle=False) as z:
        queries={k:z[k] for k in z.files}
    with np.load(out/'spatial_splits.npz',allow_pickle=False) as z:
        train=set(z['train_x_indices'].tolist());val=set(z['val_x_indices'].tolist())
    if train&val: raise ValueError('Spatial split leakage')
    for xi,split in zip(queries['x_index'],queries['split']):
        if int(xi) not in (train if split==0 else val): raise ValueError('Query split mismatch')
    return out,m,bank,queries
