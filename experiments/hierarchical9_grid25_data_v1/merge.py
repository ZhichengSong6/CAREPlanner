"""Streaming ZIP64 merge: bounded RAM; no old all-shards concatenate."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import numpy as np
from g25_common import (ROW_KEYS,require,sha,write,read,lock,point_ids,nshards,stage,metadata,verify_job)
from collect import validate_arrays


def stream_npz(path,shards,total_points,meta):
    """Concatenate first dimensions inside .npy ZIP members without a giant RAM array."""
    with np.load(shards[0],allow_pickle=False) as z:
        schema={k:(z[k].shape[1:],z[k].dtype) for k in ROW_KEYS}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as out:
        for key,(tail,dtype) in schema.items():
            count=0
            with out.open(key+'.npy','w',force_zip64=True) as f:
                np.lib.format.write_array_header_2_0(f,{'descr':np.lib.format.dtype_to_descr(dtype),
                      'fortran_order':False,'shape':(total_points,*tail)})
                for shard in shards:
                    with np.load(shard,allow_pickle=False) as z:a=np.ascontiguousarray(z[key])
                    require(a.shape[1:]==tail and a.dtype==dtype,'Shard merge schema changed')
                    f.write(memoryview(a).cast('B'));count+=len(a)
            require(count==total_points,'Missing/extra merged points')
        for key,a in meta.items():
            require(key not in schema,'Metadata collides with point data')
            with out.open(key+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(a),allow_pickle=False)


def archive_check(path,job,crc=True):
    """Check all member byte counts; optional full streaming decompression CRC."""
    with zipfile.ZipFile(path) as z:
        names=z.namelist();require(len(names)==len(set(names)),'Duplicate archive members')
        for key in ROW_KEYS:
            with z.open(key+'.npy') as f:
                version=np.lib.format.read_magic(f)
                require(version in ((1,0),(2,0)),'Unknown NPY version')
                shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f))
                require(shape[0]==job['cfg']['grid_points'] and not fortran and not dtype.hasobject,'Invalid merged shape')
                require(z.getinfo(key+'.npy').file_size==f.tell()+int(np.prod(shape))*dtype.itemsize,'Truncated merged member')
        if crc:require(z.testzip() is None,'Archive CRC error')
    with np.load(path,allow_pickle=False) as z:
        require(np.array_equal(z['x'],np.load(Path(job['out'])/'x.npy',allow_pickle=False)),'Merged coordinates changed')
        require(np.array_equal(z['k'],np.arange(job['cfg']['grid_points'])),'Grid ID order changed')
        for k,v in metadata(job).items():require(np.array_equal(z[k],v),f'Merged metadata differs: {k}')


def merge(job):
    verify_job(job);root=Path(job['out']);dest=root/'dataset'
    with lock(root/'locks/merge.lock') as acquired:
        if not acquired:print('[merge] another collector is finalizing',flush=True);return False
        if dest.exists():verify_dataset(job);return True
        missing=[i for i in range(nshards(job['cfg'])) if not (root/'shards'/f'{i:05d}'/'complete.json').is_file()]
        if missing:
            print(f'[NOT COMPLETE] {nshards(job["cfg"])-len(missing)}/{nshards(job["cfg"])} shards; no final dataset published',flush=True);return False
        all_stats=[];paths=[];points=np.load(root/'x.npy',mmap_mode='r')
        for sid in range(nshards(job['cfg'])):
            m=stage(job,sid);p=root/'shards'/f'{sid:05d}'/'data.npz';ids=point_ids(job['cfg'],sid)
            with np.load(p,allow_pickle=False) as z:validate_arrays({k:z[k] for k in ROW_KEYS},ids,points[ids],job)
            all_stats.append(m);paths.append(p)
        raw_bytes=0
        for path in paths:
            with zipfile.ZipFile(path) as z:raw_bytes+=sum(i.file_size for i in z.infolist())
        require(shutil.disk_usage(root).free>=raw_bytes+(1<<30),'Insufficient free disk for conservative merge bound')
        tmp=Path(tempfile.mkdtemp(prefix='.dataset.',dir=root))
        try:
            path=tmp/'visibility_grid25_fovonly.npz'
            stream_npz(path,paths,job['cfg']['grid_points'],metadata(job));archive_check(path,job)
            by=np.sum([m['by_sensor'] for m in all_stats],axis=0).tolist()
            unresolved=np.sum([m['unresolved_sensor_pairs'] for m in all_stats],axis=0).tolist()
            payload=dict(status='COMPLETE',format=job['cfg']['format'],points=job['cfg']['grid_points'],
                grid_shape=job['cfg']['grid_shape'],frame=job['cfg']['frame'],spacing_mm=25,
                reference_sampling=job['reference'],valid_q0_by_sensor=by,unresolved_x_sensor_by_sensor=unresolved,
                points_with_q0=sum(m['points_with_q0'] for m in all_stats),
                files={path.name:sha(path)},job_sha256=sha(root/'job.json'),
                shard_hashes={str(i):m['data_sha256'] for i,m in enumerate(all_stats)},
                scope='NEW_Q0_BANK_ONLY; FOV-only; no CASE data, training, V4 pairs, or success replay',
                training_ready=False,reason='Rebuild derived V4 pairs/cache/splits for this grid before latest training')
            write(tmp/'complete.json',payload);os.rename(tmp,dest)
        except BaseException:
            shutil.rmtree(tmp,ignore_errors=True);raise
        print(f'[DATASET COMPLETE] {dest}/visibility_grid25_fovonly.npz',flush=True)
        return True


def verify_dataset(job):
    verify_job(job);root=Path(job['out']);p=root/'dataset';m=read(p/'complete.json')
    require(m['status']=='COMPLETE' and m['points']==job['cfg']['grid_points'] and m['job_sha256']==sha(root/'job.json'),'Wrong final dataset identity')
    for name,digest in m['files'].items():require(sha(p/name)==digest,'Final dataset hash differs')
    for sid in range(nshards(job['cfg'])):
        require(stage(job,sid)['data_sha256']==m['shard_hashes'][str(sid)],'Final shard set changed')
    archive_check(p/'visibility_grid25_fovonly.npz',job)
    print('[VERIFIED] complete grid, shard and merged hashes/CRC; original sampling semantics; no training',flush=True)


def summary(job):
    root=Path(job['out']);stats=[]
    for i in range(nshards(job['cfg'])):
        if (root/'shards'/f'{i:05d}'/'complete.json').exists():stats.append(stage(job,i,False))
    print(f'Grid: {job["cfg"]["grid_shape"]}, points={job["cfg"]["grid_points"]}, spacing=25 mm; frame=base_link')
    print(f'Completed shards: {len(stats)}/{nshards(job["cfg"])}; points={sum(x["points"] for x in stats)}')
    print('Inherited sampling:',{k:job['reference'][k] for k in ('q_init_per_p','optimize_top_k_per_sensor','q0_per_sensor','zero_level_optimizer','max_iter','epsilon')})
    if stats:
        print('Valid Q0 by sensor:',np.sum([x['by_sensor'] for x in stats],axis=0).tolist())
        print('Unresolved boundary search by sensor:',np.sum([x['unresolved_sensor_pairs'] for x in stats],axis=0).tolist())
    final=root/'dataset/complete.json';print('Dataset status:',read(final)['status'] if final.exists() else 'INCOMPLETE')
    print('Scope: new original-format Q0 bank. V4 pairs/training caches must be rebuilt; no model trained.')
