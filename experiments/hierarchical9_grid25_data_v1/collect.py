"""Resumeable GPU collection using the unchanged historical per-point generator."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from g25_common import (ROW_KEYS,require,sha,write,lock,point_ids,nshards,load_generator,
                        verify_job,args_for,metadata,stage)


def validate_arrays(data,ids,points,job):
    k=job['reference']['q0_per_sensor'];n=len(ids);v=data['valid_fov'];q=data['q']
    require(q.shape==(n,k,7,8) and v.shape==(n,k,8) and v.dtype==np.bool_,'Bad Q0 layout')
    require(np.array_equal(data['x'],points) and np.array_equal(data['k'],ids),'Wrong grid coordinates/IDs')
    require(data['g'].shape==(n,k,8) and data['sensor_margins'].shape==(n,k,8,8) and data['active_planes'].shape==(n,k,8,8),'Bad diagnostics layout')
    require(np.isfinite(data['sensor_margins'][v]).all(),'Nonfinite retained sensor margins')
    ap=data['active_planes'][v];require(np.all((ap>=0)&(ap<6)),'Invalid retained active face')
    finite=np.isfinite(q).all(axis=2);require(np.array_equal(finite,v),'Finite Q0 and valid mask disagree')
    require(np.isposinf(q.transpose(0,1,3,2)[~v]).all(),'Invalid Q0 must stay +inf')
    lo=np.asarray(job['cfg']['q_min'],np.float32);hi=np.asarray(job['cfg']['q_max'],np.float32)
    flat=q.transpose(0,1,3,2)[v]
    require(np.all(flat>lo) and np.all(flat<hi),'Stored Q0 violates original strict limits')
    require(np.isfinite(data['g'][v]).all() and np.all(np.abs(data['g'][v])<=job['reference']['epsilon']+1e-6),'Stored Q0 violates inherited epsilon')
    require(np.array_equal(v.sum(axis=1),data['num_sensor_q0']),'Incorrect per-sensor counts')
    require(np.array_equal(v.any(axis=(1,2)),data['valid_point_fov']),'Incorrect point validity')
    require(np.all(data['num_sensor_q0_before_downsample']>=data['num_sensor_q0']),'Bad downsample counts')
    for s in range(8):
        sv=v[:,:,s]
        if not sv.any():continue
        margin=data['sensor_margins'][:,:,s,s][sv]-job['cfg']['fov']['delta']
        require(np.allclose(margin,data['g'][:,:,s][sv],rtol=0,atol=2e-6),'Inconsistent FOV sign/margin convention')
    return dict(points=n,points_with_q0=int(v.any(axis=(1,2)).sum()),
        valid_q0=int(v.sum()),by_sensor=v.sum(axis=(0,1)).astype(int).tolist(),
        unresolved_sensor_pairs=(~v.any(axis=1)).sum(axis=0).astype(int).tolist(),
        max_abs_g_m=float(np.abs(data['g'][v]).max()) if v.any() else None)


def collect_shard(job,sid,gen,specs,lo,hi,limits,device):
    import torch
    ids=point_ids(job['cfg'],sid);points=np.load(Path(job['out'])/'x.npy',mmap_mode='r')[ids]
    a=args_for(job);n=len(ids);data=gen.allocate_yiming_style_arrays(n,a.q0_per_sensor,7,8)
    data['x'][:]=points;data['k'][:]=ids
    t0=time.monotonic();max_check=0.
    for i,(pid,x) in enumerate(zip(ids,points)):
        rng=np.random.default_rng(np.random.SeedSequence([a.seed,int(pid)]))
        qi=gen.sample_q_batch(rng,limits,a.joint_names,a.q_init_per_p)
        results=gen.generate_for_one_point(x,qi,specs,lo,hi,None,a,device)
        # Keep historical farthest-point downsampling formula, but use GLOBAL
        # point ID instead of worker-local index. No label or optimizer changes.
        sa=copy.copy(a);sa.seed=a.seed+1000003*(int(pid)-i)
        for s,r in enumerate(results):
            if r is None:continue
            gen.store_sensor_block(data,i,int(pid),x,s,r['q0'],r['g0'],r['sensor_margins0'],r['active_planes0'],r['occluded'],sa)
            count=int(data['num_sensor_q0'][i,s])
            # Recheck retained Q0 with original FOV; never introduce new epsilon.
            if count:
                qs=torch.tensor(data['q'][i,:count,:,s],device=device)
                with torch.no_grad():
                    _,margin,_,_=gen.visibility_g_batch(torch.tensor(x,device=device),qs,specs,
                       a.horizontal_fov_deg,a.vertical_fov_deg,a.z_min,a.z_max,a.delta)
                gs=(margin[:,s]-a.delta).detach().cpu().numpy()
                err=float(np.max(np.abs(gs-data['g'][i,:count,s])));max_check=max(max_check,err)
                require(err<=2e-6 and np.all(np.abs(gs)<=a.epsilon+2e-6),'Independent retained-Q0 FOV check failed')
        if (i+1)%8==0 or i+1==n:
            print(f'[collect shard={sid}] {i+1}/{n} points; elapsed={time.monotonic()-t0:.1f}s',flush=True)
    data['valid_point_fov']=data['valid_fov'].any(axis=(1,2))
    compact={key:data[key] for key in ROW_KEYS};stats=validate_arrays(compact,ids,points,job)
    root=Path(job['out']);tmp=Path(tempfile.mkdtemp(prefix=f'.{sid:05d}.',dir=root/'shards'))
    try:
        np.savez_compressed(tmp/'data.npz',**compact)
        # Reopen after compression; incomplete files are never published.
        with np.load(tmp/'data.npz',allow_pickle=False) as z:
            validate_arrays({k:z[k] for k in ROW_KEYS},ids,points,job)
        write(tmp/'complete.json',dict(status='COMPLETE',shard_id=sid,point_ids=ids.tolist(),
            job_sha256=sha(root/'job.json'),data_sha256=sha(tmp/'data.npz'),
            elapsed_seconds=time.monotonic()-t0,fov_recheck_max_abs_m=max_check,**stats))
        os.rename(tmp,root/'shards'/f'{sid:05d}')
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    print(f'[shard COMPLETE] id={sid} valid_q0={stats["valid_q0"]}',flush=True)


def worker(job,slot):
    import torch
    verify_job(job);require(os.environ.get('SLURM_JOB_ID'),'Collection requires a Slurm allocation')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Worker must see exactly one assigned GPU')
    torch.cuda.set_device(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    free,total=torch.cuda.mem_get_info()
    require(free>=job['cfg']['min_free_gpu_gib']*(1<<30),'Too little global free GPU memory; no fallback or process killing')
    gen=load_generator(job);a=args_for(job);dev=torch.device('cuda:0')
    robot=gen.URDF.from_xml_file(a.urdf);limits=gen.get_joint_limits(robot,a.joint_names)
    lo=gen.torch_tensor(np.asarray(job['cfg']['q_min'],np.float32),dev)
    hi=gen.torch_tensor(np.asarray(job['cfg']['q_max'],np.float32),dev)
    specs=gen.prepare_chain_specs(robot,a.base_frame,a.sensor_frames,a.joint_names,dev)
    require(np.array_equal(gen.SENSOR_CHAIN_MASKS,np.asarray(job['reference']['sensor_chain_masks'])),'Upstream masks changed')
    total=nshards(job['cfg']);root=Path(job['out']);done=0
    # Every worker visits the whole pool in a different order. More GPUs improve
    # throughput only; completion/sampling never depend on launch world size.
    order=np.random.default_rng(np.random.SeedSequence([731025,slot])).permutation(total)
    for sid in map(int,order):
        dest=root/'shards'/f'{sid:05d}'
        if dest.exists():stage(job,sid,False);continue
        with lock(root/'locks'/f'shard_{sid:05d}.lock') as acquired:
            if not acquired:continue
            if dest.exists():stage(job,sid);continue
            collect_shard(job,sid,gen,specs,lo,hi,limits,dev);done+=1
    print(f'[worker COMPLETE] slot={slot} new_shards={done}; other jobs may still be collecting',flush=True)
