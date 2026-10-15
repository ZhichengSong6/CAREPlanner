#!/usr/bin/env python3
"""One four-GPU evaluation job: geometry -> input freeze -> paired solver -> report."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
from common import (HERE,ARMS,R1_SHA,NEW_SHA,require,read,sha,load,write,write_lines,jsonl,
                    verify_sources,stage_complete,preflight)


def freeze(job):
    out=Path(job['out']);verify_sources(job)
    if (out/'inputs_frozen.json').exists():return stage_complete(out,'inputs_frozen.json')
    require(sha(out/'points.json')==job['points_sha256'],'Points changed')
    points=read(out/'points.json')['points'];coverage=[];rows=[];shard_sha={}
    for rank in range(job['cfg']['world']):
        folder=out/'geometry'/f'rank{rank}';stage_complete(folder,'complete.json')
        shard_sha[str(folder.relative_to(out)/'complete.json')]=sha(folder/'complete.json')
        cov=read(folder/'coverage.json')['rows']
        require(all(c['point_id']%job['cfg']['world']==rank for c in cov),'Geometry shard assignment changed')
        coverage+=cov;rows+=list(jsonl(folder/'starts.jsonl'))
    keys=[(c['point_id'],c['sensor']) for c in coverage]
    require(len(keys)==job['cfg']['points']*8 and len(set(keys))==len(keys),'Missing/duplicate geometry pairs')
    require(set(keys)=={(pid,s) for pid in range(job['cfg']['points']) for s in range(8)},'Invalid coverage identity')
    lookup={(c['point_id'],c['sensor']):c for c in coverage}
    count={k:{'local':0,'uniform':0} for k in keys};unique=set();lo=np.asarray(job['lo']);hi=np.asarray(job['hi'])
    for r in rows:
        pid,s=r['point_id'],r['sensor'];c=r['cohort']
        require(r['x']==points[pid] and (pid,s) in lookup and c in ('local','uniform'),'Unknown start')
        require(lookup[pid,s]['status']=='WITNESS_CONFIRMED','Start lacks witness')
        require(np.isfinite(r['initial_g_m']) and r['initial_g_m'] < -job['cfg']['side_margin_m'],'Not FOV-outside start')
        require(np.isfinite(r['witness_g_m']) and r['witness_g_m']>job['cfg']['side_margin_m'],'Invalid visible witness')
        for name in ('q_init','witness_q'):
            q=np.asarray(r[name]);require(q.shape==(7,) and np.isfinite(q).all() and (q>=lo).all() and (q<=hi).all(),'Invalid q')
        if c=='local':
            radius=r['radius_rad'];q0=np.asarray(r['q_boundary']);normal=np.asarray(r['normal'])
            require(radius in job['cfg']['local_radii_rad'] and abs(np.linalg.norm(normal)-1)<1e-5,'Invalid local radius/normal')
            require(np.max(abs(normal*(1-np.asarray(job['masks'][s]))))<1e-7,'Inactive local normal')
            require(np.allclose(r['q_init'],q0-radius*normal,rtol=0,atol=3e-6) and
                    np.allclose(r['witness_q'],q0+radius*normal,rtol=0,atol=3e-6),'Wrong same-anchor pair')
            slot=radius
        else:slot=r['sample_slot'];require(slot in range(job['cfg']['uniform_per_pair']),'Invalid uniform slot')
        key=(pid,s,c,slot);require(key not in unique,'Duplicated start');unique.add(key);count[pid,s][c]+=1
    for key,c in lookup.items():
        require(count[key]['uniform']==c['uniform_count'] and count[key]['local']==c['local_count'],'Coverage/start counts disagree')
        require(c['uniform_count']<=4 and c['local_count']<=3,'Too many tests per pair')
    require(rows,'No independently qualified test cases; no network evaluation')
    rows.sort(key=lambda r:(r['point_id'],r['sensor'],r['cohort'],r.get('sample_slot',r.get('radius_rad'))))
    rows=[dict(r,case_id=i) for i,r in enumerate(rows)]
    coverage.sort(key=lambda c:(c['point_id'],c['sensor']))
    # Source files may already exist only after a crash between final writes.
    # Reconstruct identical content; never silently overwrite an existing plan.
    for path,content in [(out/'starts.jsonl',rows)]:
        if path.exists():require(list(jsonl(path))==content,'Existing frozen start file differs')
        else:write_lines(path,content)
    payload={'rows':coverage,'interpretation':'UNRESOLVED_WITNESS is bounded-search failure, not infeasible.'}
    if (out/'coverage.json').exists():require(read(out/'coverage.json')==payload,'Coverage changed')
    else:write(out/'coverage.json',payload)
    write(out/'inputs_frozen.json',{'status':'COMPLETE','case_count':len(rows),'point_count':len(points),
        'models_loaded_during_generation':False,'model_independent':True,
        'files':{'points.json':sha(out/'points.json'),'starts.jsonl':sha(out/'starts.jsonl'),'coverage.json':sha(out/'coverage.json'),**shard_sha}})
    print(f'[FROZEN] x={len(points)} starts={len(rows)} sha={sha(out/"starts.jsonl")}; model loading begins only after this gate',flush=True)


def parameters_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate(job,rank):
    import torch
    out=Path(job['out']);verify_sources(job);stage_complete(out,'inputs_frozen.json')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Each evaluation rank needs exactly one visible GPU')
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');repo=Path(job['repo'])
    scripts=repo/'src/care_visibility_cdf/scripts';sys.path.insert(0,str(scripts))
    arch=load('offgrid_frozen_arch',repo/'experiments/hierarchical9_scratch50k_r012_v1/model.py')
    pa=load('offgrid_legacy_probe',repo/'experiments/hierarchical9_boundary_audit_v1/runtime_probe.py')
    oa=load('offgrid_runtime_oracle',repo/'experiments/hierarchical9_boundary_audit_v1/oracle.py')
    ro=load('offgrid_joint_order',repo/'experiments/hierarchical9_offline_relabel_v1/repo_oracle.py')
    from hierarchical_visibility_cdf_model import HierarchicalSensorView
    oracle=oa.SensorOracle(Path(job['urdf']),device,ro.DEFAULT_JOINTS,ro.DEFAULT_SENSORS)
    lo=torch.tensor(job['lo'],device=device);hi=torch.tensor(job['hi'],device=device);masks=torch.tensor(job['masks'],device=device)
    models={};probes={};model_hashes={}
    trained_job=read(Path(job['sgroot'])/'job.json')
    for name in ARMS:
        checkpoint=Path(job['checkpoints'][name]);require(sha(checkpoint)==job['checkpoint_sha256'][name],'Checkpoint changed')
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        require(cp.get('step')==50000 and cp.get('arm')=='R1' and cp.get('initialization')=='random_from_scratch','Wrong checkpoint lineage')
        require(cp.get('training_stream_sha256')==trained_job['base_stream'],'Checkpoint sample stream differs')
        require(cp.get('metadata',{}).get('urdf_sha256')==job['inputs'][job['urdf']]['sha256'],'Checkpoint URDF differs')
        if name=='SOLVER_GOAL':require(cp.get('metadata',{}).get('solver_goal_v1')==trained_job,'New model metadata differs')
        m=arch.build_model('R1');m.load_state_dict(cp['model_state'],strict=True)
        m.to(device).eval().requires_grad_(False);models[name]=m;model_hashes[name]=parameters_sha(m)
        probes[name]=pa.make_probe(HierarchicalSensorView(m),masks,lo,hi)
        require(probes[name].projection_epsilon_f==.03 and probes[name].projection_iters==10 and probes[name].branch_ascent_steps==1,'Legacy solver altered')
        del cp
    starts=list(jsonl(out/'starts.jsonl'));mine=starts[rank::job['cfg']['world']]
    folder=out/'evaluation';folder.mkdir(exist_ok=True)
    final=folder/f'rank{rank}';require(not final.exists(),'Evaluation shard already exists')
    tmp=Path(tempfile.mkdtemp(prefix=f'.rank{rank}.building.',dir=folder))
    if mine:
        sample=mine[0];x=torch.tensor(sample['x'],device=device);q=torch.tensor(sample['q_init'],device=device)
        for _ in range(2):
            for name in ARMS:pa.run_probe(probes[name],oracle,x,q,int(sample['sensor']))
    file=tmp/'solves.jsonl'
    with file.open('x') as f:
        for k,start in enumerate(mine):
            x=torch.tensor(start['x'],device=device,dtype=torch.float32)
            q=torch.tensor(start['q_init'],device=device,dtype=torch.float32);s=int(start['sensor'])
            qp=torch.tensor(start['witness_q'],device=device,dtype=torch.float32)
            require(oracle.value(x,qp,s)>job['cfg']['side_margin_m'],'Visible witness not reproduced by runtime oracle')
            row={'case_id':start['case_id'],'start':start,'models':{}}
            # Balanced order on each rank; no model-specific geometry regeneration.
            order=ARMS if k%2==0 else ARMS[::-1]
            for name in order:row['models'][name]=pa.run_probe(probes[name],oracle,x,q,s)
            from common import dumps
            f.write(dumps(row)+'\n')
            if (k+1)%100==0:
                f.flush();print(f'[eval rank{rank}] {k+1}/{len(mine)}',flush=True)
        f.flush();os.fsync(f.fileno())
    for name,m in models.items():require(parameters_sha(m)==model_hashes[name],'Frozen model mutated')
    write(tmp/'complete.json',{'status':'COMPLETE','case_count':len(mine),'starts_sha256':sha(out/'starts.jsonl'),
        'checkpoint_sha256':job['checkpoint_sha256'],'parameter_sha256':model_hashes,'warmup_solver_calls_per_model':2 if mine else 0,
        'files':{'solves.jsonl':sha(file)}})
    os.rename(tmp,final)
    print(f'[eval rank{rank}] COMPLETE cases={len(mine)}',flush=True)


def run_parallel(job,stage):
    out=Path(job['out']);world=job['cfg']['world'];processes=[];handles=[]
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    require(len(gpu)==world,'Four GPU identifiers required')
    try:
        for rank in range(world):
            target=out/('geometry' if stage=='generate' else 'evaluation')/f'rank{rank}'
            if target.exists():
                stage_complete(target,'complete.json');print(f'[reuse] {stage} rank{rank}: verified complete shard',flush=True);continue
            log=(out/'logs'/f'{stage}_rank{rank}_{os.getpid()}.log').open('x');handles.append(log)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='' if stage=='generate' else gpu[rank]
            env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
            p=subprocess.Popen([sys.executable,'-u',str(HERE/'run.py'),stage,'--job',str(out/'job.json'),'--rank',str(rank)],
                               stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            processes.append((p,rank,log.name))
        while processes:
            for p,r,log in list(processes):
                rc=p.poll()
                if rc is None:continue
                require(rc==0,f'{stage} rank{r} failed ({rc}); read {log}')
                processes.remove((p,r,log))
            if processes:time.sleep(.4)
    finally:
        for p,_,_ in processes:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for p,_,_ in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
        for h in handles:h.close()


def worker(job):
    import torch
    out=Path(job['out']);verify_sources(job)
    ready=read(out/'preflight_complete.json')
    require(ready.get('status')=='COMPLETE' and ready.get('job_sha256')==sha(out/'job.json'),'Preflight gate missing or job changed')
    require(torch.cuda.is_available() and torch.cuda.device_count()==4,'Formal job requires four GPUs')
    (out/'logs').mkdir(exist_ok=True)
    def stop(*_):raise KeyboardInterrupt('Job interrupted; completed shards are reusable')
    signal.signal(signal.SIGTERM,stop)
    if not (out/'inputs_frozen.json').exists():run_parallel(job,'generate')
    freeze(job);run_parallel(job,'evaluate');verify_sources(job)
    for name,path in job['checkpoints'].items():require(sha(path)==job['checkpoint_sha256'][name],'Checkpoint changed during job')
    from report import merge
    if not (out/'complete.json').exists():merge(job)
    else:stage_complete(out,'complete.json')
    print('[COMPLETE] off-grid test only; no weights or training data modified',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','generate','evaluate','freeze','worker','verify'))
    p.add_argument('--job',type=Path);p.add_argument('--repo',type=Path);p.add_argument('--sgroot',type=Path)
    p.add_argument('--cache',type=Path);p.add_argument('--out',type=Path);p.add_argument('--rank',type=int,default=0)
    a=p.parse_args()
    if a.mode=='prepare':
        require(all(v is not None for v in (a.repo,a.sgroot,a.cache,a.out)),'prepare requires repo,sgroot,cache,out')
        job=preflight(a.repo,a.sgroot,a.cache,a.out)
        # Real-input CPU check BEFORE GPU allocation, including exact source Jacobian.
        from geometry import ExactGeometry
        exact=ExactGeometry(job);x=read(Path(job['out'])/'points.json')['points'][0]
        checks=[exact.check_parity(x,(np.asarray(job['lo'])+job['hi'])/2,s) for s in range(8)]
        write(Path(job['out'])/'preflight_complete.json',{'status':'COMPLETE','geometry':checks,'job_sha256':sha(Path(job['out'])/'job.json')})
        print('[preflight] actual URDF FOV/Jacobian parity PASS; four-GPU formal job ready',flush=True);return
    require(a.job is not None,'--job required');job=read(a.job);require(0<=a.rank<job['cfg']['world'],'Bad rank')
    if a.mode=='generate':
        from geometry import generate
        generate(job,a.rank)
    elif a.mode=='evaluate':evaluate(job,a.rank)
    elif a.mode=='freeze':freeze(job)
    elif a.mode=='worker':worker(job)
    else:verify_sources(job);stage_complete(Path(job['out']),'complete.json');print('[VERIFIED] complete off-grid report')

if __name__=='__main__':main()
